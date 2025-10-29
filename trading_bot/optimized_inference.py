import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import time
from functools import lru_cache
import hashlib
import pickle
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# Absolute imports
from trading_bot.advanced_models import (
    LiquidTimeConstantLayer, PiecewiseLinearLayer, 
    SelectiveStateSpaceLayer, MemoryAugmentedLayer, TextEncoder
)
from trading_bot.model_builder import ModelBuilder

@dataclass
class InferenceCache:
    """Cache for inference results"""
    predictions: Dict[str, Tuple[float, float, float]] = None  # hash -> (prediction, confidence, timestamp)
    processed_data: Dict[str, Any] = None  # hash -> processed features
    ttl: float = 300.0  # 5 minutes TTL
    
    def __post_init__(self):
        if self.predictions is None:
            self.predictions = {}
        if self.processed_data is None:
            self.processed_data = {}
    
    def get_prediction(self, data_hash: str) -> Optional[Tuple[float, float]]:
        """Get cached prediction if still valid"""
        if data_hash in self.predictions:
            pred, conf, timestamp = self.predictions[data_hash]
            if time.time() - timestamp < self.ttl:
                return pred, conf
            else:
                del self.predictions[data_hash]
        return None
    
    def cache_prediction(self, data_hash: str, prediction: float, confidence: float):
        """Cache prediction result"""
        self.predictions[data_hash] = (prediction, confidence, time.time())
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, _, timestamp) in self.predictions.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.predictions[key]

class OptimizedModel(nn.Module):
    """Optimized version of trading model with quantization and pruning support"""
    
    def __init__(self, original_model: nn.Module, optimize_level: str = 'balanced'):
        super().__init__()
        self.original_model = original_model
        self.optimize_level = optimize_level  # 'fast', 'balanced', 'accurate'
        self.quantized = False
        
        # Apply optimizations based on level
        if optimize_level == 'fast':
            self._apply_aggressive_optimizations()
        elif optimize_level == 'balanced':
            self._apply_balanced_optimizations()
        # 'accurate' uses original model
    
    def _apply_aggressive_optimizations(self):
        """Apply aggressive optimizations for maximum speed"""
        # Quantize model
        self._quantize_model()
        # Simplify architecture
        self._simplify_layers()
    
    def _apply_balanced_optimizations(self):
        """Apply balanced optimizations (speed vs accuracy)"""
        # Light quantization
        self._quantize_model(qconfig='fbgemm')
        # Selective layer optimization
        self._optimize_heavy_layers()
    
    def _quantize_model(self, qconfig='fbgemm'):
        """Apply dynamic quantization"""
        try:
            if hasattr(torch.quantization, 'quantize_dynamic'):
                self.original_model = torch.quantization.quantize_dynamic(
                    self.original_model, {nn.Linear}, dtype=torch.qint8
                )
                self.quantized = True
        except Exception as e:
            print(f"Quantization failed: {e}, using original model")
    
    def _simplify_layers(self):
        """Simplify heavy layers for faster inference"""
        # Replace heavy layers with lighter alternatives
        for name, module in self.original_model.named_modules():
            if isinstance(module, MemoryAugmentedLayer):
                # Replace with simpler attention
                simplified = self._create_simplified_attention(
                    module.input_size, module.memory_size
                )
                setattr(self.original_model, name.split('.')[-1], simplified)
    
    def _optimize_heavy_layers(self):
        """Optimize heavy layers while preserving accuracy"""
        for module in self.original_model.modules():
            if isinstance(module, (LiquidTimeConstantLayer, SelectiveStateSpaceLayer)):
                # Reduce internal computation precision for speed
                module.float()
    
    def _create_simplified_attention(self, input_size: int, memory_size: int) -> nn.Module:
        """Create a simplified attention mechanism"""
        return nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size)
        )
    
    def forward(self, *args, **kwargs):
        return self.original_model(*args, **kwargs)

class BatchInferenceEngine:
    """Optimized batch inference engine for multiple symbols"""
    
    def __init__(self, model: nn.Module, max_batch_size: int = 8, device: str = 'auto'):
        self.model = model
        self.max_batch_size = max_batch_size
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Inference queue for batching
        self.inference_queue = Queue()
        self.result_callbacks = {}
        self.processing_thread = None
        self.stop_processing = threading.Event()
    
    def start_batch_processing(self):
        """Start background batch processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(target=self._batch_processor)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def stop_batch_processing(self):
        """Stop background batch processing"""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def _batch_processor(self):
        """Background thread for batch processing"""
        batch_data = []
        batch_ids = []
        
        while not self.stop_processing.is_set():
            try:
                # Collect batch
                while len(batch_data) < self.max_batch_size and not self.stop_processing.is_set():
                    try:
                        item = self.inference_queue.get(timeout=0.1)
                        batch_data.append(item['data'])
                        batch_ids.append(item['id'])
                    except:
                        break
                
                if batch_data:
                    # Process batch
                    results = self._process_batch(batch_data)
                    
                    # Return results to callbacks
                    for batch_id, result in zip(batch_ids, results):
                        if batch_id in self.result_callbacks:
                            callback = self.result_callbacks.pop(batch_id)
                            callback(result)
                    
                    # Clear batch
                    batch_data.clear()
                    batch_ids.clear()
                
            except Exception as e:
                print(f"Batch processing error: {e}")
    
    def _process_batch(self, batch_data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process a batch of data"""
        with torch.no_grad():
            # Stack batch data
            if all(isinstance(data, tuple) for data in batch_data):
                # Handle (price_data, text_data) tuples
                price_batch = torch.stack([item[0] for item in batch_data])
                text_batch = torch.stack([item[1] for item in batch_data]) if batch_data[0][1] is not None else None
                
                # Reshape for batch processing
                batch_size, seq_len, features = price_batch.shape
                price_batch = price_batch.view(batch_size, seq_len, features)
                
                # Model inference
                if text_batch is not None:
                    predictions = self.model(price_batch.to(self.device), text_batch.to(self.device))
                else:
                    predictions = self.model(price_batch.to(self.device))
            else:
                # Handle single tensor batch
                batch_tensor = torch.stack(batch_data)
                predictions = self.model(batch_tensor.to(self.device))
            
            return [pred.cpu() for pred in predictions]
    
    def predict_async(self, data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], callback=None) -> str:
        """Submit data for async batch prediction"""
        request_id = str(time.time()) + str(np.random.randint(10000))
        
        if callback:
            self.result_callbacks[request_id] = callback
        
        self.inference_queue.put({'id': request_id, 'data': data})
        return request_id
    
    def predict_sync(self, data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Synchronous prediction"""
        with torch.no_grad():
            if isinstance(data, tuple):
                price_data, text_data = data
                if text_data is not None:
                    return self.model(price_data.to(self.device), text_data.to(self.device))
                else:
                    return self.model(price_data.to(self.device))
            else:
                return self.model(data.to(self.device))

class OptimizedInferenceManager:
    """Main inference manager with caching and optimization"""
    
    def __init__(self, model: nn.Module, optimization_level: str = 'balanced'):
        self.original_model = model
        self.optimized_model = OptimizedModel(model, optimization_level)
        self.cache = InferenceCache()
        
        # Batch inference engine
        self.batch_engine = BatchInferenceEngine(self.optimized_model)
        self.batch_engine.start_batch_processing()
        
        # Performance metrics
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _hash_data(self, price_data: torch.Tensor, text_data: Optional[torch.Tensor] = None) -> str:
        """Create hash for caching"""
        price_hash = hashlib.md5(price_data.numpy().tobytes()).hexdigest()[:16]
        if text_data is not None:
            text_hash = hashlib.md5(text_data.numpy().tobytes()).hexdigest()[:16]
            return f"{price_hash}_{text_hash}"
        return price_hash
    
    def predict(self, price_data: torch.Tensor, text_data: Optional[torch.Tensor] = None, 
                use_cache: bool = True) -> Tuple[float, float]:
        """Make optimized prediction with caching"""
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            data_hash = self._hash_data(price_data, text_data)
            cached_result = self.cache.get_prediction(data_hash)
            if cached_result is not None:
                self.cache_hits += 1
                return cached_result
            self.cache_misses += 1
        
        # Make prediction
        with torch.no_grad():
            if text_data is not None:
                prediction = self.optimized_model(price_data, text_data)
            else:
                prediction = self.optimized_model(price_data)
            
            pred_value = float(prediction.cpu().numpy()[0])
            confidence = min(1.0, abs(pred_value) / 0.1)
            
            # Cache result
            if use_cache:
                self.cache.cache_prediction(data_hash, pred_value, confidence)
        
        # Record performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Cleanup cache periodically
        if len(self.inference_times) % 100 == 0:
            self.cache.cleanup_expired()
        
        return pred_value, confidence
    
    def predict_batch(self, batch_data: List[Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> List[Tuple[float, float]]:
        """Batch prediction for multiple symbols"""
        results = []
        
        # Use batch engine for optimal processing
        for price_data, text_data in batch_data:
            if text_data is not None:
                prediction = self.batch_engine.predict_sync((price_data, text_data))
            else:
                prediction = self.batch_engine.predict_sync(price_data)
            
            pred_value = float(prediction.cpu().numpy()[0])
            confidence = min(1.0, abs(pred_value) / 0.1)
            results.append((pred_value, confidence))
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'total_predictions': len(self.inference_times),
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'model_quantized': self.optimized_model.quantized
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'batch_engine'):
            self.batch_engine.stop_batch_processing()

def create_optimized_model(model_builder: ModelBuilder, optimization_level: str = 'balanced') -> OptimizedInferenceManager:
    """Factory function to create optimized inference manager"""
    model = model_builder.build()
    return OptimizedInferenceManager(model, optimization_level)

def benchmark_inference_speed(model: nn.Module, sample_data: torch.Tensor, iterations: int = 100) -> Dict[str, float]:
    """Benchmark inference speed"""
    # Original model
    model.eval()
    original_times = []
    
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(sample_data)
            original_times.append(time.time() - start)
    
    # Optimized model
    optimized_model = OptimizedModel(model, 'balanced')
    optimized_times = []
    
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = optimized_model(sample_data)
            optimized_times.append(time.time() - start)
    
    return {
        'original_avg_time': np.mean(original_times),
        'optimized_avg_time': np.mean(optimized_times),
        'speedup_factor': np.mean(original_times) / np.mean(optimized_times),
        'original_std': np.std(original_times),
        'optimized_std': np.std(optimized_times)
    }