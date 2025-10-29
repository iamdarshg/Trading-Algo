import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import threading
import time
from queue import Queue, Empty
from dataclasses import dataclass
import pickle
import os

# Absolute imports
from trading_bot.trainer import TradingTrainer, TradingLoss
from trading_bot.data_processor import TradingDataProcessor
from trading_bot.optimized_inference import OptimizedInferenceManager

@dataclass
class TrainingBatch:
    """Single training batch for online learning"""
    price_data: torch.Tensor
    target: torch.Tensor
    text_data: Optional[torch.Tensor] = None
    timestamp: datetime = None
    symbol: str = ""
    confidence: float = 1.0

class ExperienceReplay:
    """Experience replay buffer for online learning"""
    
    def __init__(self, max_size: int = 10000, priority_sampling: bool = True):
        self.max_size = max_size
        self.priority_sampling = priority_sampling
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = 0.6  # Prioritization exponent
        self.beta = 0.4   # Importance sampling exponent
        
    def add(self, batch: TrainingBatch, td_error: float = 1.0):
        """Add experience to replay buffer"""
        self.buffer.append(batch)
        # Priority based on TD error (higher error = higher priority)
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.priorities.append(priority)
    
    def sample(self, batch_size: int = 32) -> List[TrainingBatch]:
        """Sample batch from experience replay"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if self.priority_sampling and len(self.priorities) > 0:
            # Priority sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            
            indices = np.random.choice(
                len(self.buffer), 
                size=min(batch_size, len(self.buffer)), 
                replace=False, 
                p=probabilities
            )
        else:
            # Uniform sampling
            indices = np.random.choice(
                len(self.buffer), 
                size=min(batch_size, len(self.buffer)), 
                replace=False
            )
        
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class OnlineLearningManager:
    """Manages online learning for trading models during live operation"""
    
    def __init__(self, 
                 model: nn.Module,
                 data_processor: TradingDataProcessor,
                 learning_rate: float = 1e-5,
                 replay_buffer_size: int = 10000,
                 min_samples_for_update: int = 100,
                 update_frequency: int = 3600):  # Update every hour
        
        self.model = model
        self.data_processor = data_processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Online learning components
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        self.criterion = TradingLoss()
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        
        # Experience replay
        self.replay_buffer = ExperienceReplay(max_size=replay_buffer_size)
        self.min_samples_for_update = min_samples_for_update
        self.update_frequency = update_frequency
        
        # Performance tracking
        self.online_metrics = {
            'updates_count': 0,
            'total_loss': 0.0,
            'avg_loss': 0.0,
            'last_update': None,
            'learning_rate': learning_rate,
            'performance_history': []
        }
        
        # Background learning thread
        self.learning_queue = Queue()
        self.learning_thread = None
        self.stop_learning = threading.Event()
        self.is_learning = False
        
        # Model checkpointing
        self.checkpoint_dir = "online_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def start_online_learning(self):
        """Start background online learning thread"""
        if not self.is_learning:
            self.stop_learning.clear()
            self.learning_thread = threading.Thread(target=self._learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
            self.is_learning = True
            print("Online learning started")
    
    def stop_online_learning(self):
        """Stop background online learning"""
        if self.is_learning:
            self.stop_learning.set()
            if self.learning_thread:
                self.learning_thread.join(timeout=5.0)
            self.is_learning = False
            print("Online learning stopped")
    
    def add_experience(self, price_data: np.ndarray, actual_return: float, 
                      predicted_return: float, symbol: str = "", 
                      text_data: Optional[np.ndarray] = None):
        """Add new trading experience to the replay buffer"""
        
        # Calculate TD error (temporal difference error)
        td_error = abs(actual_return - predicted_return)
        
        # Create training batch
        price_tensor = torch.FloatTensor(price_data).unsqueeze(0)  # Add batch dim
        target_tensor = torch.FloatTensor([actual_return])
        text_tensor = torch.LongTensor(text_data).unsqueeze(0) if text_data is not None else None
        
        batch = TrainingBatch(
            price_data=price_tensor,
            target=target_tensor,
            text_data=text_tensor,
            timestamp=datetime.now(),
            symbol=symbol,
            confidence=1.0 / (1.0 + td_error)  # Higher confidence for lower error
        )
        
        # Add to replay buffer
        self.replay_buffer.add(batch, td_error)
        
        # Queue for immediate processing if urgent
        if td_error > 0.1:  # High error threshold
            self.learning_queue.put(('urgent_update', batch))
    
    def _learning_loop(self):
        """Main online learning loop running in background thread"""
        last_update_time = time.time()
        
        while not self.stop_learning.is_set():
            try:
                # Check for urgent updates
                try:
                    item = self.learning_queue.get(timeout=1.0)
                    if item[0] == 'urgent_update':
                        self._perform_urgent_update(item[1])
                except Empty:
                    pass
                
                # Regular periodic updates
                current_time = time.time()
                if (current_time - last_update_time) >= self.update_frequency:
                    if len(self.replay_buffer) >= self.min_samples_for_update:
                        self._perform_batch_update()
                        last_update_time = current_time
                        
                        # Save checkpoint periodically
                        if self.online_metrics['updates_count'] % 10 == 0:
                            self._save_checkpoint()
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Error in online learning loop: {e}")
                time.sleep(5.0)  # Wait before retrying
    
    def _perform_urgent_update(self, urgent_batch: TrainingBatch):
        """Perform immediate update for high-error experiences"""
        self.model.train()
        
        # Move data to device
        price_data = urgent_batch.price_data.to(self.device)
        target = urgent_batch.target.to(self.device)
        text_data = urgent_batch.text_data.to(self.device) if urgent_batch.text_data is not None else None
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if text_data is not None:
            prediction = self.model(price_data, text_data)
        else:
            prediction = self.model(price_data)
        
        # Calculate loss with higher weight for urgent updates
        loss = self.criterion(prediction.squeeze(), target) * 2.0  # Double weight
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update metrics
        self.online_metrics['updates_count'] += 1
        self.online_metrics['total_loss'] += loss.item()
        self.online_metrics['last_update'] = datetime.now()
        
        print(f"Urgent update performed: loss={loss.item():.6f}")
    
    def _perform_batch_update(self):
        """Perform batch update using experience replay"""
        self.model.train()
        
        batch_size = min(32, len(self.replay_buffer) // 4)
        sampled_batches = self.replay_buffer.sample(batch_size)
        
        if not sampled_batches:
            return
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in sampled_batches:
            # Move data to device
            price_data = batch.price_data.to(self.device)
            target = batch.target.to(self.device)
            text_data = batch.text_data.to(self.device) if batch.text_data is not None else None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if text_data is not None:
                prediction = self.model(price_data, text_data)
            else:
                prediction = self.model(price_data)
            
            # Calculate weighted loss (based on confidence)
            loss = self.criterion(prediction.squeeze(), target) * batch.confidence
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Update learning rate
        self.scheduler.step()
        
        # Update metrics
        avg_loss = total_loss / max(num_batches, 1)
        self.online_metrics['updates_count'] += 1
        self.online_metrics['total_loss'] += total_loss
        self.online_metrics['avg_loss'] = avg_loss
        self.online_metrics['last_update'] = datetime.now()
        self.online_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        # Track performance history
        self.online_metrics['performance_history'].append({
            'timestamp': datetime.now(),
            'loss': avg_loss,
            'samples_processed': num_batches,
            'buffer_size': len(self.replay_buffer)
        })
        
        # Keep only last 1000 performance records
        if len(self.online_metrics['performance_history']) > 1000:
            self.online_metrics['performance_history'] = self.online_metrics['performance_history'][-1000:]
        
        print(f"Batch update completed: avg_loss={avg_loss:.6f}, lr={self.online_metrics['learning_rate']:.8f}")
    
    def _save_checkpoint(self):
        """Save model checkpoint with online learning state"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'online_metrics': self.online_metrics,
            'replay_buffer_size': len(self.replay_buffer),
            'timestamp': datetime.now()
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"online_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        )
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save replay buffer separately (it can be large)
        buffer_path = os.path.join(
            self.checkpoint_dir, 
            f"replay_buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        with open(buffer_path, 'wb') as f:
            pickle.dump(self.replay_buffer, f)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, buffer_path: Optional[str] = None):
        """Load model checkpoint and resume online learning"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.online_metrics = checkpoint['online_metrics']
        
        if buffer_path and os.path.exists(buffer_path):
            with open(buffer_path, 'rb') as f:
                self.replay_buffer = pickle.load(f)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive online learning statistics"""
        stats = self.online_metrics.copy()
        
        # Calculate additional metrics
        if self.online_metrics['updates_count'] > 0:
            stats['avg_loss_per_update'] = self.online_metrics['total_loss'] / self.online_metrics['updates_count']
        
        stats['buffer_utilization'] = len(self.replay_buffer) / self.replay_buffer.max_size
        stats['is_learning'] = self.is_learning
        
        # Recent performance trend
        if len(self.online_metrics['performance_history']) >= 10:
            recent_losses = [p['loss'] for p in self.online_metrics['performance_history'][-10:]]
            stats['recent_avg_loss'] = np.mean(recent_losses)
            stats['loss_trend'] = 'improving' if recent_losses[-1] < recent_losses[0] else 'degrading'
        
        return stats
    
    def adapt_learning_rate(self, performance_metric: float, target_performance: float = 0.02):
        """Dynamically adapt learning rate based on performance"""
        if performance_metric > target_performance * 1.5:  # Poor performance
            # Increase learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.1, 1e-3)
        elif performance_metric < target_performance * 0.5:  # Good performance
            # Decrease learning rate for stability
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.95, 1e-7)
        
        self.online_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
    
    def reset_learning(self):
        """Reset online learning state (useful for market regime changes)"""
        self.replay_buffer = ExperienceReplay(max_size=self.replay_buffer.max_size)
        
        self.online_metrics = {
            'updates_count': 0,
            'total_loss': 0.0,
            'avg_loss': 0.0,
            'last_update': None,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'performance_history': []
        }
        
        print("Online learning state reset")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_online_learning()

class AdaptiveTradingStrategy:
    """Trading strategy with online learning capabilities"""
    
    def __init__(self, model: nn.Module, data_processor: TradingDataProcessor, 
                 inference_manager: OptimizedInferenceManager):
        self.model = model
        self.data_processor = data_processor
        self.inference_manager = inference_manager
        
        # Initialize online learning
        self.online_learner = OnlineLearningManager(
            model=model,
            data_processor=data_processor
        )
        
        # Performance tracking for adaptive behavior
        self.prediction_history = deque(maxlen=1000)
        self.actual_returns = deque(maxlen=1000)
        
    def start_adaptive_learning(self):
        """Start online learning for the strategy"""
        self.online_learner.start_online_learning()
    
    def stop_adaptive_learning(self):
        """Stop online learning"""
        self.online_learner.stop_online_learning()
    
    def predict_with_learning(self, price_data: np.ndarray, text_data: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Make prediction and potentially trigger learning"""
        # Convert to tensors
        price_tensor = torch.FloatTensor(price_data).unsqueeze(0)
        text_tensor = torch.LongTensor(text_data).unsqueeze(0) if text_data is not None else None
        
        # Make prediction using optimized inference
        prediction, confidence = self.inference_manager.predict(price_tensor, text_tensor)
        
        # Store prediction for future learning
        self.prediction_history.append((prediction, datetime.now()))
        
        return prediction, confidence
    
    def update_with_actual_outcome(self, actual_return: float, symbol: str = ""):
        """Update model with actual trading outcome"""
        if not self.prediction_history:
            return
        
        # Get corresponding prediction
        predicted_return, prediction_time = self.prediction_history[-1]
        
        # Store actual return
        self.actual_returns.append(actual_return)
        
        # Calculate the features that were used for this prediction
        # This is simplified - in practice, you'd store the exact features
        # For now, we'll use dummy features
        dummy_price_data = np.random.randn(60, 20)  # Placeholder
        
        # Add experience to online learner
        self.online_learner.add_experience(
            price_data=dummy_price_data,
            actual_return=actual_return,
            predicted_return=predicted_return,
            symbol=symbol
        )
        
        # Adaptive learning rate adjustment
        if len(self.actual_returns) >= 10:
            recent_error = np.mean([
                abs(pred[0] - actual) 
                for pred, actual in zip(list(self.prediction_history)[-10:], list(self.actual_returns)[-10:])
            ])
            self.online_learner.adapt_learning_rate(recent_error)
    
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get metrics about the adaptive learning performance"""
        base_metrics = self.online_learner.get_learning_statistics()
        
        # Add strategy-specific metrics
        if len(self.prediction_history) > 0 and len(self.actual_returns) > 0:
            predictions = [p[0] for p in list(self.prediction_history)[-len(self.actual_returns):]]
            actual = list(self.actual_returns)
            
            if len(predictions) == len(actual):
                mse = np.mean([(p - a) ** 2 for p, a in zip(predictions, actual)])
                mae = np.mean([abs(p - a) for p, a in zip(predictions, actual)])
                
                base_metrics.update({
                    'prediction_mse': mse,
                    'prediction_mae': mae,
                    'total_predictions': len(self.prediction_history),
                    'adaptation_active': self.online_learner.is_learning
                })
        
        return base_metrics