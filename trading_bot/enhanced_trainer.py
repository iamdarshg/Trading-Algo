import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time
from dataclasses import dataclass

# Absolute imports
from trading_bot.trainer import TradingTrainer, TradingLoss, EarlyStopping
from trading_bot.enhanced_data_processor import EnhancedDataProcessor, HistoricalNewsProvider
from trading_bot.enhanced_text_processor import EnhancedTextProcessor
from trading_bot.model_builder import ModelBuilder
from config.config_manager import config_manager

@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
    # Basic training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Advanced training options
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Data augmentation
    use_data_augmentation: bool = True
    noise_level: float = 0.01
    temporal_jitter: bool = True
    
    # Learning rate scheduling
    scheduler_type: str = 'cosine_with_restarts'
    warmup_epochs: int = 10
    
    # Early stopping and validation
    early_stopping_patience: int = 15
    validation_frequency: int = 5
    
    # News integration
    include_news: bool = True
    news_lookback_days: int = 7
    news_weight: float = 0.3
    
    # Distributed training
    use_distributed: bool = False
    num_workers: int = 4
    
    # Model selection and ensemble
    use_ensemble: bool = False
    ensemble_size: int = 3
    
    # Regularization
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    label_smoothing: float = 0.0

class EnhancedTradingDataset(Dataset):
    """Enhanced dataset with data augmentation and news integration"""
    
    def __init__(self, 
                 price_data: np.ndarray,
                 targets: np.ndarray,
                 news_features: Optional[np.ndarray] = None,
                 text_data: Optional[List[str]] = None,
                 augment: bool = False,
                 noise_level: float = 0.01):
        
        self.price_data = torch.FloatTensor(price_data)
        self.targets = torch.FloatTensor(targets)
        self.news_features = torch.FloatTensor(news_features) if news_features is not None else None
        self.text_data = text_data
        self.augment = augment
        self.noise_level = noise_level
    
    def __len__(self):
        return len(self.price_data)
    
    def __getitem__(self, idx):
        price = self.price_data[idx]
        target = self.targets[idx]
        
        # Data augmentation
        if self.augment:
            # Add Gaussian noise
            noise = torch.randn_like(price) * self.noise_level
            price = price + noise
            
            # Temporal jittering (slight time shifts)
            if price.dim() == 2 and price.size(0) > 5:  # [seq_len, features]
                shift = np.random.randint(-2, 3)
                if shift != 0:
                    if shift > 0:
                        price = torch.cat([price[shift:], price[-shift:]], dim=0)
                    else:
                        price = torch.cat([price[:shift], price[:abs(shift)]], dim=0)
        
        sample = {
            'price_data': price,
            'target': target
        }
        
        # Add news features if available
        if self.news_features is not None:
            sample['news_features'] = self.news_features[idx]
        
        # Add text data if available
        if self.text_data is not None:
            sample['text_data'] = self.text_data[idx]
        
        return sample

class EnhancedTrainer:
    """Enhanced trainer with advanced features and news integration"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 device: str = 'auto'):
        
        self.model = model
        self.config = config
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Initialize training components
        self._init_optimizer()
        self._init_criterion()
        self._init_scheduler()
        
        # Mixed precision training
        if self.config.use_mixed_precision and hasattr(torch.cuda, 'amp'):
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
        
        # Training state
        self.current_epoch = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epoch_times': [],
            'news_contribution': [],
            'gradient_norms': []
        }
        
        # News processor
        self.news_processor = HistoricalNewsProvider()
        self.text_processor = EnhancedTextProcessor()
        
    def _init_optimizer(self):
        """Initialize optimizer with advanced settings"""
        # Use AdamW with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _init_criterion(self):
        """Initialize loss function"""
        self.criterion = TradingLoss(
            directional_weight=1.0,
            magnitude_weight=1.0,
            risk_penalty=0.1
        )
        
        # Label smoothing if specified
        if self.config.label_smoothing > 0:
            self.label_smoother = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        if self.config.scheduler_type == 'cosine_with_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=20,
                T_mult=2,
                eta_min=1e-7
            )
        elif self.config.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-7
            )
        elif self.config.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.early_stopping_patience // 2,
                factor=0.5
            )
        else:
            self.scheduler = None
    
    def create_data_loaders(self, 
                           processed_data: Dict[str, np.ndarray],
                           symbol: str,
                           news_data: Optional[Dict[str, Any]] = None) -> Tuple[DataLoader, DataLoader]:
        """Create enhanced data loaders with news integration"""
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # Prepare news features if available
        news_train = None
        news_val = None
        text_train = None
        text_val = None
        
        if self.config.include_news and news_data:
            if 'news_features_train' in news_data:
                news_train = news_data['news_features_train']
                news_val = news_data['news_features_val']
            
            if 'text_data' in news_data:
                # Split text data according to train/val split
                text_data = news_data['text_data']
                split_idx = len(X_train)
                text_train = text_data[:split_idx]
                text_val = text_data[split_idx:]
        
        # Create datasets
        train_dataset = EnhancedTradingDataset(
            X_train, y_train,
            news_features=news_train,
            text_data=text_train,
            augment=self.config.use_data_augmentation,
            noise_level=self.config.noise_level
        )
        
        val_dataset = EnhancedTradingDataset(
            X_val, y_val,
            news_features=news_val,
            text_data=text_val,
            augment=False  # No augmentation for validation
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Enhanced training epoch with mixed precision and gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        total_news_contribution = 0.0
        total_gradient_norm = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            price_data = batch['price_data'].to(self.device, non_blocking=True)
            targets = batch['target'].to(self.device, non_blocking=True)
            
            news_features = None
            if 'news_features' in batch:
                news_features = batch['news_features'].to(self.device, non_blocking=True)
            
            text_data = batch.get('text_data', None)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self._forward_with_news(price_data, news_features, text_data)
                    loss = self.criterion(predictions.squeeze(), targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
            else:
                predictions = self._forward_with_news(price_data, news_features, text_data)
                loss = self.criterion(predictions.squeeze(), targets)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                total_gradient_norm += grad_norm.item()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Calculate news contribution (if applicable)
            if news_features is not None:
                # This is a placeholder - implement actual news contribution calculation
                news_contribution = torch.mean(torch.abs(news_features)).item()
                total_news_contribution += news_contribution
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_news_contribution = total_news_contribution / max(num_batches, 1)
        avg_gradient_norm = total_gradient_norm / max(num_batches // self.config.gradient_accumulation_steps, 1)
        
        return {
            'loss': avg_loss,
            'news_contribution': avg_news_contribution,
            'gradient_norm': avg_gradient_norm
        }
    
    def _forward_with_news(self, price_data: torch.Tensor, 
                          news_features: Optional[torch.Tensor] = None,
                          text_data: Optional[List[str]] = None) -> torch.Tensor:
        """Forward pass integrating news data"""
        # Basic price prediction
        if hasattr(self.model, 'forward') and text_data is not None:
            # Model supports text input
            try:
                # Encode text data if provided
                if isinstance(text_data, list) and len(text_data) > 0:
                    # This is simplified - you'd want to batch encode the text
                    encoded_text = self.text_processor.encode_text(text_data[0])
                    text_tensor = torch.LongTensor(encoded_text).unsqueeze(0).to(self.device)
                    predictions = self.model(price_data, text_tensor)
                else:
                    predictions = self.model(price_data)
            except:
                predictions = self.model(price_data)
        else:
            predictions = self.model(price_data)
        
        # Integrate news features if available
        if news_features is not None and self.config.news_weight > 0:
            # Simple linear combination - you could make this more sophisticated
            news_signal = torch.mean(news_features, dim=1, keepdim=True)  # [batch, 1]
            predictions = (1 - self.config.news_weight) * predictions + self.config.news_weight * news_signal
        
        return predictions
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Enhanced validation epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                price_data = batch['price_data'].to(self.device, non_blocking=True)
                targets = batch['target'].to(self.device, non_blocking=True)
                
                news_features = None
                if 'news_features' in batch:
                    news_features = batch['news_features'].to(self.device, non_blocking=True)
                
                text_data = batch.get('text_data', None)
                
                # Forward pass
                predictions = self._forward_with_news(price_data, news_features, text_data)
                loss = self.criterion(predictions.squeeze(), targets)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.squeeze().cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate additional metrics
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)
        
        # Directional accuracy
        pred_directions = np.sign(predictions_np)
        target_directions = np.sign(targets_np)
        directional_accuracy = np.mean(pred_directions == target_directions)
        
        # Mean absolute error
        mae = np.mean(np.abs(predictions_np - targets_np))
        
        # Correlation
        correlation = np.corrcoef(predictions_np, targets_np)[0, 1] if len(predictions_np) > 1 else 0
        
        return {
            'loss': avg_loss,
            'directional_accuracy': directional_accuracy,
            'mae': mae,
            'correlation': correlation
        }
    
    def fit(self, 
           train_loader: DataLoader, 
           val_loader: DataLoader,
           verbose: bool = True) -> Dict[str, List]:
        """Enhanced training loop"""
        
        if verbose:
            print(f"Training on device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Using mixed precision: {self.use_amp}")
            print("=" * 80)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation (if it's time)
            if (epoch + 1) % self.config.validation_frequency == 0:
                val_metrics = self.validate_epoch(val_loader)
            else:
                val_metrics = {'loss': float('inf')}  # Skip validation
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Record metrics
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['news_contribution'].append(train_metrics.get('news_contribution', 0))
            self.training_history['gradient_norms'].append(train_metrics.get('gradient_norm', 0))
            
            epoch_time = time.time() - epoch_start_time
            self.training_history['epoch_times'].append(epoch_time)
            
            # Logging
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                      f"Train Loss: {train_metrics['loss']:.6f} | "
                      f"Val Loss: {val_metrics['loss']:.6f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                      f"Time: {epoch_time:.2f}s")
                
                if 'directional_accuracy' in val_metrics:
                    print(f"                      Dir Acc: {val_metrics['directional_accuracy']:.3f} | "
                          f"MAE: {val_metrics['mae']:.6f} | "
                          f"Corr: {val_metrics['correlation']:.3f}")
            
            # Early stopping check
            if val_metrics['loss'] < float('inf'):
                if early_stopping(val_metrics['loss'], self.model):
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        return self.training_history
    
    def save_checkpoint(self, filepath: str, additional_info: Dict = None):
        """Save comprehensive training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'current_epoch': self.current_epoch,
            'config': self.config,
            'device': str(self.device)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_info:
            checkpoint['additional_info'] = additional_info
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.current_epoch = checkpoint.get('current_epoch', 0)
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint.get('additional_info', {})

def train_with_enhanced_pipeline(symbol: str, 
                               model_builder: ModelBuilder,
                               config: TrainingConfig,
                               data_processor: EnhancedDataProcessor,
                               period: str = "2y",
                               interval: str = "1d") -> Tuple[nn.Module, Dict[str, List], EnhancedTrainer]:
    """Complete enhanced training pipeline"""
    
    print(f"Starting enhanced training for {symbol}...")
    
    # Create enhanced dataset with news
    dataset = data_processor.create_enhanced_dataset(
        symbol=symbol,
        period=period,
        interval=interval,
        include_news=config.include_news
    )
    
    if not dataset:
        raise ValueError(f"Failed to create dataset for {symbol}")
    
    # Build model
    model = model_builder.build()
    
    # Create trainer
    trainer = EnhancedTrainer(model, config)
    
    # Create data loaders
    train_loader, val_loader = trainer.create_data_loaders(
        dataset['processed_data'],
        symbol,
        dataset if config.include_news else None
    )
    
    # Train model
    history = trainer.fit(train_loader, val_loader)
    
    print(f"Training complete for {symbol}")
    
    return model, history, trainer