
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradingDataset(Dataset):
    """Custom dataset for trading data with text prompts"""

    def __init__(self, 
                 price_data: np.ndarray,
                 targets: np.ndarray,
                 text_data: Optional[np.ndarray] = None):

        self.price_data = torch.FloatTensor(price_data)
        self.targets = torch.FloatTensor(targets)
        self.text_data = torch.LongTensor(text_data) if text_data is not None else None

    def __len__(self):
        return len(self.price_data)

    def __getitem__(self, idx):
        sample = {
            'price_data': self.price_data[idx],
            'target': self.targets[idx]
        }

        if self.text_data is not None:
            # Use the same text for all samples (or implement your own logic)
            sample['text_data'] = self.text_data[0]  # Simplified

        return sample

class TradingLoss(nn.Module):
    """Custom loss function for trading models"""

    def __init__(self, 
                 directional_weight: float = 1.0,
                 magnitude_weight: float = 1.0,
                 risk_penalty: float = 0.1):
        super().__init__()
        self.directional_weight = directional_weight
        self.magnitude_weight = magnitude_weight
        self.risk_penalty = risk_penalty

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MSE component for magnitude
        mse_loss = nn.MSELoss()(predictions, targets)

        # Directional accuracy component
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        directional_loss = 1 - torch.mean((pred_direction == target_direction).float())

        # Risk penalty for extreme predictions
        risk_loss = torch.mean(torch.abs(predictions) ** 2)

        total_loss = (self.magnitude_weight * mse_loss + 
                     self.directional_weight * directional_loss + 
                     self.risk_penalty * risk_loss)

        return total_loss

class EarlyStopping:
    """Early stopping utility"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best:
                self.load_checkpoint(model)
            return True
        return False

    def save_checkpoint(self, model: nn.Module):
        self.best_weights = model.state_dict().copy()

    def load_checkpoint(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class TradingTrainer:
    """Comprehensive training framework for trading models"""

    def __init__(self,
                 model: nn.Module,
                 device: str = 'auto',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 scheduler_type: str = 'cosine',
                 loss_function: str = 'trading'):

        self.model = model

        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function
        if loss_function == 'trading':
            self.criterion = TradingLoss()
        elif loss_function == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_function == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        # Learning rate scheduler
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            self.scheduler = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epoch_times': []
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move data to device
            price_data = batch['price_data'].to(self.device)
            targets = batch['target'].to(self.device)

            text_data = None
            if 'text_data' in batch:
                text_data = batch['text_data'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if text_data is not None:
                predictions = self.model(price_data, text_data)
            else:
                predictions = self.model(price_data)

            # Calculate loss
            loss = self.criterion(predictions.squeeze(), targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                price_data = batch['price_data'].to(self.device)
                targets = batch['target'].to(self.device)

                text_data = None
                if 'text_data' in batch:
                    text_data = batch['text_data'].to(self.device)

                # Forward pass
                if text_data is not None:
                    predictions = self.model(price_data, text_data)
                else:
                    predictions = self.model(price_data)

                loss = self.criterion(predictions.squeeze(), targets)
                total_loss += loss.item()

                # Store predictions and targets for metrics
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        # Calculate metrics
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)

        # Directional accuracy
        pred_directions = np.sign(predictions_np)
        target_directions = np.sign(targets_np)
        directional_accuracy = np.mean(pred_directions == target_directions)

        # Mean absolute error
        mae = np.mean(np.abs(predictions_np - targets_np))

        # Root mean squared error
        rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))

        metrics = {
            'directional_accuracy': directional_accuracy,
            'mae': mae,
            'rmse': rmse
        }

        return avg_loss, metrics

    def fit(self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            epochs: int = 100,
            early_stopping_patience: int = 15,
            save_best: bool = True,
            model_path: str = 'best_model.pth',
            verbose: bool = True) -> Dict[str, List]:

        """Train the model"""
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')

        if verbose:
            print(f"Training on device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print("="*60)

        for epoch in range(epochs):
            start_time = datetime.now()

            # Training
            train_loss = self.train_epoch(train_dataloader)

            # Validation
            val_loss, val_metrics = self.validate_epoch(val_dataloader)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            epoch_time = (datetime.now() - start_time).total_seconds()
            self.history['epoch_times'].append(epoch_time)

            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, model_path)

            # Verbose output
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Dir Acc: {val_metrics['directional_accuracy']:.3f} | "
                      f"Time: {epoch_time:.2f}s")

            # Early stopping
            if early_stopping(val_loss, self.model):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

        return self.history

def create_data_loaders(processed_data: Dict[str, np.ndarray],
                       encoded_news: Optional[List[np.ndarray]] = None,
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""

    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    X_val = processed_data['X_val']
    y_val = processed_data['y_val']

    # Create datasets
    if encoded_news is not None:
        train_dataset = TradingDataset(X_train, y_train, np.array(encoded_news))
        val_dataset = TradingDataset(X_val, y_val, np.array(encoded_news))
    else:
        train_dataset = TradingDataset(X_train, y_train)
        val_dataset = TradingDataset(X_val, y_val)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader

def train_model_pipeline(model_builder,
                        processed_data: Dict[str, np.ndarray],
                        encoded_news: Optional[List[np.ndarray]] = None,
                        training_config: Optional[Dict[str, Any]] = None) -> Tuple[nn.Module, Dict[str, List]]:
    """Complete model training pipeline"""

    # Default training config
    if training_config is None:
        training_config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'early_stopping_patience': 15,
            'scheduler_type': 'cosine',
            'loss_function': 'trading'
        }

    # Build model
    model = model_builder.build()

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        processed_data,
        encoded_news,
        batch_size=training_config['batch_size']
    )

    # Create trainer
    trainer = TradingTrainer(
        model=model,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        scheduler_type=training_config['scheduler_type'],
        loss_function=training_config['loss_function']
    )

    # Train model
    history = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=training_config['epochs'],
        early_stopping_patience=training_config['early_stopping_patience'],
        verbose=True
    )

    return model, history

# Configuration templates
def get_training_config_conservative() -> Dict[str, Any]:
    """Conservative training configuration"""
    return {
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 0.0005,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'scheduler_type': 'plateau',
        'loss_function': 'trading'
    }

def get_training_config_aggressive() -> Dict[str, Any]:
    """Aggressive training configuration"""
    return {
        'epochs': 200,
        'batch_size': 64,
        'learning_rate': 0.002,
        'weight_decay': 1e-6,
        'early_stopping_patience': 25,
        'scheduler_type': 'cosine',
        'loss_function': 'trading'
    }

def get_training_config_experimental() -> Dict[str, Any]:
    """Experimental training configuration"""
    return {
        'epochs': 150,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 5e-5,
        'early_stopping_patience': 20,
        'scheduler_type': 'cosine',
        'loss_function': 'trading'
    }
