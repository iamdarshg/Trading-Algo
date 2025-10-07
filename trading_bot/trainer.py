import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

def _assert_no_nan_inf(t: torch.Tensor, where: str = ""):
    if not torch.isfinite(t).all():
        bad = t[~torch.isfinite(t)]
        raise ValueError(f"NaN/Inf detected {where}: shape={t.shape}, count={bad.numel()}")

class TradingDataset(Dataset):
    def __init__(self, price_data: np.ndarray, targets: np.ndarray, text_data: Optional[np.ndarray] = None):
        self.price_data = torch.FloatTensor(price_data)
        self.targets = torch.FloatTensor(targets)
        self.text_data = torch.LongTensor(text_data) if text_data is not None else None
    def __len__(self):
        return len(self.price_data)
    def __getitem__(self, idx):
        sample = {'price_data': self.price_data[idx], 'target': self.targets[idx]}
        if self.text_data is not None:
            sample['text_data'] = self.text_data[0]
        return sample

class TradingLoss(nn.Module):
    def __init__(self, directional_weight: float = 1.0, magnitude_weight: float = 1.0, risk_penalty: float = 0.1):
        super().__init__()
        self.directional_weight = directional_weight
        self.magnitude_weight = magnitude_weight
        self.risk_penalty = risk_penalty
        self._mse = nn.MSELoss()
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _assert_no_nan_inf(predictions, "in TradingLoss predictions")
        _assert_no_nan_inf(targets, "in TradingLoss targets")
        mse_loss = self._mse(predictions, targets)
        _assert_no_nan_inf(mse_loss, "in TradingLoss mse")
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        directional_loss = 1 - torch.mean((pred_direction == target_direction).float())
        _assert_no_nan_inf(directional_loss, "in TradingLoss directional")
        risk_loss = torch.mean(torch.abs(predictions) ** 2)
        _assert_no_nan_inf(risk_loss, "in TradingLoss risk")
        total_loss = (self.magnitude_weight * mse_loss + self.directional_weight * directional_loss + self.risk_penalty * risk_loss)
        _assert_no_nan_inf(total_loss, "in TradingLoss total")
        return total_loss

class EarlyStopping:
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
    def __init__(self, model: nn.Module, device: str = 'auto', learning_rate: float = 1e-4, weight_decay: float = 1e-5, scheduler_type: str = 'cosine', loss_function: str = 'trading'):
        self.model = model
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self._init_weights_kaiming(self.model)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if loss_function == 'trading':
            self.criterion = TradingLoss()
        elif loss_function == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_function == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        else:
            self.scheduler = None
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': [], 'epoch_times': []}

    @staticmethod
    def _init_weights_kaiming(module: nn.Module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        nn.init.kaiming_normal_(param, nonlinearity='relu')
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _assert_batch_clean(self, batch: Dict[str, torch.Tensor], stage: str):
        for k in ['price_data', 'target']:
            if k in batch:
                _assert_no_nan_inf(batch[k], f"in {stage} batch[{k}]")
        if 'text_data' in batch and batch['text_data'] is not None:
            _assert_no_nan_inf(batch['text_data'].float(), f"in {stage} batch[text_data]")

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            self._assert_batch_clean(batch, stage='train')
            price_data = batch['price_data'].to(self.device)
            targets = batch['target'].to(self.device)
            text_data = batch.get('text_data')
            text_data = text_data.to(self.device) if text_data is not None else None
            self.optimizer.zero_grad()
            try:
                predictions = self.model(price_data, text_data) if text_data is not None else self.model(price_data)
                _assert_no_nan_inf(predictions, "after model forward (train)")
            except Exception as e:
                raise ValueError(f"Model forward produced NaN/Inf or error on train batch: {e}")
            loss = self.criterion(predictions.squeeze(), targets)
            _assert_no_nan_inf(loss, "after loss compute (train)")
            loss.backward()
            for name, p in self.model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise ValueError(f"NaN/Inf detected in gradients for parameter {name}")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_predictions: List[float] = []
        all_targets: List[float] = []
        with torch.no_grad():
            for batch in dataloader:
                self._assert_batch_clean(batch, stage='val')
                price_data = batch['price_data'].to(self.device)
                targets = batch['target'].to(self.device)
                text_data = batch.get('text_data')
                text_data = text_data.to(self.device) if text_data is not None else None
                try:
                    predictions = self.model(price_data, text_data) if text_data is not None else self.model(price_data)
                    _assert_no_nan_inf(predictions, "after model forward (val)")
                except Exception as e:
                    raise ValueError(f"Model forward produced NaN/Inf or error on val batch: {e}")
                loss = self.criterion(predictions.squeeze(), targets)
                _assert_no_nan_inf(loss, "after loss compute (val)")
                total_loss += float(loss.item())
                all_predictions.extend(predictions.squeeze().cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
        avg_loss = total_loss / max(len(dataloader), 1)
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)
        if predictions_np.size == 0:
            metrics = {'directional_accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0}
            return avg_loss, metrics
        pred_directions = np.sign(predictions_np)
        target_directions = np.sign(targets_np)
        directional_accuracy = float(np.mean(pred_directions == target_directions))
        mae = float(np.mean(np.abs(predictions_np - targets_np)))
        rmse = float(np.sqrt(np.mean((predictions_np - targets_np) ** 2)))
        metrics = {'directional_accuracy': directional_accuracy, 'mae': mae, 'rmse': rmse}
        return avg_loss, metrics

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int = 100, early_stopping_patience: int = 15, save_best: bool = True, model_path: str = 'best_model.pth', verbose: bool = True) -> Dict[str, List]:
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        if verbose:
            print(f"Training on device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print("="*60)
        for epoch in range(epochs):
            start_time = datetime.now()
            train_loss = self.train_epoch(train_dataloader)
            val_loss, val_metrics = self.validate_epoch(val_dataloader)
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            epoch_time = (datetime.now() - start_time).total_seconds()
            self.history['epoch_times'].append(epoch_time)
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'val_loss': val_loss, 'val_metrics': val_metrics}, model_path)
            if verbose and ((epoch + 1) % 10 == 0 or not np.isfinite(train_loss) or not np.isfinite(val_loss)):
                print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Dir Acc: {val_metrics['directional_accuracy']:.3f} | Time: {epoch_time:.2f}s")
            if early_stopping(val_loss, self.model):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break
        return self.history


def create_data_loaders(processed_data: Dict[str, np.ndarray], encoded_news: Optional[List[np.ndarray]] = None, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    X_val = processed_data['X_val']
    y_val = processed_data['y_val']
    if encoded_news is not None:
        train_dataset = TradingDataset(X_train, y_train, np.array(encoded_news))
        val_dataset = TradingDataset(X_val, y_val, np.array(encoded_news))
    else:
        train_dataset = TradingDataset(X_train, y_train)
        val_dataset = TradingDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def train_model_pipeline(model_builder, processed_data: Dict[str, np.ndarray], encoded_news: Optional[List[np.ndarray]] = None, training_config: Optional[Dict[str, Any]] = None, init_model: Optional[nn.Module] = None, init_state_dict: Optional[Dict[str, Any]] = None) -> Tuple[nn.Module, Dict[str, List]]:
    """Build (or reuse) a model from model_builder, optionally initialize from an existing model or state_dict,
    then train it using the provided processed_data and training_config.

    Parameters:
      - model_builder: ModelBuilder instance
      - processed_data: dict with X_train, y_train, X_val, y_val
      - encoded_news: optional list of encoded text features
      - training_config: training hyperparameters (epochs, batch_size, learning_rate...)
      - init_model: optional nn.Module to use as starting model (skips model_builder.build())
      - init_state_dict: optional state_dict to load into the built model (uses strict=False fallback)
    """
    if training_config is None:
        training_config = {'epochs': 100, 'batch_size': 32, 'learning_rate': 1e-4, 'weight_decay': 1e-5, 'early_stopping_patience': 15, 'scheduler_type': 'cosine', 'loss_function': 'trading'}

    # Use provided model instance if available, otherwise build from builder
    if init_model is not None:
        model = init_model
    else:
        model = model_builder.build()

    # If a state dict was provided, try to load it (try strict then non-strict)
    if init_state_dict is not None:
        try:
            # strict load to ensure exact architectural match
            model.load_state_dict(init_state_dict)
        except Exception as e_strict:
            # try a non-strict load to report missing/unexpected keys for diagnostics, but then fail
            try:
                res = model.load_state_dict(init_state_dict, strict=False)
                missing = getattr(res, 'missing_keys', None)
                unexpected = getattr(res, 'unexpected_keys', None)
            except Exception:
                missing = None
                unexpected = None
            msg_lines = [
                "Failed to load provided state_dict into model (strict load failed).",
                f"Original error: {e_strict}",
            ]
            if missing is not None or unexpected is not None:
                msg_lines.append(f"Missing keys: {missing}")
                msg_lines.append(f"Unexpected keys: {unexpected}")
            msg_lines.append("Possible fixes:\n - Ensure the saved model's architecture (model_config) matches the current builder.\n - If you intentionally changed the architecture, delete or move the saved model file and re-run training.\n - Re-save the model using the current ModelBuilder.get_config() so reconstruction is possible.")
            raise RuntimeError("\n".join(msg_lines))

    train_loader, val_loader = create_data_loaders(processed_data, encoded_news, batch_size=training_config['batch_size'])
    # Quick forward-pass validation to catch shape mismatches early
    try:
        model.eval()
        with torch.no_grad():
            # get a single batch
            for b in train_loader:
                price = b['price_data']
                text = b.get('text_data', None)
                # attempt forward
                _ = model(price.to(next(model.parameters()).device), text.to(next(model.parameters()).device) if text is not None else None)
                break
    except Exception as e_forward:
        # Fail early with an actionable message so the user can fix architecture/weights
        msg_lines = [
            "Model forward validation failed during initialization.",
            f"Error during forward pass: {e_forward}",
            "Possible causes:",
            " - The loaded state_dict does not match the model architecture (shapes/parameter names differ).",
            " - The ModelBuilder used to reconstruct the model is incompatible with the weights in the saved file.",
            "Actions to fix:",
            " 1) Ensure the saved model file includes a matching 'model_config' and that ModelBuilder.load_config(payload['model_config']) recreates the exact architecture.",
            " 2) If you intentionally changed the architecture, delete or move the saved model file so training starts fresh.",
            " 3) Rebuild/save the model with the current code and call save_model_and_config so future loads will match.",
            " 4) If you want to force a partial load (not recommended), implement a controlled migration that maps keys and shapes.",
        ]
        raise RuntimeError("\n".join(msg_lines))

    trainer = TradingTrainer(model=model, learning_rate=training_config['learning_rate'], weight_decay=training_config['weight_decay'], scheduler_type=training_config['scheduler_type'], loss_function=training_config['loss_function'])
    history = trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader, epochs=training_config['epochs'], early_stopping_patience=training_config['early_stopping_patience'], verbose=True)
    return model, history


def get_training_config_conservative() -> Dict[str, Any]:
    return {'epochs': 50, 'batch_size': 16, 'learning_rate': 1e-4, 'weight_decay': 1e-4, 'early_stopping_patience': 10, 'scheduler_type': 'plateau', 'loss_function': 'trading'}

def get_training_config_aggressive() -> Dict[str, Any]:
    return {'epochs': 200, 'batch_size': 64, 'learning_rate': 1e-4, 'weight_decay': 1e-6, 'early_stopping_patience': 25, 'scheduler_type': 'cosine', 'loss_function': 'trading'}

def get_training_config_experimental() -> Dict[str, Any]:
    return {'epochs': 150, 'batch_size': 32, 'learning_rate': 1e-4, 'weight_decay': 5e-5, 'early_stopping_patience': 20, 'scheduler_type': 'cosine', 'loss_function': 'trading'}
