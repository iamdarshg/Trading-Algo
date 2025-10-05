import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

class TradingModelBase(nn.Module, ABC):
    """Base class for all trading models with standardized interface"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    @abstractmethod
    def forward(self, x: torch.Tensor, text_prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

class LiquidTimeConstantLayer(nn.Module):
    """Liquid Time-Constant Network Layer for non-differentiable time series"""
    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        # Sensory weights
        self.sensory_w = nn.Linear(input_size, hidden_size)
        self.sensory_mu = nn.Linear(input_size, hidden_size)
        self.sensory_sigma = nn.Linear(input_size, hidden_size)
        # Inter-neuron weights
        self.inter_w = nn.Linear(hidden_size, hidden_size)
        self.inter_mu = nn.Linear(hidden_size, hidden_size)
        self.inter_sigma = nn.Linear(hidden_size, hidden_size)
        # Time constant modulation
        self.time_constant = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]
            # Calculate liquid time constants
            tau = torch.sigmoid(self.time_constant(torch.cat([xt, h], dim=1))) + 0.1
            # Sensory processing
            sensory_input = self.sensory_w(xt) * torch.sigmoid(self.sensory_mu(xt)) * torch.exp(self.sensory_sigma(xt))
            # Inter-neuron processing
            inter_input = self.inter_w(h) * torch.sigmoid(self.inter_mu(h)) * torch.exp(self.inter_sigma(h))
            # Differential equation update
            dh_dt = (sensory_input + inter_input - h) / tau
            h = h + self.dt * dh_dt
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)

class PiecewiseLinearLayer(nn.Module):
    """Piecewise Linear Layer for handling discontinuities"""
    def __init__(self, input_size: int, output_size: int, num_pieces: int = 8):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_pieces = num_pieces
        # Breakpoints for piecewise linear functions
        self.breakpoints = nn.Parameter(torch.linspace(-3, 3, num_pieces))
        # Slopes and intercepts for each piece
        self.slopes = nn.Parameter(torch.randn(num_pieces, input_size, output_size))
        self.intercepts = nn.Parameter(torch.randn(num_pieces, output_size))
        # Gating network to determine which piece to use
        hidden_size = min(64, input_size * 2)
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_pieces),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.shape
        x_flat = x.view(-1, input_size)
        # Determine which piece to use for each input
        piece_weights = self.gate(x_flat)  # [batch*seq, num_pieces]
        # Compute output for each piece
        piece_outputs = []
        for i in range(self.num_pieces):
            linear_out = torch.matmul(x_flat, self.slopes[i]) + self.intercepts[i]
            piece_outputs.append(linear_out)
        piece_outputs = torch.stack(piece_outputs, dim=1)  # [batch*seq, num_pieces, output_size]
        # Weighted combination of pieces
        output = torch.sum(piece_weights.unsqueeze(-1) * piece_outputs, dim=1)
        return output.view(batch_size, seq_len, self.output_size)

class SelectiveStateSpaceLayer(nn.Module):
    """Mamba-style Selective State Space Layer"""
    def __init__(self, input_size: int, state_size: int, dt_rank: int = 16):
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.dt_rank = dt_rank
        # Selective mechanism parameters
        self.dt_proj = nn.Linear(dt_rank, input_size)
        self.A_log = nn.Parameter(torch.randn(input_size, state_size))
        self.D = nn.Parameter(torch.randn(input_size))
        # Input projections
        self.x_proj = nn.Linear(input_size, dt_rank + state_size * 2)
        self.out_proj = nn.Linear(state_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.shape
        # Project input to get dt, B, C
        x_dbl = self.x_proj(x)  # [batch, seq, dt_rank + 2*state_size]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.state_size, self.state_size], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [batch, seq, input_size]
        A = -torch.exp(self.A_log.float())  # [input_size, state_size]
        # Selective scan
        h = torch.zeros(batch_size, self.state_size).to(x.device)
        outputs = []
        for t in range(seq_len):
            # Discretize continuous parameters
            dt_t = dt[:, t, :].unsqueeze(-1)  # [batch, input_size, 1]
            # Correct broadcasting: A -> [1, input_size, state_size]
            A_discrete = torch.exp(A.unsqueeze(0) * dt_t)  # [batch, input_size, state_size]
            B_discrete = B[:, t, :].unsqueeze(1) * dt_t  # [batch, input_size, state_size]
            # State update
            h = A_discrete.sum(dim=1) * h + (B_discrete * x[:, t, :].unsqueeze(-1)).sum(dim=1)
            # Output
            y = torch.sum(C[:, t, :].unsqueeze(1) * h.unsqueeze(1), dim=-1)
            y = y + self.D * x[:, t, :]
            outputs.append(y.unsqueeze(1))
        return torch.cat(outputs, dim=1)

class MemoryAugmentedLayer(nn.Module):
    """Memory-Augmented Layer for long-term dependencies"""
    def __init__(self, input_size: int, memory_size: int, num_heads: int = 8):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        # Memory bank
        self.memory = nn.Parameter(torch.randn(memory_size, input_size))
        # Attention mechanisms
        self.query_proj = nn.Linear(input_size, input_size)
        self.key_proj = nn.Linear(input_size, input_size)
        self.value_proj = nn.Linear(input_size, input_size)
        # Memory update mechanism
        self.update_gate = nn.Linear(input_size * 2, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.shape
        outputs = []
        current_memory = self.memory.unsqueeze(0).repeat(batch_size, 1, 1)
        for t in range(seq_len):
            xt = x[:, t, :]  # [batch, input_size]
            # Attention over memory
            q = self.query_proj(xt).unsqueeze(1)  # [batch, 1, input_size]
            k = self.key_proj(current_memory)  # [batch, memory_size, input_size]
            v = self.value_proj(current_memory)  # [batch, memory_size, input_size]
            attention_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(input_size), dim=-1)
            memory_output = torch.matmul(attention_weights, v).squeeze(1)  # [batch, input_size]
            # Update memory
            update_signal = torch.sigmoid(self.update_gate(torch.cat([xt, memory_output], dim=-1)))
            # Select which memory slots to update
            _, top_indices = attention_weights.squeeze(1).topk(k=min(8, self.memory_size), dim=-1)
            for b in range(batch_size):
                for idx in top_indices[b]:
                    current_memory[b, idx] = (1 - update_signal[b]) * current_memory[b, idx] + update_signal[b] * xt[b]
            outputs.append(memory_output.unsqueeze(1))
        return torch.cat(outputs, dim=1)

class TextEncoder(nn.Module):
    """Simple text encoder for trading prompts"""
    def __init__(self, vocab_size: int = 10000, embed_size: int = 128, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        # text_tokens: [batch, seq_len]
        embedded = self.embedding(text_tokens)
        lstm_out, (hidden, _) = self.lstm(embedded)
        return self.output_proj(hidden[-1])  # Use last hidden state

class AdvancedTradingModel(TradingModelBase):
    """Advanced Trading Model combining multiple architectures"""
    def __init__(self, 
                 price_input_size: int,
                 text_vocab_size: int = 10000,
                 hidden_size: int = 256,
                 output_size: int = 1,
                 use_ltc: bool = True,
                 use_piecewise: bool = True,
                 use_selective_ssm: bool = True,
                 use_memory: bool = True):
        super().__init__(price_input_size, hidden_size, output_size)
        self.use_ltc = use_ltc
        self.use_piecewise = use_piecewise
        self.use_selective_ssm = use_selective_ssm
        self.use_memory = use_memory
        # Text encoder
        self.text_encoder = TextEncoder(text_vocab_size, 128, hidden_size)
        # Input projection
        self.input_proj = nn.Linear(price_input_size, hidden_size)
        # Advanced layers
        self.layers = nn.ModuleList()
        if use_ltc:
            self.layers.append(LiquidTimeConstantLayer(hidden_size, hidden_size))
        if use_piecewise:
            self.layers.append(PiecewiseLinearLayer(hidden_size, hidden_size))
        if use_selective_ssm:
            self.layers.append(SelectiveStateSpaceLayer(hidden_size, hidden_size // 2))
        if use_memory:
            self.layers.append(MemoryAugmentedLayer(hidden_size, 64))
        # Text-price fusion
        self.fusion_layer = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, price_data: torch.Tensor, text_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Price processing
        x = self.input_proj(price_data)
        # Apply advanced layers
        for layer in self.layers:
            if isinstance(layer, (LiquidTimeConstantLayer, MemoryAugmentedLayer, SelectiveStateSpaceLayer)):
                x = layer(x)
            elif isinstance(layer, PiecewiseLinearLayer):
                x = layer(x)
        # Text-price fusion if text is provided
        if text_tokens is not None:
            text_features = self.text_encoder(text_tokens)  # [batch, hidden_size]
            text_features = text_features.unsqueeze(1).repeat(1, x.size(1), 1)  # [batch, seq_len, hidden_size]
            # Cross-attention between text and price features
            fused_features, _ = self.fusion_layer(x, text_features, text_features)
            x = x + fused_features
        # Final prediction
        return self.output_layers(x[:, -1, :])  # Use last timestep

    def get_config(self) -> Dict[str, Any]:
        return {
            'price_input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'use_ltc': self.use_ltc,
            'use_piecewise': self.use_piecewise,
            'use_selective_ssm': self.use_selective_ssm,
            'use_memory': self.use_memory
        }
