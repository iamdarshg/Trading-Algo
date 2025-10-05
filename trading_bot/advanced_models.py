import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
EPS = 1e-8
BIG_CLAMP = 50.0  # to prevent exp overflow

def safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(x, max=BIG_CLAMP))

def safe_softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

def has_nan_or_inf(x: torch.Tensor) -> bool:
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()

class TradingModelBase(nn.Module, ABC):
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
    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = max(dt, EPS)
        self.sensory_w = nn.Linear(input_size, hidden_size)
        self.sensory_mu = nn.Linear(input_size, hidden_size)
        self.sensory_sigma = nn.Linear(input_size, hidden_size)
        self.inter_w = nn.Linear(hidden_size, hidden_size)
        self.inter_mu = nn.Linear(hidden_size, hidden_size)
        self.inter_sigma = nn.Linear(hidden_size, hidden_size)
        self.time_constant = nn.Linear(input_size + hidden_size, hidden_size)
        for lin in [self.sensory_w, self.sensory_mu, self.sensory_sigma,
                    self.inter_w, self.inter_mu, self.inter_sigma, self.time_constant]:
            nn.init.kaiming_normal_(lin.weight, nonlinearity='linear')
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if h is None:
            h = torch.ones(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]
            tau_pre = self.time_constant(torch.cat([xt, h], dim=1))
            tau = safe_softplus(tau_pre) + 0.1
            sensory_input = self.sensory_w(xt) * torch.sigmoid(self.sensory_mu(xt)) * safe_exp(self.sensory_sigma(xt))
            inter_input = self.inter_w(h) * torch.sigmoid(self.inter_mu(h)) * safe_exp(self.inter_sigma(h))
            dh_dt = (sensory_input + inter_input - h) / torch.clamp(tau, min=EPS)
            h = h + self.dt * dh_dt
            if has_nan_or_inf(dh_dt):
                raise ValueError(f"NaN/Inf detected in dh_dt-LiquidTimeConstantLayer at t={t}")
            outputs.append(h.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        if has_nan_or_inf(out):
            raise ValueError("NaN/Inf detected in LiquidTimeConstantLayer output")
        return out

class PiecewiseLinearLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_pieces: int = 8):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_pieces = num_pieces
        self.breakpoints = nn.Parameter(torch.linspace(-3, 3, num_pieces))
        self.slopes = nn.Parameter(torch.empty(num_pieces, input_size, output_size))
        self.intercepts = nn.Parameter(torch.empty(num_pieces, output_size))
        nn.init.kaiming_normal_(self.slopes, nonlinearity='linear')
        nn.init.zeros_(self.intercepts)
        hidden_size = min(64, input_size * 2)
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_pieces),
            nn.Softmax(dim=-1)
        )
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.shape
        x_flat = x.view(-1, input_size)
        piece_weights = self.gate(x_flat)
        piece_outputs = []
        for i in range(self.num_pieces):
            linear_out = torch.matmul(x_flat, self.slopes[i]) + self.intercepts[i]
            piece_outputs.append(linear_out)
        piece_outputs = torch.stack(piece_outputs, dim=1)
        output = torch.sum(piece_weights.unsqueeze(-1) * piece_outputs, dim=1)
        if has_nan_or_inf(output):
            raise ValueError("NaN/Inf detected in PiecewiseLinearLayer output")
        return output.view(batch_size, seq_len, self.output_size)

class SelectiveStateSpaceLayer(nn.Module):
    def __init__(self, input_size: int, state_size: int, dt_rank: int = 16):
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.dt_rank = dt_rank
        self.dt_proj = nn.Linear(dt_rank, input_size)
        self.A_log = nn.Parameter(torch.randn(input_size, state_size) * 0.02)
        self.D = nn.Parameter(torch.randn(input_size) * 0.02)
        self.x_proj = nn.Linear(input_size, dt_rank + state_size * 2)
        self.out_proj = nn.Linear(state_size, input_size)
        for lin in [self.dt_proj, self.x_proj, self.out_proj]:
            nn.init.kaiming_normal_(lin.weight, nonlinearity='linear')
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.shape
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.state_size, self.state_size], dim=-1)
        dt = safe_softplus(self.dt_proj(dt)) + EPS
        A = -safe_exp(self.A_log.float())
        h = torch.zeros(batch_size, self.state_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)
            A_discrete = torch.exp(torch.clamp(A.unsqueeze(0) * dt_t, max=BIG_CLAMP))
            B_discrete = B[:, t, :].unsqueeze(1) * dt_t
            h = torch.clamp(A_discrete.sum(dim=1), min=-1e6, max=1e6) * h + (torch.clamp(B_discrete * x[:, t, :].unsqueeze(-1), min=-1e6, max=1e6)).sum(dim=1)
            y = torch.sum(C[:, t, :].unsqueeze(1) * h.unsqueeze(1), dim=-1)
            y = y + self.D * x[:, t, :]
            if has_nan_or_inf(h) or has_nan_or_inf(y):
                raise ValueError(f"NaN/Inf detected in SelectiveStateSpaceLayer at t={t}")
            outputs.append(y.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        if has_nan_or_inf(out):
            raise ValueError("NaN/Inf detected in SelectiveStateSpaceLayer output")
        return out

class MemoryAugmentedLayer(nn.Module):
    def __init__(self, input_size: int, memory_size: int, num_heads: int = 8):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.memory = nn.Parameter(torch.randn(memory_size, input_size) * 0.1)
        self.query_proj = nn.Linear(input_size, input_size)
        self.key_proj = nn.Linear(input_size, input_size)
        self.value_proj = nn.Linear(input_size, input_size)
        self.update_gate = nn.Linear(input_size * 2, input_size)
        for lin in [self.query_proj, self.key_proj, self.value_proj, self.update_gate]:
            nn.init.kaiming_normal_(lin.weight, nonlinearity='linear')
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        nn.init.normal_(self.memory, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.shape
        outputs = []
        current_memory = self.memory.unsqueeze(0).repeat(batch_size, 1, 1)
        for t in range(seq_len):
            xt = x[:, t, :]
            q = self.query_proj(xt).unsqueeze(1)
            k = self.key_proj(current_memory)
            v = self.value_proj(current_memory)
            scale = float(max(input_size, 1)) ** 0.5
            logits = torch.matmul(q, k.transpose(-2, -1)) / max(scale, EPS)
            attention_weights = F.softmax(logits, dim=-1)
            memory_output = torch.matmul(attention_weights, v).squeeze(1)
            update_signal = torch.sigmoid(self.update_gate(torch.cat([xt, memory_output], dim=-1)))
            _, top_indices = attention_weights.squeeze(1).topk(k=min(8, self.memory_size), dim=-1)
            for b in range(batch_size):
                for idx in top_indices[b]:
                    current_memory[b, idx] = (1 - update_signal[b]) * current_memory[b, idx] + update_signal[b] * xt[b]
            if has_nan_or_inf(memory_output):
                raise ValueError(f"NaN/Inf detected in MemoryAugmentedLayer at t={t}")
            outputs.append(memory_output.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        if has_nan_or_inf(out):
            raise ValueError("NaN/Inf detected in MemoryAugmentedLayer output")
        return out

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 10000, embed_size: int = 128, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='sigmoid')
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.kaiming_normal_(self.output_proj.weight, nonlinearity='linear')
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text_tokens)
        lstm_out, (hidden, _) = self.lstm(embedded)
        out = self.output_proj(hidden[-1])
        if has_nan_or_inf(out):
            raise ValueError("NaN/Inf detected in TextEncoder output")
        return out

class AdvancedTradingModel(TradingModelBase):
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
        self.text_encoder = TextEncoder(text_vocab_size, 128, hidden_size)
        self.input_proj = nn.Linear(price_input_size, hidden_size)
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity='linear')
        nn.init.zeros_(self.input_proj.bias)
        self.layers = nn.ModuleList()
        if use_ltc:
            self.layers.append(LiquidTimeConstantLayer(hidden_size, hidden_size))
        if use_piecewise:
            self.layers.append(PiecewiseLinearLayer(hidden_size, hidden_size))
        if use_selective_ssm:
            self.layers.append(SelectiveStateSpaceLayer(hidden_size, hidden_size // 2))
        if use_memory:
            self.layers.append(MemoryAugmentedLayer(hidden_size, 64))
        self.fusion_layer = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        for name, param in self.fusion_layer.named_parameters():
            if param.dim() >= 2:
                nn.init.kaiming_normal_(param, nonlinearity='linear')
            else:
                nn.init.zeros_(param)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )
        for m in self.output_layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, price_data: torch.Tensor, text_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(price_data)
        if has_nan_or_inf(x):
            raise ValueError("NaN/Inf detected after input projection")
        for layer in self.layers:
            x = layer(x)
            if has_nan_or_inf(x):
                raise ValueError(f"NaN/Inf detected after layer {layer.__class__.__name__}")
        if text_tokens is not None:
            text_features = self.text_encoder(text_tokens)
            text_features = text_features.unsqueeze(1).repeat(1, x.size(1), 1)
            fused_features, _ = self.fusion_layer(x, text_features, text_features)
            x = x + fused_features
        out = self.output_layers(x[:, -1, :])
        if has_nan_or_inf(out):
            raise ValueError("NaN/Inf detected in AdvancedTradingModel output")
        return out

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
