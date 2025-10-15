from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import json
import torch.nn as nn
# Absolute imports
from trading_bot.advanced_models import (
    LiquidTimeConstantLayer,
    PiecewiseLinearLayer,
    SelectiveStateSpaceLayer,
    MemoryAugmentedLayer,
    TextEncoder
)
class LayerConfig(ABC):
    """Base class for layer configurations"""
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
    @abstractmethod
    def build(self, input_size: int) -> nn.Module:
        """Build the actual layer"""
        pass
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'params': self.params
        }
class LTCConfig(LayerConfig):
    """Liquid Time-Constant Layer Configuration"""
    def __init__(self, hidden_size: int, dt: float = 0.1, **kwargs):
        super().__init__('LTC', hidden_size=hidden_size, dt=dt, **kwargs)
    def build(self, input_size: int) -> nn.Module:
        return LiquidTimeConstantLayer(
            input_size=input_size,
            hidden_size=self.params['hidden_size'],
            dt=self.params.get('dt', 0.1)
        )
class PiecewiseLinearConfig(LayerConfig):
    """Piecewise Linear Layer Configuration"""
    def __init__(self, output_size: int, num_pieces: int = 8, **kwargs):
        super().__init__('PiecewiseLinear', output_size=output_size, num_pieces=num_pieces, **kwargs)
    def build(self, input_size: int) -> nn.Module:
        return PiecewiseLinearLayer(
            input_size=input_size,
            output_size=self.params['output_size'],
            num_pieces=self.params.get('num_pieces', 8)
        )
class SelectiveSSMConfig(LayerConfig):
    """Selective State Space Model Configuration"""
    def __init__(self, state_size: int, dt_rank: int = 16, **kwargs):
        super().__init__('SelectiveSSM', state_size=state_size, dt_rank=dt_rank, **kwargs)
    def build(self, input_size: int) -> nn.Module:
        return SelectiveStateSpaceLayer(
            input_size=input_size,
            state_size=self.params['state_size'],
            dt_rank=self.params.get('dt_rank', 16)
        )
class MemoryAugmentedConfig(LayerConfig):
    """Memory-Augmented Layer Configuration"""
    def __init__(self, memory_size: int, num_heads: int = 8, **kwargs):
        super().__init__('MemoryAugmented', memory_size=memory_size, num_heads=num_heads, **kwargs)
    def build(self, input_size: int) -> nn.Module:
        return MemoryAugmentedLayer(
            input_size=input_size,
            memory_size=self.params['memory_size'],
            num_heads=self.params.get('num_heads', 8)
        )
class LinearConfig(LayerConfig):
    """Linear Layer Configuration"""
    def __init__(self, output_size: int, activation: str = 'relu', dropout: float = 0.0, **kwargs):
        super().__init__('Linear', output_size=output_size, activation=activation, dropout=dropout, **kwargs)
    def build(self, input_size: int) -> nn.Module:
        layers = [nn.Linear(input_size, self.params['output_size'])]
        # Add activation
        activation = self.params.get('activation', 'relu')
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        # Add dropout
        dropout = self.params.get('dropout', 0.0)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)
class ModelBuilder:
    """Keras-style model builder for trading models"""
    def __init__(self):
        self.layers: List[LayerConfig] = []
        self.input_size: Optional[int] = None
        self.text_vocab_size: int = 10000
        self.model_name: str = "CustomTradingModel"
    def set_input_size(self, size: int) -> 'ModelBuilder':
        """Set the input size for the model"""
        self.input_size = size
        return self
    def set_text_vocab_size(self, size: int) -> 'ModelBuilder':
        """Set the text vocabulary size"""
        self.text_vocab_size = size
        return self
    def set_name(self, name: str) -> 'ModelBuilder':
        """Set the model name"""
        self.model_name = name
        return self
    def add_ltc(self, hidden_size: int, dt: float = 0.1) -> 'ModelBuilder':
        """Add a Liquid Time-Constant layer"""
        self.layers.append(LTCConfig(hidden_size=hidden_size, dt=dt))
        return self
    def add_piecewise_linear(self, output_size: int, num_pieces: int = 8) -> 'ModelBuilder':
        """Add a Piecewise Linear layer"""
        self.layers.append(PiecewiseLinearConfig(output_size=output_size, num_pieces=num_pieces))
        return self
    def add_selective_ssm(self, state_size: int, dt_rank: int = 16) -> 'ModelBuilder':
        """Add a Selective State Space Model layer"""
        self.layers.append(SelectiveSSMConfig(state_size=state_size, dt_rank=dt_rank))
        return self
    def add_memory_augmented(self, memory_size: int, num_heads: int = 8) -> 'ModelBuilder':
        """Add a Memory-Augmented layer"""
        self.layers.append(MemoryAugmentedConfig(memory_size=memory_size, num_heads=num_heads))
        return self
    def add_linear(self, output_size: int, activation: str = 'relu', dropout: float = 0.0) -> 'ModelBuilder':
        """Add a Linear layer"""
        self.layers.append(LinearConfig(output_size=output_size, activation=activation, dropout=dropout))
        return self
    def build(self) -> nn.Module:
        """Build the complete model"""
        if self.input_size is None:
            raise ValueError("Input size must be set before building the model")
        import torch.nn.functional as F
        # Helper: infer final sequence size and validate configs
        def _infer_sequence_size(start_size: int, layers_cfg: List[LayerConfig]) -> int:
            cur = int(start_size)
            for lc in layers_cfg:
                params = getattr(lc, 'params', {})
                if 'output_size' in params:
                    cur = int(params['output_size'])
                elif 'hidden_size' in params:
                    cur = int(params['hidden_size'])
                elif 'state_size' in params:
                    # SelectiveSSM typically maps back to input size; keep cur
                    cur = int(cur)
                elif 'memory_size' in params:
                    # Memory-augmented layers usually preserve feature dim
                    cur = int(cur)
                else:
                    cur = int(cur)
                if cur <= 0:
                    raise ValueError(f"Invalid inferred layer size {cur} for layer {getattr(lc, 'name', lc)}")
            return cur
        # Validate layer sequence
        try:
            _ = _infer_sequence_size(self.input_size, self.layers)
        except Exception as e:
            raise ValueError(f"ModelBuilder detected incompatible layer configuration: {e}")
        class CustomTradingModel(nn.Module):
            def __init__(self, layers_config, input_size, text_vocab_size):
                super().__init__()
                # Keep original input size (do not multiply by vocab)
                self.input_size = int(input_size)
                self.text_vocab_size = int(text_vocab_size)
                # Text encoder
                # Use reasonable defaults; TextEncoder should accept (vocab_size, embed_size, hidden_size)
                self.text_encoder = TextEncoder(text_vocab_size, 128, 256)
                # Build layers
                self.layers = nn.ModuleList()
                current_size = int(input_size)
                # Keep a copy of layer configs for analysis
                self._layer_configs = list(layers_config)
                for layer_config in layers_config:
                    layer = layer_config.build(current_size)
                    self.layers.append(layer)
                    # Update current size for next layer
                    if hasattr(layer_config, 'params'):
                        if 'output_size' in layer_config.params:
                            current_size = int(layer_config.params['output_size'])
                        elif 'hidden_size' in layer_config.params:
                            current_size = int(layer_config.params['hidden_size'])

                # Detect if last configured layer is a final linear to 1. If so, pop it from
                # self.layers and treat it as the final output projection applied after fusion.
                self._final_layer = None
                fusion_input_size = current_size
                if len(self._layer_configs) > 0:
                    last_cfg = self._layer_configs[-1]
                    last_params = getattr(last_cfg, 'params', {})
                    if isinstance(last_params, dict) and last_params.get('output_size') == 1:
                        # recompute size prior to final linear
                        cur = int(input_size)
                        for cfg in self._layer_configs[:-1]:
                            p = getattr(cfg, 'params', {})
                            if 'output_size' in p:
                                cur = int(p['output_size'])
                            elif 'hidden_size' in p:
                                cur = int(p['hidden_size'])
                        fusion_input_size = cur
                        # remove final layer from self.layers and save it
                        if len(self.layers) > 0:
                            self._final_layer = self.layers.pop(-1)
                # Fusion layer for text and price data: try attention, otherwise linear projection
                # Determine embedding dim for text encoder
                # Determine the dimensionality of the text encoder's output features.
                # Prefer the output projection's out_features (final text feature dim).
                out_proj = getattr(self.text_encoder, 'output_proj', None)
                emb_dim = None
                if out_proj is not None:
                    emb_dim = getattr(out_proj, 'out_features', None)
                if emb_dim is None:
                    # fallback to embedding dim if output proj not available
                    emb_dim = getattr(getattr(self.text_encoder, 'embedding', None), 'embedding_dim', None)
                if emb_dim is None:
                    emb_dim = 128
                suggested_heads = max(1, int(emb_dim) // 32)
                if fusion_input_size > 0 and (fusion_input_size % suggested_heads == 0):
                    # Use multi-head attention for fusion
                    self.fusion_layer = nn.MultiheadAttention(fusion_input_size, num_heads=suggested_heads, batch_first=True)
                    # create optional projection to map text feature dim -> fusion_input_size
                    if int(emb_dim) != fusion_input_size:
                        self._text_proj = nn.Linear(int(emb_dim), fusion_input_size)
                    else:
                        self._text_proj = None
                    self._fusion_type = 'attention'
                else:
                    # Linear projection fallback (no attention)
                    self.fusion_layer = nn.Linear(int(emb_dim), fusion_input_size)
                    self._text_proj = None
                    self._fusion_type = 'linear'
                # Final output layer
                if self._final_layer is not None:
                    self.output_layer = self._final_layer
                else:
                    self.output_layer = nn.Linear(current_size, 1)
            def forward(self, price_data, text_tokens=None):
                x = price_data
                # Apply layers sequentially
                for layer in self.layers:
                    if hasattr(layer, 'forward'):
                        try:
                            x = layer(x)
                        except Exception as e:
                            raise ValueError(f"Error in layer {layer}: {e}")
                    else:
                        # Handle sequential (stateless) layers that expect 2D input
                        if x.dim() == 3:  # [batch, seq, features]
                            batch_size, seq_len, features = x.shape
                            x = x.view(-1, features)
                            x = layer(x)
                            x = x.view(batch_size, seq_len, -1)
                        else:
                            x = layer(x)
                # Text-price fusion if text is provided
                if text_tokens is not None:
                    text_features = self.text_encoder(text_tokens)
                    if x.dim() == 3:
                        # Expand text features across the time dimension
                        text_features = text_features.unsqueeze(1).repeat(1, x.size(1), 1)
                        # For attention, ensure the price/features fed to attention have the fusion_input_size
                        if self._fusion_type == 'attention':
                            if x.size(-1) != self.fusion_layer.embed_dim:
                                # try to project/trim x to fusion_input_size if possible
                                if x.size(-1) > self.fusion_layer.embed_dim:
                                    x_for_fusion = x[..., :self.fusion_layer.embed_dim]
                                else:
                                    # pad with zeros if x is smaller (rare)
                                    pad = torch.zeros(x.size(0), x.size(1), self.fusion_layer.embed_dim - x.size(-1), device=x.device)
                                    x_for_fusion = torch.cat([x, pad], dim=-1)
                            else:
                                x_for_fusion = x
                            # Project text embeddings to match attention embed dim if needed
                            if getattr(self, '_text_proj', None) is not None:
                                tproj = self._text_proj(text_features)
                            else:
                                tproj = text_features
                            fused_features, _ = self.fusion_layer(x_for_fusion, tproj, tproj)
                        else:
                            proj = self.fusion_layer(text_features)
                            fused_features = proj
                        x = x + fused_features
                # Return last timestep if 3D
                if x.dim() == 3:
                    x = x[:, -1, :]
                # Final output mapping without attention
                out = self.output_layer(x)
                return out
        return CustomTradingModel(self.layers, self.input_size, self.text_vocab_size)
    def save_config(self, filepath: str):
        """Save model configuration to JSON"""
        config = {
            'model_name': self.model_name,
            'input_size': self.input_size,
            'text_vocab_size': self.text_vocab_size,
            'layers': [layer.to_dict() for layer in self.layers]
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    def get_config(self) -> Dict[str, Any]:
        """Return a serializable config dict for this builder."""
        return {
            'model_name': self.model_name,
            'input_size': self.input_size,
            'text_vocab_size': self.text_vocab_size,
            'layers': [layer.to_dict() for layer in self.layers]
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ModelBuilder':
        """Reconstruct a ModelBuilder from an in-memory config dict."""
        builder = cls()
        builder.model_name = config.get('model_name', 'CustomTradingModel')
        builder.input_size = config.get('input_size')
        builder.text_vocab_size = config.get('text_vocab_size', 10000)
        # Recreate layers
        layer_map = {
            'LTC': LTCConfig,
            'PiecewiseLinear': PiecewiseLinearConfig,
            'SelectiveSSM': SelectiveSSMConfig,
            'MemoryAugmented': MemoryAugmentedConfig,
            'Linear': LinearConfig
        }
        for layer_dict in config.get('layers', []):
            name = layer_dict.get('name')
            params = layer_dict.get('params', {})
            layer_class = layer_map.get(name)
            if layer_class is None:
                # unknown layer; skip
                continue
            layer_config = layer_class(**params)
            builder.layers.append(layer_config)
        return builder

    @classmethod
    def load_config(cls, source) -> 'ModelBuilder':
        """Load model configuration.

        "source" can be a filepath (str/Path) or a dict-like object.
        """
        if isinstance(source, dict):
            return cls.from_dict(source)
        with open(source, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)
# Predefined model configurations
# Ensure the last builder-added layer is a simple linear projection to 1 output
# with no activation so the final model output is not expanded (e.g., to 1024)
def create_ltc_model(input_size: int, hidden_size: int = 16) -> ModelBuilder:
    """Create a model focused on Liquid Time-Constants for non-differentiable signals"""
    return (ModelBuilder()
            .set_input_size(input_size)
            .add_ltc(hidden_size=hidden_size, dt=0.1)
            .add_ltc(hidden_size=hidden_size//2, dt=0.05)
            # Last builder layer must be simple linear to 1 with no activation
            .add_linear(output_size=1, activation='relu', dropout=0.0))

def create_hybrid_model(input_size: int, hidden_size: int = 16) -> ModelBuilder:
    """Create a hybrid model with multiple advanced architectures"""
    return (ModelBuilder()
            .set_input_size(input_size)
            .add_ltc(hidden_size=hidden_size)
            .add_selective_ssm(state_size=hidden_size//3)
            .add_memory_augmented(memory_size=hidden_size//2, num_heads=4)
            .add_piecewise_linear(output_size=hidden_size//2, num_pieces=8)
            # Fix the bug: do NOT output 1024; use 1 with no activation
            .add_linear(output_size=1, activation='relu', dropout=0.0))

def create_memory_focused_model(input_size: int, hidden_size: int = 16) -> ModelBuilder:
    """Create a model focused on long-term memory for trend analysis"""
    return (ModelBuilder()
            .set_input_size(input_size)
            .add_memory_augmented(memory_size=128, num_heads=8)
            .add_selective_ssm(state_size=hidden_size)
            .add_memory_augmented(memory_size=64, num_heads=4)
            # Last builder layer simple linear to 1, no activation
            .add_linear(output_size=1, activation='relu', dropout=0.0))
