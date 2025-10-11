#!/usr/bin/env python3
"""
Utility script to train, continue training, or assemble many trading bots into a PortfolioManager
Modes:
  - train-new: create a new model for a single symbol and train it
  - load-and-train: load an existing model/config and continue training on new data
  - merge-and-live: load many existing models for symbols and spawn a PortfolioManager to run simulated live iterations
  - pipeline: train/sim/promote bots from a symbol list file into live via PortfolioManager
Naming conventions (defaults):
  models/{SYMBOL}_model.pth
  models/{SYMBOL}_config.json
This script is defensive: training calls are wrapped and will fall back to building an untrained model if training fails.
"""
import os
import sys
import argparse
import json
import time
from typing import List, Optional, Callable, Dict, Any
import math
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trading_bot.data_processor import create_sample_dataset
from trading_bot.model_builder import ModelBuilder, create_hybrid_model
from trading_bot.trainer import train_model_pipeline, get_training_config_conservative, get_training_config_aggressive
from trading_bot.live_broker import LiveDataBroker
from trading_bot.ml_strategy import AdvancedMLStrategy, create_trading_bot
from trading_bot.portfolio_manager import PortfolioManager, BotState

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def _standard_paths(symbol: str, models_dir: str = MODELS_DIR):
    sym = symbol.upper()
    model_path = os.path.join(models_dir, f"{sym}_model.pth")
    config_path = os.path.join(models_dir, f"{sym}_config.json")
    return model_path, config_path

def infer_input_size_from_processed(processed: dict) -> Optional[int]:
    try:
        fn = processed.get('feature_names')
        if isinstance(fn, (list, tuple)) and len(fn) > 0:
            return len(fn)
        X_train = processed.get('X_train')
        if X_train is not None:
            shape = getattr(X_train, 'shape', None)
            if shape is not None:
                if len(shape) >= 3:
                    return int(shape[-1])
                elif len(shape) == 2:
                    return int(shape[1])
                else:
                    return int(shape[0])
        raw = processed.get('raw_data')
        if raw is not None and hasattr(raw, 'shape'):
            if len(raw.shape) > 1:
                return int(raw.shape[1])
            if hasattr(raw, 'columns'):
                return len(raw.columns)
    except Exception:
        return None
    return None

def save_model_and_config(symbol: str, model: torch.nn.Module, model_builder: Optional[ModelBuilder], model_path: str, config_path: str):
    payload = {
        'model_state_dict': model.state_dict(),
        'saved_at': time.time()
    }
    try:
        if model_builder is not None and hasattr(model_builder, 'get_config'):
            payload['model_config'] = model_builder.get_config()
    except Exception:
        pass
    torch.save(payload, model_path)
    if model_builder is not None and hasattr(model_builder, 'save_config'):
        try:
            model_builder.save_config(config_path)
        except Exception:
            with open(config_path, 'w') as f:
                json.dump({'note': 'config not serializable via save_config'}, f, indent=2)

def load_builder_from_config(config_path: str) -> Optional[ModelBuilder]:
    try:
        if os.path.exists(config_path):
            return ModelBuilder.load_config(config_path)
    except Exception:
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            return ModelBuilder.load_config(cfg)
        except Exception:
            return None
    return None

def _build_or_reconstruct_builder(symbol: str, processed: dict, config_path: str, model_path: str, default_hidden: int = 256) -> ModelBuilder:
    builder = None
    if os.path.exists(config_path):
        try:
            builder = ModelBuilder.load_config(config_path)
        except Exception:
            builder = None
    payload = None
    if os.path.exists(model_path):
        try:
            payload = torch.load(model_path, map_location='cpu')
        except Exception:
            payload = None
    if builder is None and isinstance(payload, dict) and 'model_config' in payload:
        try:
            builder = ModelBuilder.load_config(payload['model_config'])
        except Exception:
            builder = None
    if builder is None:
        input_size = infer_input_size_from_processed(processed) or 20
        builder = create_hybrid_model(input_size, hidden_size=default_hidden)
    return builder

def train_new(symbol: str, epochs: int = 50, models_dir: str = MODELS_DIR, allow_partial_load: bool = False):
    print(f"Training new model for {symbol}...")
    model_path, config_path = _standard_paths(symbol, models_dir)
    sample = create_sample_dataset(symbol)
    processed = sample['processed_data']
    input_size = infer_input_size_from_processed(processed) or 20
    builder = create_hybrid_model(input_size, hidden_size=2048)
    init_model = None
    init_state = None
    if os.path.exists(model_path):
        try:
            payload = torch.load(model_path, map_location='cpu')
            if isinstance(payload, dict) and 'model_config' in payload:
                try:
                    builder = ModelBuilder.load_config(payload['model_config'])
                    print(f"Reconstructed builder from payload model_config for {symbol}")
                except Exception:
                    pass
            if isinstance(payload, dict) and 'model_state_dict' in payload:
                init_state = payload['model_state_dict']
        except Exception as e:
            print(f"Warning: could not read existing model payload: {e}")
    print("Training configuration:")
    print("Enter 1 for aggressive training settings")
    print("Enter 2 for conservative training settings")
    choice = input("Choose training configuration (1 or 2): ").strip()
    if choice == '2':
        training_config = get_training_config_conservative()
    else:
        training_config = get_training_config_aggressive()
    training_config['epochs'] = epochs
    try:
        if init_model is not None:
            model, history = train_model_pipeline(builder, processed, sample.get('encoded_news'), training_config, init_model=init_model)
        else:
            model, history = train_model_pipeline(builder, processed, sample.get('encoded_news'), training_config, init_state_dict=init_state)
        print(f"Training completed for {symbol}")
    except Exception as e:
        print(f"Training failed for {symbol}: {e}")
        raise
    save_model_and_config(symbol, model, builder, model_path, config_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved config -> {config_path}")
    return model_path, config_path

def load_and_train(symbol: str, epochs: int = 20, models_dir: str = MODELS_DIR, allow_partial_load: bool = False):
    print(f"Loading existing model/config for {symbol} and continuing training...")
    model_path, config_path = _standard_paths(symbol, models_dir)
    sample = create_sample_dataset(symbol)
    processed = sample['processed_data']
    builder = None
    if os.path.exists(config_path):
        try:
            builder = ModelBuilder.load_config(config_path)
            print(f"Loaded config from {config_path}")
        except Exception as e:
            print(f"Failed to load config file: {e}")
            builder = None
    payload = None
    if os.path.exists(model_path):
        try:
            payload = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(f"Warning: could not read saved model payload: {e}")
            payload = None
    if builder is None and isinstance(payload, dict) and 'model_config' in payload:
        try:
            builder = ModelBuilder.load_config(payload['model_config'])
            print("Reconstructed builder from payload model_config")
        except Exception:
            builder = None
    if builder is None:
        input_size = infer_input_size_from_processed(processed) or 20
        builder = create_hybrid_model(input_size, hidden_size=256)
        print("Using default hybrid builder because config could not be loaded")
    training_config = get_training_config_conservative()
    training_config['epochs'] = epochs
    try:
        init_state = None
        if isinstance(payload, dict) and 'model_state_dict' in payload:
            init_state = payload['model_state_dict']
        model, history = train_model_pipeline(builder, processed, sample.get('encoded_news'), training_config, init_state_dict=init_state, allow_partial_load=allow_partial_load)
        print(f"Continued training completed for {symbol}")
    except Exception as e:
        print(f"Continued training failed for {symbol}: {e}")
        raise
    save_model_and_config(symbol, model, builder, model_path, config_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved config -> {config_path}")
    return model_path, config_path

# Fee parsing

def parse_fee_function(fee_spec: Optional[str], fee_params: Optional[str]) -> Optional[Callable[..., float]]:
    if not fee_spec:
        return None
    params: Dict[str, Any] = {}
    if fee_params:
        for kv in fee_params.split(','):
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    params[k.strip()] = float(v)
                except Exception:
                    try:
                        params[k.strip()] = json.loads(v)
                    except Exception:
                        params[k.strip()] = v
    spec = fee_spec.lower()
    if spec == 'flat':
        rate = float(params.get('rate', 0.001))
        def fn(amount: float = 0.0, qty: float = 0.0, price: float = 0.0, side: str = 'buy', **kwargs) -> float:
            return abs(amount) * rate
        return fn
    if spec == 'tiered':
        tiers = params.get('tiers')
        if isinstance(tiers, str):
            try:
                tiers = json.loads(tiers)
            except Exception:
                tiers = None
        if not isinstance(tiers, list):
            tiers = [
                {"notional_lt": 1000, "rate": 0.002},
                {"notional_lt": 10000, "rate": 0.001},
                {"rate": 0.0005}
            ]
        def fn(amount: float = 0.0, qty: float = 0.0, price: float = 0.0, side: str = 'buy', **kwargs) -> float:
            notional = abs(amount)
            for t in tiers:
                if 'notional_lt' in t and notional < float(t['notional_lt']):
                    return notional * float(t['rate'])
            return notional * float(tiers[-1]['rate'])
        return fn
    if spec == 'power':
        a = float(params.get('a', 0.0))
        b = float(params.get('b', 1.0))
        c = float(params.get('c', 0.0))
        def fn(amount: float = 0.0, qty: float = 0.0, price: float = 0.0, side: str = 'buy', **kwargs) -> float:
            notional = abs(amount)
            return a * (notional ** b) + c
        return fn
    if spec == 'lambda':
        expr = params.get('expr')
        if isinstance(expr, str) and expr.strip().startswith('lambda'):
            try:
                fn = eval(expr, {'__builtins__': {'abs': abs, 'max': max, 'min': min, 'math': math}}, {})
                if callable(fn):
                    return fn
            except Exception:
                pass
    return None

# Merge and live with fees option

def merge_and_live(symbols: List[str], iterations: int = 3, models_dir: str = MODELS_DIR, allow_partial_load: bool = False,
                   roi_week_threshold: Optional[float] = None,
                   fee_spec: Optional[str] = None,
                   fee_params: Optional[str] = None):
    print(f"Merging {len(symbols)} symbols into PortfolioManager and running {iterations} live iterations")
    pm = PortfolioManager(tickers=symbols)
    fee_fn = parse_fee_function(fee_spec, fee_params)
    for sym in symbols:
        model_path, config_path = _standard_paths(sym, models_dir)
        sample = create_sample_dataset(sym)
        processed = sample['processed_data']
        builder = _build_or_reconstruct_builder(sym, processed, config_path, model_path, default_hidden=256)
        model = builder.build()
        try:
            if os.path.exists(model_path):
                payload = torch.load(model_path, map_location='cpu')
                if isinstance(payload, dict) and 'model_state_dict' in payload:
                    try:
                        model.load_state_dict(payload['model_state_dict'], strict=False)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: failed to load weights for {sym}: {e}")
        broker = LiveDataBroker(initial_balance=100000.0)
        if fee_fn is not None:
            def _apply_fee(amount: float, qty: float, price: float, side: str) -> float:
                fee = float(fee_fn(amount=amount, qty=qty, price=price, side=side))
                return max(0.0, amount - fee)
            setattr(broker, 'apply_fee', _apply_fee)  # monkey-patch for sim
        strategy = AdvancedMLStrategy(model=model, data_processor=sample['data_processor'], text_processor=sample.get('text_processor'))
        bot = create_trading_bot(strategy=strategy, broker=broker, symbols=[sym], update_interval=300, max_positions=3)
        pm.bots[sym] = BotState(bot=bot, broker=broker, strategy=strategy, symbol=sym)
        print(f"Added bot for {sym}")
    if roi_week_threshold is not None:
        setattr(pm, 'roi_week_threshold', roi_week_threshold)
    for i in range(iterations):
        print(f"\n--- Portfolio iteration {i+1}/{iterations} ---")
        try:
            pm.run_iteration()
        except Exception as e:
            print(f"Portfolio iteration error: {e}")
        metrics = pm.compute_portfolio_metrics()
        print(f"Metrics after iter {i+1}: {metrics}")
    print("Merge-and-live complete")
    return pm

# Pipeline

def load_symbols_from_file(path: str) -> List[str]:
    syms: List[str] = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip().upper()
            if not s or s.startswith('#'):
                continue
            for tok in s.replace('\t', ',').split(','):
                tok = tok.strip().upper()
                if tok:
                    syms.append(tok)
    seen = set()
    uniq: List[str] = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def compute_weekly_roi_pct(broker: LiveDataBroker) -> float:
    try:
        current = broker.get_portfolio_value()
        initial = getattr(broker, 'initial_balance', 0.0) or 0.0
        if initial > 0:
            return (current - initial) / initial * 100.0
    except Exception:
        pass
    return 0.0
# ... [existing code above] ...

def pipeline(symbol_file: str, models_dir: str, epochs: int, iterations: int, allow_partial_load: bool,
             roi_week_threshold: float, fee_spec: Optional[str], fee_params: Optional[str]):
    """Full pipeline: load symbols, train, simulate, promote to live trading"""
    symbols = load_symbols_from_file(symbol_file)
    if not symbols:
        raise ValueError("No symbols loaded from file")
    
    print(f"Pipeline processing {len(symbols)} symbols: {symbols}")
    
    # Train or load+train all bots
    for sym in symbols:
        model_path, _ = _standard_paths(sym, models_dir)
        if os.path.exists(model_path):
            try:
                load_and_train(sym, epochs=max(1, epochs//2), models_dir=models_dir, allow_partial_load=allow_partial_load)
                print(f"Continued training for existing {sym}")
            except Exception as e:
                print(f"Failed to continue training {sym}: {e}")
        else:
            try:
                train_new(sym, epochs=epochs, models_dir=models_dir, allow_partial_load=allow_partial_load)
                print(f"Trained new model for {sym}")
            except Exception as e:
                print(f"Failed to train new model for {sym}: {e}")
    
    # Create portfolio manager and run simulation
    pm = PortfolioManager(tickers=symbols)
    fee_fn = parse_fee_function(fee_spec, fee_params)
    
    # Add all bots to portfolio
    for sym in symbols:
        model_path, config_path = _standard_paths(sym, models_dir)
        sample = create_sample_dataset(sym)
        processed = sample['processed_data']
        
        builder = _build_or_reconstruct_builder(sym, processed, config_path, model_path, default_hidden=256)
        model = builder.build()
        
        # Load model weights
        try:
            if os.path.exists(model_path):
                payload = torch.load(model_path, map_location='cpu')
                if isinstance(payload, dict) and 'model_state_dict' in payload:
                    model.load_state_dict(payload['model_state_dict'], strict=False)
        except Exception as e:
            print(f"Warning: failed to load weights for {sym}: {e}")
        
        # Create broker with fee function
        broker = LiveDataBroker(initial_balance=100000.0)
        if fee_fn is not None:
            def _apply_fee(amount: float, qty: float, price: float, side: str) -> float:
                fee = float(fee_fn(amount=amount, qty=qty, price=price, side=side))
                return max(0.0, amount - fee)
            setattr(broker, 'apply_fee', _apply_fee)
        
        strategy = AdvancedMLStrategy(model=model, data_processor=sample['data_processor'], 
                                    text_processor=sample.get('text_processor'))
        bot = create_trading_bot(strategy=strategy, broker=broker, symbols=[sym], 
                               update_interval=300, max_positions=3)
        
        pm.bots[sym] = BotState(bot=bot, broker=broker, strategy=strategy, symbol=sym, state='simulation')
        print(f"Added {sym} bot to portfolio in simulation mode")
    
    # Set ROI promotion threshold
    if roi_week_threshold is not None:
        setattr(pm, 'roi_week_threshold', roi_week_threshold)
        print(f"ROI promotion threshold set to {roi_week_threshold}% per week")
    
    # Run simulation iterations
    for i in range(iterations):
        print(f"\n--- Pipeline iteration {i+1}/{iterations} ---")
        try:
            pm.run_iteration()
            
            # Check for ROI-based promotions after each iteration
            if roi_week_threshold is not None:
                promote_bots_by_roi(pm, roi_week_threshold)
                
        except Exception as e:
            print(f"Pipeline iteration error: {e}")
        
        # Display metrics
        metrics = pm.compute_portfolio_metrics()
        print(f"Portfolio metrics after iteration {i+1}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    print("\nPipeline complete!")
    return pm

def promote_bots_by_roi(pm: PortfolioManager, threshold: float):
    """Promote simulation bots to live trading if ROI exceeds threshold"""
    promoted = []
    
    for symbol, bot_state in pm.bots.items():
        if bot_state.state == 'simulation':
            roi = compute_weekly_roi_pct(bot_state.broker)
            if roi >= threshold:
                bot_state.state = 'live'
                promoted.append(symbol)
                print(f"🚀 PROMOTED {symbol} to live trading! (ROI: {roi:.2f}%)")
    
    if promoted:
        print(f"Promoted {len(promoted)} bots to live trading: {promoted}")
    else:
        print("No bots met ROI threshold for promotion")

def main():
    parser = argparse.ArgumentParser(description='Training Bot Management CLI')
    parser.add_argument('mode', choices=['train-new', 'load-and-train', 'merge-and-live', 'pipeline'],
                       help='Operation mode')
    
    # Common arguments
    parser.add_argument('--symbol', type=str, help='Single symbol to operate on')
    parser.add_argument('--symbol-file', type=str, help='File containing list of symbols (one per line)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--iterations', type=int, default=3, help='Live trading iterations')
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR, help='Directory for model storage')
    parser.add_argument('--allow-partial-load', action='store_true', 
                       help='Allow partial loading of model weights')
    
    # ROI and fee arguments
    parser.add_argument('--roi-week-threshold', type=float, 
                       help='ROI threshold (%) for promoting bots to live trading')
    parser.add_argument('--fee-spec', type=str, choices=['flat', 'tiered', 'power', 'lambda'],
                       help='Fee structure type')
    parser.add_argument('--fee-params', type=str, 
                       help='Fee parameters as comma-separated key=value pairs')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train-new':
            if not args.symbol:
                raise ValueError("--symbol required for train-new mode")
            train_new(args.symbol, args.epochs, args.models_dir, args.allow_partial_load)
            
        elif args.mode == 'load-and-train':
            if not args.symbol:
                raise ValueError("--symbol required for load-and-train mode")
            load_and_train(args.symbol, args.epochs, args.models_dir, args.allow_partial_load)
            
        elif args.mode == 'merge-and-live':
            if args.symbol_file:
                symbols = load_symbols_from_file(args.symbol_file)
            elif args.symbol:
                symbols = [args.symbol]
            else:
                raise ValueError("Either --symbol or --symbol-file required for merge-and-live mode")
            
            merge_and_live(symbols, args.iterations, args.models_dir, args.allow_partial_load,
                          args.roi_week_threshold, args.fee_spec, args.fee_params)
            
        elif args.mode == 'pipeline':
            if not args.symbol_file:
                raise ValueError("--symbol-file required for pipeline mode")
            if args.roi_week_threshold is None:
                raise ValueError("--roi-week-threshold required for pipeline mode")
            
            pipeline(args.symbol_file, args.models_dir, args.epochs, args.iterations,
                    args.allow_partial_load, args.roi_week_threshold, args.fee_spec, args.fee_params)
                    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
