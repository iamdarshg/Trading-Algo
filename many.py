#!/usr/bin/env python3
"""
Utility script to train, continue training, or assemble many trading bots into a PortfolioManager

Modes:
  - train-new: create a new model for a single symbol and train it
  - load-and-train: load an existing model/config and continue training on new data
  - merge-and-live: load many existing models for symbols and spawn a PortfolioManager to run simulated live iterations

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
from typing import List, Optional

import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_bot.data_processor import create_sample_dataset, TradingDataProcessor, TextProcessor
from trading_bot.model_builder import ModelBuilder, create_hybrid_model
from trading_bot.trainer import train_model_pipeline, get_training_config_conservative, get_training_config_aggressive
from trading_bot.live_broker import LiveDataBroker
from trading_bot.ml_strategy import AdvancedMLStrategy, create_trading_bot, TradingBot
from trading_bot.portfolio_manager import PortfolioManager, BotState

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def _standard_paths(symbol: str, models_dir: str = MODELS_DIR):
    sym = symbol.upper()
    model_path = os.path.join(models_dir, f"{sym}_model.pth")
    config_path = os.path.join(models_dir, f"{sym}_config.json")
    return model_path, config_path


def infer_input_size_from_processed(processed: dict) -> Optional[int]:
    # Prefer feature_names
    try:
        fn = processed.get('feature_names')
        if isinstance(fn, (list, tuple)) and len(fn) > 0:
            return len(fn)
        # Try X_train
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
        # Try raw_data columns
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
        # Include model config if available
        if model_builder is not None and hasattr(model_builder, 'get_config'):
            payload['model_config'] = model_builder.get_config()
    except Exception:
        pass

    torch.save(payload, model_path)

    # Save config also as JSON if model_builder provided
    if model_builder is not None and hasattr(model_builder, 'save_config'):
        try:
            model_builder.save_config(config_path)
        except Exception:
            # fallback: write minimal JSON
            with open(config_path, 'w') as f:
                json.dump({'note': 'config not serializable via save_config'}, f, indent=2)


def load_builder_from_config(config_path: str) -> Optional[ModelBuilder]:
    try:
        if os.path.exists(config_path):
            # ModelBuilder.load_config now accepts a filepath or dict
            return ModelBuilder.load_config(config_path)
    except Exception:
        try:
            # try reading raw json and converting
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            return ModelBuilder.load_config(cfg)
        except Exception:
            return None
    return None


def train_new(symbol: str, epochs: int = 50, models_dir: str = MODELS_DIR):
    print(f"Training new model for {symbol}...")
    model_path, config_path = _standard_paths(symbol, models_dir)

    # Prepare data
    sample = create_sample_dataset(symbol)
    processed = sample['processed_data']
    input_size = infer_input_size_from_processed(processed) or 20

    # Build a default hybrid model
    builder = create_hybrid_model(input_size, hidden_size=2048)

    # If a saved model exists, attempt to use its config/weights to initialize
    init_model = None
    init_state = None
    if os.path.exists(model_path):
        try:
            payload = torch.load(model_path, map_location='cpu')
            # try to reconstruct builder from payload model_config first (aggressive)
            if isinstance(payload, dict) and 'model_config' in payload:
                try:
                    builder = ModelBuilder.load_config(payload['model_config'])
                    print(f"Reconstructed builder from payload model_config for {symbol}")
                except Exception:
                    pass
            # capture state dict for partial load
            if isinstance(payload, dict) and 'model_state_dict' in payload:
                init_state = payload['model_state_dict']
        except Exception as e:
            print(f"Warning: could not read existing model payload: {e}")

    # Train
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
        # pass init_model or init_state to train pipeline if available
        if init_model is not None:
            model, history = train_model_pipeline(builder, processed, sample.get('encoded_news'), training_config, init_model=init_model)
        else:
            model, history = train_model_pipeline(builder, processed, sample.get('encoded_news'), training_config, init_state_dict=init_state)
        print(f"Training completed for {symbol}")
    except Exception as e:
        print(f"Training failed for {symbol}: {e}")
        print("Falling back to untrained model (builder.build())")
        model = builder.build()
        history = None

    save_model_and_config(symbol, model, builder, model_path, config_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved config -> {config_path}")
    return model_path, config_path


def load_and_train(symbol: str, epochs: int = 20, models_dir: str = MODELS_DIR):
    print(f"Loading existing model/config for {symbol} and continuing training...")
    model_path, config_path = _standard_paths(symbol, models_dir)

    sample = create_sample_dataset(symbol)
    processed = sample['processed_data']

    # Try file-based config first
    builder = None
    if os.path.exists(config_path):
        try:
            builder = ModelBuilder.load_config(config_path)
            print(f"Loaded config from {config_path}")
        except Exception as e:
            print(f"Failed to load config file: {e}")
            builder = None

    # If there is a saved model payload, check for embedded model_config and state_dict
    payload = None
    if os.path.exists(model_path):
        try:
            payload = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(f"Warning: could not read saved model payload: {e}")
            payload = None
    # If builder not loaded from file, try to reconstruct from payload
    if builder is None and isinstance(payload, dict) and 'model_config' in payload:
        try:
            builder = ModelBuilder.load_config(payload['model_config'])
            print("Reconstructed builder from payload model_config")
        except Exception:
            builder = None

    if builder is None:
        # infer input and build a default
        input_size = infer_input_size_from_processed(processed) or 20
        builder = create_hybrid_model(input_size, hidden_size=256)
        print("Using default hybrid builder because config could not be loaded")

    # Continue training
    training_config = get_training_config_conservative()
    training_config['epochs'] = epochs

    try:
        # Build model and attempt to preload weights (non-strict) if payload contains them
        init_state = None
        if isinstance(payload, dict) and 'model_state_dict' in payload:
            init_state = payload['model_state_dict']

        # If train_model_pipeline supports init_model/init_state_dict, pass them
        model, history = train_model_pipeline(builder, processed, sample.get('encoded_news'), training_config, init_state_dict=init_state)
        print(f"Continued training completed for {symbol}")
    except Exception as e:
        print(f"Continued training failed for {symbol}: {e}")
        print("Falling back to builder.build()")
        model = builder.build()
        history = None

    save_model_and_config(symbol, model, builder, model_path, config_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved config -> {config_path}")
    return model_path, config_path


def merge_and_live(symbols: List[str], iterations: int = 3, models_dir: str = MODELS_DIR):
    print(f"Merging {len(symbols)} symbols into PortfolioManager and running {iterations} live iterations")

    pm = PortfolioManager(tickers=symbols)

    # Build BotState objects and add to pm.bots
    for sym in symbols:
        model_path, config_path = _standard_paths(sym, models_dir)

        builder = None
        model = None
        try:
            if os.path.exists(config_path):
                builder = ModelBuilder.load_config(config_path)
        except Exception:
            builder = None

        sample = create_sample_dataset(sym)
        processed = sample['processed_data']

        if builder is None:
            input_size = infer_input_size_from_processed(processed) or 20
            builder = create_hybrid_model(input_size, hidden_size=256)

        try:
            # If a saved model exists, load its payload and attempt to reconstruct builder and load weights (non-strict)
            payload = None
            if os.path.exists(model_path):
                try:
                    payload = torch.load(model_path, map_location='cpu')
                except Exception as e:
                    print(f"Warning: failed to read payload for {sym}: {e}")
                    payload = None

            # Reconstruct builder from payload if possible
            if payload and isinstance(payload, dict) and 'model_config' in payload:
                try:
                    builder = ModelBuilder.load_config(payload['model_config'])
                    print(f"Reconstructed builder for {sym} from payload config")
                except Exception:
                    pass

            model = builder.build()
            if payload and isinstance(payload, dict) and 'model_state_dict' in payload:
                # Try strict load first, then fallback to non-strict partial load
                try:
                    model.load_state_dict(payload['model_state_dict'])
                except Exception:
                    try:
                        model.load_state_dict(payload['model_state_dict'], strict=False)
                        print(f"Partial weight load for {sym} (non-strict)")
                    except Exception as e_load:
                        print(f"Failed to load weights for {sym}: {e_load}")
        except Exception as e:
            print(f"Warning: failed to build/load model for {sym}: {e}")
            model = builder.build()

        # Create broker and strategy
        broker = LiveDataBroker(initial_balance=100000.0)
        strategy = AdvancedMLStrategy(model=model, data_processor=sample['data_processor'], text_processor=sample['text_processor'])

        bot = create_trading_bot(strategy=strategy, broker=broker, symbols=[sym], update_interval=300, max_positions=3)

        # Store BotState in portfolio manager
        bot_state = BotState(bot=bot, broker=broker, strategy=strategy, symbol=sym)
        pm.bots[sym] = bot_state
        print(f"Added bot for {sym}")

    # Run iterations
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


def parse_args():
    p = argparse.ArgumentParser(description='Train/load/merge trading bots')
    p.add_argument('mode', choices=['train-new', 'load-and-train', 'merge-and-live'], help='Operation mode')
    p.add_argument('--symbol', help='Single symbol for train-new or load-and-train')
    p.add_argument('--symbols', nargs='+', help='List of symbols for merge-and-live')
    p.add_argument('--epochs', type=int, default=50, help='Epochs for training')
    p.add_argument('--models-dir', default=MODELS_DIR, help='Directory to save/load models')
    p.add_argument('--iterations', type=int, default=3, help='Iterations for live merge-and-live')
    return p.parse_args()


def main_cli():
    args = parse_args()
    if args.mode == 'train-new':
        if not args.symbol:
            print('Please specify --symbol for train-new')
            return
        train_new(args.symbol, epochs=args.epochs, models_dir=args.models_dir)
    elif args.mode == 'load-and-train':
        if not args.symbol:
            print('Please specify --symbol for load-and-train')
            return
        load_and_train(args.symbol, epochs=args.epochs, models_dir=args.models_dir)
    elif args.mode == 'merge-and-live':
        syms = args.symbols or []
        if not syms:
            print('Please specify --symbols for merge-and-live')
            return
        merge_and_live(syms, iterations=args.iterations, models_dir=args.models_dir)


if __name__ == '__main__':
    main_cli()
