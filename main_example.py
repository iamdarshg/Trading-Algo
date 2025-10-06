#!/usr/bin/env python3
"""
Advanced Algorithmic Trading Bot with Neural Architectures
This script demonstrates how to use the advanced trading bot library with
state-of-the-art neural architectures for time series with non-differentiable
characteristics.

Updated by comet-assistant-2 to include PortfolioManager examples.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#!/usr/bin/env python3
"""
Advanced Algorithmic Trading Bot with Neural Architectures

This script demonstrates how to use the advanced trading bot library with
state-of-the-art neural architectures for time series with non-differentiable
characteristics.

Features:
- Liquid Time-Constant Networks (LTCs)
- Piecewise Linear Networks  
- Selective State Space Models (Mamba-style)
- Memory-Augmented Networks
- Real-time data access with dummy trading functions
- Text prompt integration
- Comprehensive training framework
- Risk management
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 Advanced Algorithmic Trading Bot")
    print("="*60)
    print("Featuring state-of-the-art neural architectures for")
    print("non-differentiable time series prediction")
    print("="*60)

    # Import modules
    try:
        from trading_bot.model_builder import (
            ModelBuilder, create_ltc_model, create_hybrid_model, create_memory_focused_model
        )
        print("✅ model_builder imported")
        from trading_bot.data_processor import create_sample_dataset
        print("✅ data_processor imported")
        from trading_bot.trainer import (
            train_model_pipeline, get_training_config_conservative, 
            get_training_config_aggressive, get_training_config_experimental
        )
        print("✅ trainer imported")
        from trading_bot.live_broker import LiveDataBroker
        print("✅ live_broker imported")
        from trading_bot.ml_strategy import AdvancedMLStrategy, TradingBot, create_trading_bot
        print("✅ ml_strategy imported")
        from trading_bot.test_suite import run_integration_test
        print("✅ All modules imported successfully\n")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return

    # Configuration
    SYMBOL = "AAPL"  # You can change this to any stock symbol
    INITIAL_BALANCE = 100000.0
    TRAINING_EPOCHS = 50  # Increase for better performance

    print(f"📊 Configuration:")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"   Training Epochs: {TRAINING_EPOCHS}")
    print()

    # Step 1: Create and prepare data
    print("📈 Step 1: Preparing market data...")
    try:
        sample_data = create_sample_dataset(SYMBOL, period="2y", interval="1h")
        if not sample_data:
            print("❌ Failed to fetch market data. Check internet connection.")
            return

        print(f"✅ Data prepared for {SYMBOL}")
        print(f"   Features: {len(sample_data['processed_data']['feature_names'])}")
        print(f"   Training samples: {len(sample_data['processed_data']['X_train'])}")
        print(f"   Validation samples: {len(sample_data['processed_data']['X_val'])}")
        print()

    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return

    # Step 2: Model selection and creation
    print("🧠 Step 2: Creating neural architecture...")

    # Robustly infer input_size from processed data to avoid accidental 1-feature problems
    processed = sample_data.get('processed_data', {}) if isinstance(sample_data, dict) else {}
    input_size = None

    # 1) Prefer explicit feature names if provided
    feature_names = processed.get('feature_names') if isinstance(processed, dict) else None
    if isinstance(feature_names, (list, tuple)) and len(feature_names) > 0:
        input_size = len(feature_names)

    # 2) Fallback to shape of X_train if available (handles sequence inputs)
    if (input_size is None or input_size <= 1) and 'X_train' in processed:
        X_train = processed.get('X_train')
        try:
            # support numpy arrays and torch tensors
            shape = getattr(X_train, 'shape', None)
            if shape is not None:
                # for sequence data, features are last dim
                if len(shape) >= 3:
                    input_size = int(shape[-1])
                elif len(shape) == 2:
                    input_size = int(shape[1])
                else:
                    input_size = int(shape[0])
        except Exception:
            input_size = input_size or None

    # 3) Fallback to raw_data columns (DataFrame)
    if (input_size is None or input_size <= 1) and 'raw_data' in sample_data:
        rd = sample_data.get('raw_data')
        try:
            if hasattr(rd, 'shape') and len(rd.shape) > 1:
                input_size = int(rd.shape[1])
            elif hasattr(rd, 'columns'):
                input_size = len(rd.columns)
        except Exception:
            input_size = input_size or None

    # Final fallback: choose a conservative default greater than 1 so model building won't fail
    if input_size is None or input_size <= 1:
        DEFAULT_INPUT_SIZE = 20
        print(f"⚠️  Warning: could not reliably infer input_size (got {input_size}). Using default {DEFAULT_INPUT_SIZE}.\n" \
              "If you expect a different number of features, set `input_size` manually in the script.")
        input_size = DEFAULT_INPUT_SIZE

    print(f"Determined input_size = {input_size}")

    # Allow manual override from CLI or environment
    if 'INPUT_SIZE' in globals() and isinstance(globals()['INPUT_SIZE'], int) and globals()['INPUT_SIZE'] > 1:
        print(f"Using user-provided INPUT_SIZE override: {globals()['INPUT_SIZE']}")
        input_size = globals()['INPUT_SIZE']

    print(f"Determined input_size = {input_size}")

    # Let user choose model type
    print("Available model architectures:")
    print("1. LTC Model (Liquid Time-Constants) - Best for non-differentiable signals")
    print("2. Hybrid Model (LTC + SSM + Memory) - Comprehensive approach")
    print("3. Memory-Focused Model - Best for long-term patterns")
    print("4. Custom Model - Build your own")

    choice = input("\nSelect model type (1-4) [1]: ").strip()
    if not choice:
        choice = "1"

    if choice == "1":
        model_builder = create_ltc_model(input_size, hidden_size=256)
        model_name = "LTC_Model"
    elif choice == "2":
        model_builder = create_hybrid_model(input_size, hidden_size=256)
        model_name = "Hybrid_Model"
    elif choice == "3":
        model_builder = create_memory_focused_model(input_size, hidden_size=256)
        model_name = "Memory_Model"
    elif choice == "4":
        # Custom model building example
        model_builder = (ModelBuilder()
                        .set_input_size(input_size)
                        .set_name("Custom_Model")
                        .add_ltc(hidden_size=128, dt=0.1)
                        .add_selective_ssm(state_size=64)
                        .add_memory_augmented(memory_size=32)
                        .add_piecewise_linear(output_size=64, num_pieces=8)
                        .add_linear(output_size=32, activation='relu', dropout=0.1)
                        .add_linear(output_size=1, activation='tanh'))
        model_name = "Custom_Model"
    else:
        print("Invalid choice, using LTC model")
        model_builder = create_ltc_model(input_size, hidden_size=256)
        model_name = "LTC_Model"

    print(f"✅ {model_name} architecture created")
    print()

    # Step 3: Training configuration
    print("🏋️ Step 3: Training configuration...")

    print("Training configurations:")
    print("1. Conservative (50 epochs, safe parameters)")
    print("2. Aggressive (200 epochs, fast learning)")
    print("3. Experimental (150 epochs, balanced)")

    train_choice = input("\nSelect training config (1-3) [1]: ")

    if not train_choice:
        train_choice = "1"

    if train_choice == "1":
        training_config = get_training_config_conservative()
        config_name = "Conservative"
    elif train_choice == "2":
        training_config = get_training_config_aggressive()
        config_name = "Aggressive"
    elif train_choice == "3":
        training_config = get_training_config_experimental()
        config_name = "Experimental"
    else:
        print("Invalid choice, using Conservative config")
        training_config = get_training_config_conservative()
        config_name = "Conservative"

    print(f"✅ {config_name} training configuration selected")
    print()

    # Step 4: Train model
    print("🚀 Step 4: Training the model...")
    # try:
    # Unpack sample data
    X_train, y_train = sample_data['processed_data']['X_train'], sample_data['processed_data']['y_train']
    X_val, y_val = sample_data['processed_data']['X_val'], sample_data['processed_data']['y_val']

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Train the model using the selected configuration
    train_model_pipeline(
        model_builder=model_builder,
        processed_data={
            'X_train': X_train_tensor,
            'y_train': y_train_tensor,
            'X_val': X_val_tensor,
            'y_val': y_val_tensor
        },
        training_config=training_config
    )

    print("✅ Model training complete")
    print()

    # except Exception as e:
    #     print(f"❌ Model training failed: {e}")
    #     return

    # Step 5: Live trading setup
    print("📈 Step 5: Setting up live trading...")
    try:
        # Initialize live data broker
        live_broker = LiveDataBroker(symbol=SYMBOL)

        # Example: Fetch and display latest market data
        latest_data = live_broker.get_latest_data()
        print(f"✅ Live data feed established for {SYMBOL}")
        print(latest_data)

        # Build or reuse model: if `model` exists (from training) use it; otherwise build an untrained model
        try:
            final_model = model
        except NameError:
            print("No trained model found in this session, building untrained model for demo...")
            final_model = model_builder.build()

        # Ensure the model is on CPU for this demo and in eval mode
        try:
            final_model.cpu()
            final_model.eval()
        except Exception:
            pass

        # Create the ML strategy
        try:
            strategy = AdvancedMLStrategy(
                model=final_model,
                data_processor=sample_data['data_processor'],
                text_processor=sample_data['text_processor'],
                confidence_threshold=0.6,
                max_position_size=0.1
            )
            print("✅ Strategy created")
        except Exception as e:
            print(f"⚠️ Could not create AdvancedMLStrategy: {e}\nFalling back to a trivial strategy that holds.")
            class HoldStrategy(BaseStrategy):
                def __init__(self):
                    super().__init__('HoldStrategy')
                def generate_signal(self, data):
                    return TradingSignal(symbol=data.get('symbol',''), signal_type='hold', confidence=0.0)
            strategy = HoldStrategy()

        # Create or reuse broker
        bot_broker = live_broker if 'live_broker' in locals() else LiveDataBroker(initial_balance=INITIAL_BALANCE)

        # Create TradingBot via helper
        try:
            bot = create_trading_bot(strategy=strategy, broker=bot_broker, symbols=[SYMBOL], update_interval=60, max_positions=3)
            print("✅ TradingBot instantiated")
        except Exception as e:
            print(f"❌ Failed to create TradingBot: {e}")
            return

        # Run a few simulated iterations
        print("Running 3 demo trading iterations...")
        for i in range(3):
            print(f"-- Iteration {i+1} --")
            try:
                bot.run_single_iteration()
            except Exception as ex:
                print(f"Iteration {i+1} failed: {ex}")

        # Print portfolio summary
        try:
            summary = bot.get_portfolio_summary()
            print("\nPortfolio Summary:")
            print(f"  Total Value: ${summary.get('total_value', 0):.2f}")
            print(f"  Cash Balance: ${summary.get('cash_balance', 0):.2f}")
            print(f"  P&L: ${summary.get('total_pnl', 0):.2f}")
            print(f"  Positions: {summary.get('num_positions', 0)}")
        except Exception as e:
            print(f"Could not retrieve portfolio summary: {e}")

    except Exception as e:
        print(f"❌ Live trading setup failed: {e}")
        return

    print("All steps completed successfully. The bot is ready for live trading!")
    print("Monitor the performance and adjust configurations as necessary.")



def main2():
    main()
    print("🚀 Advanced Algorithmic Trading Bot")
    print("="*60)
    print("Featuring state-of-the-art neural architectures for")
    print("non-differentiable time series prediction")
    print("="*60)
    try:
        from trading_bot.model_builder import (
            ModelBuilder, create_ltc_model, create_hybrid_model, create_memory_focused_model
        )
        from trading_bot.data_processor import create_sample_dataset
        from trading_bot.trainer import (
            train_model_pipeline, get_training_config_conservative,
            get_training_config_aggressive, get_training_config_experimental
        )
        from trading_bot.live_broker import LiveDataBroker
        from trading_bot.ml_strategy import AdvancedMLStrategy, TradingBot, create_trading_bot
        # NEW: PortfolioManager import
        from trading_bot.portfolio_manager import PortfolioManager
        print("✅ Modules imported (including PortfolioManager)")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return

    # Minimal example: create and run PortfolioManager for multiple tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    pm = PortfolioManager(tickers=tickers, vram_gb=8)
    # Toggle chart pattern NN as an input modifier across all bots
    pm.spawn_bots(use_chart_patterns=True, initial_balance=100000.0)
    # Run one iteration over all bots (fetch, signal, execute, track equity)
    pm.run_iteration()
    # Compute portfolio-level risk metrics
    metrics = pm.compute_portfolio_metrics(timeframe='daily')
    print("\n📈 Portfolio Metrics (daily):")
    for k, v in metrics['portfolio'].items():
        print(f"  {k}: {v}")
    print("High-correlation clusters:", metrics.get('high_correlation_clusters', []))

    # Optional: turn off chart patterns and run another iteration
    pm.pattern_nn.enabled = False
    pm.run_iteration()
    metrics2 = pm.compute_portfolio_metrics(timeframe='weekly')
    print("\n📈 Portfolio Metrics (weekly):")
    for k, v in metrics2['portfolio'].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main2()
