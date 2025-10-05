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

def main():
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
    main()
