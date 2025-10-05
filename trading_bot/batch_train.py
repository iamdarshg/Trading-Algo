#!/usr/bin/env python3
"""
Batch Processor for Multi-Symbol Trading Model Training

This script trains advanced neural architectures on multiple stock symbols
simultaneously, allowing for better generalization and cross-market learning.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Absolute imports
from trading_bot.model_builder import ModelBuilder, create_hybrid_model, create_ltc_model, create_memory_focused_model
from trading_bot.data_processor import create_sample_dataset, TradingDataProcessor
from trading_bot.trainer import (
    train_model_pipeline, 
    get_training_config_conservative,
    get_training_config_aggressive, 
    get_training_config_experimental
)

class BatchTrainingManager:
    """Manages batch training across multiple symbols"""

    def __init__(self,
                 symbols: List[str],
                 model_type: str = 'hybrid',
                 training_config: str = 'conservative',
                 data_period: str = '2y',
                 output_dir: str = 'models',
                 use_parallel: bool = True,
                 max_workers: int = 4):

        self.symbols = symbols
        self.model_type = model_type
        self.training_config = training_config
        self.data_period = data_period
        self.output_dir = output_dir
        self.use_parallel = use_parallel
        self.max_workers = max_workers

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Training results
        self.results = {}
        self.failed_symbols = []

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/batch_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        if self.training_config == 'conservative':
            return get_training_config_conservative()
        elif self.training_config == 'aggressive':
            return get_training_config_aggressive()
        elif self.training_config == 'experimental':
            return get_training_config_experimental()
        else:
            raise ValueError(f"Unknown training config: {self.training_config}")

    def create_model_builder(self, input_size: int) -> ModelBuilder:
        """Create model builder based on type"""
        if self.model_type == 'ltc':
            return create_ltc_model(input_size)
        elif self.model_type == 'hybrid':
            return create_hybrid_model(input_size)
        elif self.model_type == 'memory':
            return create_memory_focused_model(input_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """Train model on a single symbol"""
        self.logger.info(f"Starting training for {symbol}")

        try:
            # Create dataset
            self.logger.info(f"Fetching data for {symbol}")
            dataset = create_sample_dataset(symbol, self.data_period)

            if not dataset:
                raise ValueError(f"Failed to create dataset for {symbol}")

            # Get input size
            input_size = len(dataset['processed_data']['feature_names'])
            self.logger.info(f"Input size for {symbol}: {input_size}")

            # Create model
            model_builder = self.create_model_builder(input_size)

            # Get training config
            config = self.get_training_config()

            # Train model
            self.logger.info(f"Training model for {symbol} with config: {self.training_config}")
            model, history = train_model_pipeline(
                model_builder,
                dataset['processed_data'],
                dataset['encoded_news'],
                config
            )

            # Save model and results
            model_filename = f"{symbol.lower()}_{self.model_type}_model.pth"
            config_filename = f"{symbol.lower()}_{self.model_type}_config.json"
            data_filename = f"{symbol.lower()}_data_processor.pkl"

            model_path = os.path.join(self.output_dir, model_filename)
            config_path = os.path.join(self.output_dir, config_filename)
            data_path = os.path.join(self.output_dir, data_filename)

            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'training_history': history,
                'symbol': symbol,
                'model_type': self.model_type,
                'training_config': self.training_config,
                'input_size': input_size,
                'feature_names': dataset['processed_data']['feature_names']
            }, model_path)

            # Save model config
            model_builder.set_name(f"{symbol}_{self.model_type}_Model")
            model_builder.save_config(config_path)

            # Save data processor
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'data_processor': dataset['data_processor'],
                    'text_processor': dataset['text_processor']
                }, f)

            # Calculate final metrics
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            best_val_loss = min(history['val_loss'])
            epochs_trained = len(history['train_loss'])

            result = {
                'symbol': symbol,
                'status': 'success',
                'model_path': model_path,
                'config_path': config_path,
                'data_path': data_path,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'best_val_loss': best_val_loss,
                'epochs_trained': epochs_trained,
                'input_size': input_size,
                'feature_count': len(dataset['processed_data']['feature_names'])
            }

            self.logger.info(f"✅ Successfully trained {symbol}")
            self.logger.info(f"   Final validation loss: {final_val_loss:.6f}")
            self.logger.info(f"   Best validation loss: {best_val_loss:.6f}")
            self.logger.info(f"   Epochs trained: {epochs_trained}")

            return result

        except Exception as e:
            error_msg = f"Failed to train {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': error_msg
            }

    def train_all_symbols_sequential(self) -> Dict[str, Any]:
        """Train models sequentially"""
        self.logger.info(f"Starting sequential training for {len(self.symbols)} symbols")

        for symbol in self.symbols:
            result = self.train_single_symbol(symbol)
            self.results[symbol] = result

            if result['status'] == 'failed':
                self.failed_symbols.append(symbol)

        return self.results

    def train_all_symbols_parallel(self) -> Dict[str, Any]:
        """Train models in parallel"""
        self.logger.info(f"Starting parallel training for {len(self.symbols)} symbols with {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all training jobs
            future_to_symbol = {
                executor.submit(self.train_single_symbol, symbol): symbol 
                for symbol in self.symbols
            }

            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]

                try:
                    result = future.result()
                    self.results[symbol] = result

                    if result['status'] == 'failed':
                        self.failed_symbols.append(symbol)

                except Exception as e:
                    error_msg = f"Exception in training {symbol}: {str(e)}"
                    self.logger.error(error_msg)
                    self.results[symbol] = {
                        'symbol': symbol,
                        'status': 'failed',
                        'error': error_msg
                    }
                    self.failed_symbols.append(symbol)

        return self.results

    def run_batch_training(self) -> Dict[str, Any]:
        """Run the batch training process"""
        start_time = datetime.now()

        self.logger.info("="*80)
        self.logger.info("🚀 STARTING BATCH TRAINING")
        self.logger.info("="*80)
        self.logger.info(f"Symbols: {', '.join(self.symbols)}")
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"Training Config: {self.training_config}")
        self.logger.info(f"Data Period: {self.data_period}")
        self.logger.info(f"Output Directory: {self.output_dir}")
        self.logger.info(f"Parallel Processing: {self.use_parallel}")
        if self.use_parallel:
            self.logger.info(f"Max Workers: {self.max_workers}")
        self.logger.info("="*80)

        # Run training
        if self.use_parallel:
            results = self.train_all_symbols_parallel()
        else:
            results = self.train_all_symbols_sequential()

        # Calculate summary statistics
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        successful_symbols = [s for s in self.symbols if s not in self.failed_symbols]
        success_rate = len(successful_symbols) / len(self.symbols) * 100

        # Generate summary report
        summary = {
            'total_symbols': len(self.symbols),
            'successful_symbols': len(successful_symbols),
            'failed_symbols': len(self.failed_symbols),
            'success_rate': success_rate,
            'total_time': total_time,
            'avg_time_per_symbol': total_time / len(self.symbols),
            'model_type': self.model_type,
            'training_config': self.training_config,
            'failed_symbol_list': self.failed_symbols
        }

        # Log summary
        self.logger.info("="*80)
        self.logger.info("📊 BATCH TRAINING SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Total Symbols: {summary['total_symbols']}")
        self.logger.info(f"Successful: {summary['successful_symbols']}")
        self.logger.info(f"Failed: {summary['failed_symbols']}")
        self.logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"Total Time: {total_time/60:.1f} minutes")
        self.logger.info(f"Avg Time per Symbol: {summary['avg_time_per_symbol']/60:.1f} minutes")

        if self.failed_symbols:
            self.logger.warning(f"Failed Symbols: {', '.join(self.failed_symbols)}")

        # Save summary report
        summary_path = os.path.join(self.output_dir, f"batch_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'summary': summary,
                'detailed_results': results
            }, f, indent=2, default=str)

        self.logger.info(f"Summary report saved: {summary_path}")
        self.logger.info("="*80)
        self.logger.info("✅ BATCH TRAINING COMPLETED")
        self.logger.info("="*80)

        return {
            'summary': summary,
            'results': results,
            'summary_path': summary_path
        }

def get_popular_symbols() -> List[str]:
    """Get a list of popular trading symbols"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'META', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'DIS',
        'ADBE', 'PYPL', 'INTC', 'CMCSA', 'VZ'
    ]

def get_sector_etfs() -> List[str]:
    """Get sector ETF symbols"""
    return [
        'XLK',  # Technology
        'XLV',  # Healthcare
        'XLF',  # Financials
        'XLE',  # Energy
        'XLI',  # Industrials
        'XLB',  # Materials
        'XLY',  # Consumer Discretionary
        'XLP',  # Consumer Staples
        'XLU',  # Utilities
        'XLRE', # Real Estate
        'XLC'   # Communications
    ]

def main():
    """Main batch training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Batch Training for Multiple Symbols')
    parser.add_argument('--symbols', nargs='+', help='List of symbols to train on')
    parser.add_argument('--model-type', choices=['ltc', 'hybrid', 'memory'], 
                       default='hybrid', help='Model architecture type')
    parser.add_argument('--training-config', choices=['conservative', 'aggressive', 'experimental'],
                       default='conservative', help='Training configuration')
    parser.add_argument('--data-period', default='2y', help='Data period (e.g., 1y, 2y, 5y)')
    parser.add_argument('--output-dir', default='models', help='Output directory for models')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--max-workers', type=int, default=4, help='Max parallel workers')
    parser.add_argument('--popular', action='store_true', help='Use popular stock symbols')
    parser.add_argument('--etfs', action='store_true', help='Use sector ETF symbols')

    args = parser.parse_args()

    # Determine symbols to use
    if args.symbols:
        symbols = args.symbols
    elif args.popular:
        symbols = get_popular_symbols()
        print(f"Using popular symbols: {symbols}")
    elif args.etfs:
        symbols = get_sector_etfs()
        print(f"Using sector ETFs: {symbols}")
    else:
        # Default symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        print(f"Using default symbols: {symbols}")

    # Create batch trainer
    trainer = BatchTrainingManager(
        symbols=symbols,
        model_type=args.model_type,
        training_config=args.training_config,
        data_period=args.data_period,
        output_dir=args.output_dir,
        use_parallel=args.parallel,
        max_workers=args.max_workers
    )

    # Run training
    results = trainer.run_batch_training()

    # Print final summary
    summary = results['summary']
    print("\n" + "="*60)
    print("🎉 BATCH TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Successful: {summary['successful_symbols']}/{summary['total_symbols']} models")
    print(f"⏱️  Total Time: {summary['total_time']/60:.1f} minutes")
    print(f"📁 Models saved in: {args.output_dir}/")

    if summary['failed_symbols'] > 0:
        print(f"❌ Failed: {summary['failed_symbols']} models")
        print(f"   Failed symbols: {', '.join(summary['failed_symbol_list'])}")

    print("="*60)

if __name__ == "__main__":
    main()
