# Advanced Algorithmic Trading Bot

A sophisticated algorithmic trading bot featuring state-of-the-art neural architectures designed for time series with non-differentiable characteristics.

## 🚀 Key Features

### ✅ **FIXED: No More Import Issues!**
- **Proper Python package structure** with absolute imports
- **No relative import errors** - everything works out of the box
- **Clean, modular architecture** following Python best practices

### 🧠 **Advanced Neural Architectures**
- **Liquid Time-Constant Networks (LTCs)** - Adaptive time constants for non-smooth dynamics
- **Piecewise Linear Networks** - Natural handling of discontinuities  
- **Selective State Space Models** - Mamba-style architecture for efficient sequence modeling
- **Memory-Augmented Networks** - Long-term dependency modeling
- **Text Integration** - Incorporate news sentiment and market analysis

### 📊 **NEW: Batch Training System**
- **Train on multiple symbols simultaneously**
- **Parallel processing support** for faster training
- **Comprehensive reporting** and model management
- **Popular stocks and sector ETFs** built-in

## 📦 Installation & Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick system test
python main_example.py --test

# 3. Run main demo
python main_example.py

# 4. Batch train on multiple symbols
python scripts/batch_train.py --popular --parallel
```

## 🏁 Quick Demo

```python
# Single symbol training and trading
from trading_bot import *

# 1. Get data and train model
data = create_sample_dataset("AAPL", "1y")
model_builder = create_hybrid_model(input_size=len(data['processed_data']['feature_names']))
model, history = train_model_pipeline(model_builder, data['processed_data'])

# 2. Create and run trading bot  
broker = LiveDataBroker(initial_balance=100000)
strategy = AdvancedMLStrategy(model, data['data_processor'], data['text_processor'])
bot = TradingBot(strategy, broker, ["AAPL"])
bot.run_single_iteration()  # Test with real data, dummy trades
```

## 🔧 Project Structure

```
advanced_trading_bot/
├── trading_bot/                 # Main package
│   ├── __init__.py             # Package imports
│   ├── models/                 # Neural architectures
│   │   ├── __init__.py
│   │   ├── advanced_models.py  # LTC, SSM, Memory layers
│   │   └── model_builder.py    # Keras-style model builder
│   ├── data/                   # Data processing
│   │   ├── __init__.py
│   │   └── data_processor.py   # Features, indicators, news
│   ├── brokers/                # Trading interfaces
│   │   ├── __init__.py
│   │   └── live_broker.py      # Live data + dummy trading
│   ├── strategies/             # Trading strategies
│   │   ├── __init__.py
│   │   └── ml_strategy.py      # ML strategy + bot orchestrator
│   └── training/               # Training framework
│       ├── __init__.py
│       └── trainer.py          # Training pipeline + configs
├── scripts/                    # Utility scripts
│   └── batch_train.py         # Batch training for multiple symbols
├── configs/                    # Configuration files
├── logs/                      # Training and trading logs  
├── models/                    # Saved models
├── main_example.py            # Main demonstration script
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🎯 Usage Examples

### Single Symbol Training
```bash
# Basic training
python main_example.py --symbol AAPL --epochs 50

# Quick test
python main_example.py --test
```

### Batch Training (NEW!)
```bash
# Train on popular stocks with parallel processing
python scripts/batch_train.py --popular --parallel --max-workers 4

# Train on sector ETFs
python scripts/batch_train.py --etfs --model-type hybrid

# Train on custom symbols
python scripts/batch_train.py --symbols AAPL MSFT GOOGL TSLA --training-config aggressive
```

## 🧠 Neural Architectures

### 1. **LTC Model** (Best for Non-Differentiable Signals)
- Adaptive time constants that adjust to signal characteristics
- Stable bounded behavior for discontinuous inputs
- Superior for irregular price movements

### 2. **Hybrid Model** (Comprehensive - Recommended)
- Combines LTC + State Space + Memory + Piecewise Linear
- Maximum flexibility and performance
- Best overall choice for most use cases

### 3. **Memory-Focused Model** (Best for Long-Term Patterns)
- Multiple memory-augmented layers
- Superior long-term dependency modeling
- Excellent for trend-following strategies

## ⚠️ Important Notes

### **For Educational Use Only**
This system is designed for educational and research purposes. Always:
- **Test thoroughly** with paper trading
- **Start with small amounts**
- **Monitor performance closely**  
- **Understand the risks** involved

### **Broker Integration**
The current implementation uses **dummy trading functions** for safety. To use with real money:

1. **Replace dummy functions** in `trading_bot/brokers/live_broker.py`
2. **Implement your broker's API** (Interactive Brokers, Alpaca, etc.)
3. **Test extensively** in paper trading mode

## 🔥 What's New in This Version

### ✅ **FIXED: Import Issues**
- **No more "attempted relative import beyond top-level package" errors**
- **Proper Python package structure** with `__init__.py` files
- **Absolute imports throughout** - clean and reliable

### 🆕 **NEW: Batch Training System**
- **Train on multiple symbols** simultaneously  
- **Parallel processing** for faster training
- **Built-in symbol lists** (popular stocks, sector ETFs)
- **Comprehensive reporting** and model management

### 🚀 **Enhanced Architecture**
- **Improved model building** with proper layer connections
- **Better error handling** and logging
- **Streamlined training pipeline**
- **Professional project structure**

---

**Ready to build the future of algorithmic trading? Let's go! 🚀**

⚠️ **DISCLAIMER**: Trading involves substantial risk of loss. This software is for educational purposes only. Always consult with a qualified financial advisor before making trading decisions.
