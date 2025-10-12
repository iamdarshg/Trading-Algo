# Trading Algorithm CLI Tools

This README documents the `main_example.py` and `many.py` command-line tools for backtesting and analyzing trading strategies.

## Overview

These tools implement a trading algorithm framework that:
- Backtests trading strategies on historical data
- Manages portfolios across multiple symbols
- Calculates performance metrics (ROI, Sharpe ratio, etc.)
- Supports customizable fee structures
- Provides pipeline-based data processing

## main_example.py

### Description
`main_example.py` is the single-symbol backtesting tool that demonstrates the trading algorithm on one cryptocurrency pair.

### CLI Options

```bash
python main_example.py [OPTIONS]
```

**Options:**
- `--symbol`: Trading symbol to backtest (default: configurable in script)
- `--start-date`: Start date for backtesting (format: YYYY-MM-DD)
- `--end-date`: End date for backtesting (format: YYYY-MM-DD)
- `--initial-capital`: Starting portfolio value (default: varies by script)
- `--fee`: Trading fee percentage (default: 0.1%)

### Symbol Configuration

The tool supports various cryptocurrency trading pairs:
- BTC/USDT
- ETH/USDT
- Other pairs as configured

Symbols can be specified via command-line arguments or in the script configuration.

### Fee Options

Fee structure supports:
- **Maker fees**: Fees for limit orders that add liquidity
- **Taker fees**: Fees for market orders that remove liquidity
- **Flat percentage**: Simple percentage-based fee model

Default fee: 0.1% per trade (configurable)

### ROI and Performance Metrics

The tool calculates:
- **ROI (Return on Investment)**: Percentage return over the backtesting period
- **Total Return**: Absolute profit/loss in base currency
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline

### Pipeline Architecture

The backtesting pipeline consists of:

1. **Data Loading**: Fetches historical OHLCV data
2. **Indicator Calculation**: Computes technical indicators (MA, RSI, etc.)
3. **Signal Generation**: Produces buy/sell signals based on strategy
4. **Position Management**: Executes trades and tracks positions
5. **Performance Analysis**: Calculates metrics and generates reports

### Example Usage

```bash
# Basic backtest with default settings
python main_example.py

# Backtest specific symbol with custom parameters
python main_example.py --symbol BTC/USDT --start-date 2023-01-01 --end-date 2023-12-31 --initial-capital 10000 --fee 0.15
```

## many.py

### Description
`many.py` is the multi-symbol backtesting tool that enables portfolio analysis across multiple trading pairs simultaneously.

### CLI Options

```bash
python many.py [OPTIONS]
```

**Options:**
- `--symbols`: Comma-separated list of trading symbols
- `--start-date`: Start date for backtesting (format: YYYY-MM-DD)
- `--end-date`: End date for backtesting (format: YYYY-MM-DD)
- `--initial-capital`: Starting portfolio value
- `--allocation`: Capital allocation strategy (equal/weighted)
- `--fee`: Trading fee percentage
- `--rebalance`: Rebalancing frequency (daily/weekly/monthly)

### Symbol List Management

Multiple symbols can be specified:

```bash
# Method 1: Command-line argument
python many.py --symbols BTC/USDT,ETH/USDT,BNB/USDT

# Method 2: Configuration file
# Define symbols in config.json or similar
python many.py --config config.json
```

### Portfolio Management

The tool implements:

#### Capital Allocation
- **Equal allocation**: Divides capital equally among all symbols
- **Weighted allocation**: Allocates based on predefined weights
- **Dynamic allocation**: Adjusts based on performance metrics

#### Rebalancing
- Periodic rebalancing to maintain target allocations
- Configurable frequencies (daily, weekly, monthly)
- Cost-aware rebalancing (considers transaction fees)

#### Risk Management
- Per-symbol position limits
- Portfolio-level stop losses
- Correlation-based diversification

### Fee Options

Same as `main_example.py`, with additional considerations:
- Aggregated fee calculation across all symbols
- Per-symbol fee customization
- Fee optimization strategies

### ROI Promotion Strategies

The multi-symbol approach promotes ROI through:

1. **Diversification**: Reduces risk by spreading capital across symbols
2. **Correlation Analysis**: Selects symbols with low correlation
3. **Dynamic Weighting**: Increases allocation to high-performing assets
4. **Risk Parity**: Balances risk contribution across symbols

### Pipeline Architecture

Extends the single-symbol pipeline:

1. **Multi-Data Loading**: Parallel data fetching for all symbols
2. **Synchronized Processing**: Ensures consistent timing across symbols
3. **Cross-Symbol Analysis**: Identifies inter-market opportunities
4. **Portfolio Optimization**: Adjusts allocations dynamically
5. **Aggregated Reporting**: Consolidates metrics across all positions

### Performance Metrics

Additional metrics for multi-symbol analysis:
- **Portfolio ROI**: Overall return across all symbols
- **Symbol Contribution**: Individual symbol impact on portfolio
- **Correlation Matrix**: Inter-symbol relationships
- **Portfolio Sharpe Ratio**: Risk-adjusted portfolio performance
- **Diversification Benefit**: Value added by multi-symbol strategy

### Example Usage

```bash
# Basic multi-symbol backtest
python many.py --symbols BTC/USDT,ETH/USDT,BNB/USDT

# Advanced portfolio backtest
python many.py --symbols BTC/USDT,ETH/USDT,BNB/USDT,ADA/USDT \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --initial-capital 50000 \
  --allocation weighted \
  --fee 0.1 \
  --rebalance weekly
```

## Data Requirements

Both tools require:
- Historical OHLCV data (Open, High, Low, Close, Volume)
- Data sources: Exchange APIs, CSV files, or databases
- Recommended timeframes: 1m, 5m, 15m, 1h, 1d

## Output

Both tools generate:
- Performance summary (text)
- Trade log (CSV)
- Equity curve (plot)
- Metrics report (JSON)

## Dependencies

```
pandas
numpy
ccxt (for exchange data)
matplotlib (for visualization)
ta-lib (for technical indicators)
```

Install dependencies:
```bash
pip install pandas numpy ccxt matplotlib ta-lib
```

## Notes

- Backtesting results are based on historical data and do not guarantee future performance
- Ensure data quality and completeness for accurate results
- Consider slippage and market impact in real trading scenarios
- Test strategies thoroughly before live deployment

## License

See repository LICENSE file for details.
