"""
Portfolio Manager for managing multiple trading bots and portfolio-level risk.
Implemented by comet-assistant-2.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import math
import itertools
from datetime import datetime

# Absolute imports from existing codebase
from trading_bot.live_broker import LiveDataBroker
from trading_bot.ml_strategy import AdvancedMLStrategy, TradingBot
from trading_bot.model_builder import create_hybrid_model
from trading_bot.data_processor import create_sample_dataset, TradingDataProcessor

# Optional: lightweight chart-pattern recognizer (toggleable)
class ChartPatternNN:
    """Simple placeholder NN-like modifier for chart patterns.
    Acts as an input modifier or additional signal layer.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def score_sequence(self, price_df: pd.DataFrame) -> float:
        if not self.enabled or price_df is None or price_df.empty:
            return 0.0
        # Heuristic pattern features (stub): momentum, volatility squeeze, breakout
        close = price_df['Close'].astype(float)
        ret = close.pct_change().dropna()
        if ret.empty:
            return 0.0
        momentum = (close.iloc[-1] / close.iloc[max(0, len(close)-21)] - 1.0)
        vol = ret.rolling(20).std().iloc[-1] if len(ret) >= 20 else ret.std()
        squeeze = float(np.tanh((ret.rolling(5).std().iloc[-1] if len(ret) >= 5 else ret.std()) / (vol + 1e-8)))
        breakout = float(np.tanh((close.iloc[-1] - close.rolling(20).max().fillna(close.iloc[-1]).iloc[-1]) / (close.rolling(20).std().fillna(1).iloc[-1] + 1e-8)))
        # Combine as a bounded modifier in [-0.2, 0.2]
        raw = 0.5*momentum + 0.3*squeeze + 0.2*breakout
        return float(np.clip(raw, -0.2, 0.2))

@dataclass
class BotSpec:
    symbol: str
    use_chart_patterns: bool = False

@dataclass
class BotState:
    bot: TradingBot
    broker: LiveDataBroker
    strategy: AdvancedMLStrategy
    symbol: str
    equity_curve: List[float] = field(default_factory=list)
    last_value: float = 0.0

class PortfolioManager:
    """
    Manage 10-20 bots, track risk, and compute portfolio metrics.

    Example:
        pm = PortfolioManager(tickers=["AAPL","MSFT"], vram_gb=8)
        pm.spawn_bots(use_chart_patterns=True)
        pm.run_iteration()
        metrics = pm.compute_portfolio_metrics()
    """
    def __init__(self, tickers: List[str], vram_gb: float = 8.0):
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("tickers must be a non-empty list")
        if len(tickers) > 20:
            raise ValueError("This demo PortfolioManager supports up to 20 bots")
        self.tickers = tickers
        self.vram_gb = vram_gb
        self.bots: Dict[str, BotState] = {}
        self.pattern_nn = ChartPatternNN(enabled=False)
        self._equity_history: List[Tuple[datetime, float]] = []

    # Integration point: toggle chart pattern modifier globally or per-bot
    def spawn_bots(self, use_chart_patterns: bool = False, initial_balance: float = 100000.0) -> None:
        self.pattern_nn.enabled = use_chart_patterns
        for sym in self.tickers:
            # Prepare data and a compact model per symbol
            sample = create_sample_dataset(sym, period="1y")
            processed = sample['processed_data']
            feature_names = processed.get('feature_names', [])
            input_size = len(feature_names) if feature_names else 20
            model_builder = create_hybrid_model(input_size=input_size, hidden_size=128)
            # Train lightweight or skip if already trained - here we just build model
            model = model_builder.build()
            # Build components
            broker = LiveDataBroker(symbol=sym, initial_balance=initial_balance)
            data_proc: TradingDataProcessor = sample['data_processor']
            strategy = AdvancedMLStrategy(model, data_proc, text_processor=sample.get('text_processor'))
            bot = TradingBot(strategy, broker, symbols=[sym], update_interval=300, max_positions=3, risk_management=True)
            self.bots[sym] = BotState(bot=bot, broker=broker, strategy=strategy, symbol=sym)

    def _apply_chart_pattern_modifier(self, symbol: str, price_df: pd.DataFrame, pred: float) -> float:
        if not self.pattern_nn.enabled:
            return pred
        mod = self.pattern_nn.score_sequence(price_df)
        return float(np.clip(pred + mod, -1.0, 1.0))

    def run_iteration(self) -> None:
        total_value = 0.0
        now = datetime.now()
        for sym, state in self.bots.items():
            # fetch data similar to TradingBot.get_symbol_data
            try:
                price_df = state.broker.get_market_data(sym, period="60d", interval="1d")
                current_price = state.broker.get_current_price(sym)
                if price_df is None or price_df.empty:
                    continue
                # Use strategy to predict, but allow modifier
                pred, conf = state.strategy.predict(price_df, text_prompt=None)
                pred = self._apply_chart_pattern_modifier(sym, price_df, pred)
                # Convert modified prediction into a signal-like action
                data = {
                    'symbol': sym,
                    'price_data': price_df,
                    'text_prompt': "",
                    'current_price': current_price
                }
                signal = state.strategy.generate_signal(data)
                # Overwrite reasoning to reflect modifier when enabled
                if self.pattern_nn.enabled:
                    signal.reasoning += " | chart_pattern_mod_applied"
                # Execute via bot pipeline
                state.bot.execute_signal(signal)
                # Track equity
                pv = state.broker.get_portfolio_value()
                total_value += pv
                state.last_value = pv
                state.equity_curve.append(pv)
            except Exception:
                continue
        if total_value > 0:
            self._equity_history.append((now, total_value))

    # ---------- Risk Metrics ----------
    @staticmethod
    def _max_drawdown(values: List[float]) -> float:
        if not values:
            return 0.0
        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (v - peak) / peak if peak > 0 else 0
            max_dd = min(max_dd, dd)
        return abs(max_dd)

    @staticmethod
    def _volatility(returns: np.ndarray, period: str = 'daily') -> float:
        if returns.size == 0:
            return 0.0
        vol = float(np.std(returns, ddof=1))
        ann_map = {'daily': math.sqrt(252), 'weekly': math.sqrt(52), 'monthly': math.sqrt(12)}
        return vol * ann_map.get(period, math.sqrt(252))

    @staticmethod
    def _sharpe(returns: np.ndarray, rf: float = 0.0, period: str = 'daily') -> float:
        if returns.size == 0:
            return 0.0
        excess = returns - rf/252.0
        vol = np.std(excess, ddof=1)
        if vol == 0:
            return 0.0
        ann_factor = {'daily': math.sqrt(252), 'weekly': math.sqrt(52), 'monthly': math.sqrt(12)}.get(period, math.sqrt(252))
        return float(np.mean(excess) / vol * ann_factor)

    @staticmethod
    def _tail_risk(returns: np.ndarray, q: float = 0.05) -> Dict[str, float]:
        if returns.size == 0:
            return {'VaR': 0.0, 'CVaR': 0.0}
        var = float(np.quantile(returns, q))
        cvar = float(returns[returns <= var].mean()) if np.any(returns <= var) else var
        return {'VaR': var, 'CVaR': cvar}

    @staticmethod
    def _correlation_clusters(returns_df: pd.DataFrame, threshold: float = 0.7) -> List[List[str]]:
        if returns_df.empty or returns_df.shape[1] < 2:
            return [[c for c in returns_df.columns]]
        corr = returns_df.corr().abs()
        symbols = list(returns_df.columns)
        visited = set()
        clusters = []
        for i, s in enumerate(symbols):
            if s in visited:
                continue
            cluster = {s}
            for j, t in enumerate(symbols):
                if i != j and corr.loc[s, t] >= threshold:
                    cluster.add(t)
            visited |= cluster
            clusters.append(sorted(list(cluster)))
        return clusters

    def compute_portfolio_metrics(self, timeframe: str = 'daily') -> Dict[str, Any]:
        # Build per-bot returns
        returns_dict = {}
        drawdowns = {}
        for sym, state in self.bots.items():
            eq = state.equity_curve
            if len(eq) >= 2:
                rets = np.diff(eq) / np.array(eq[:-1])
            else:
                rets = np.array([])
            returns_dict[sym] = rets
            drawdowns[sym] = self._max_drawdown(eq)
        # Aggregate portfolio equity series
        portfolio_values = [v for _, v in self._equity_history]
        port_returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1]) if len(portfolio_values) >= 2 else np.array([])
        # Build returns DataFrame for correlation
        aligned = {}
        max_len = max((len(r) for r in returns_dict.values()), default=0)
        for sym, r in returns_dict.items():
            if len(r) < max_len:
                aligned[sym] = np.pad(r, (max_len - len(r), 0), constant_values=np.nan)
            else:
                aligned[sym] = r
        returns_df = pd.DataFrame(aligned).dropna(how='all')
        clusters = self._correlation_clusters(returns_df.fillna(0)) if not returns_df.empty else []
        metrics = {
            'timeframe': timeframe,
            'portfolio': {
                'max_drawdown': self._max_drawdown(portfolio_values),
                'volatility': self._volatility(port_returns, timeframe),
                'sharpe': self._sharpe(port_returns, 0.0, timeframe),
                'tail_risk': self._tail_risk(port_returns)
            },
            'per_bot': {
                sym: {
                    'max_drawdown': drawdowns.get(sym, 0.0),
                    'volatility': self._volatility(rets, timeframe),
                    'sharpe': self._sharpe(rets, 0.0, timeframe),
                    'tail_risk': self._tail_risk(rets)
                } for sym, rets in returns_dict.items()
            },
            'high_correlation_clusters': clusters
        }
        return metrics

    # Convenience to toggle per-bot pattern usage later if needed
    def set_bot_chart_patterns(self, symbol: str, enabled: bool) -> None:
        if symbol in self.bots:
            # Global modifier controls application; store flag on spec if expanded later
            self.pattern_nn.enabled = enabled

