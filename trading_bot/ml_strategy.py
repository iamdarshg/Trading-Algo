import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from abc import ABC, abstractmethod

# Absolute imports
from trading_bot.live_broker import LiveDataBroker, Order, create_market_order, create_limit_order
from trading_bot.data_processor import TradingDataProcessor, TextProcessor

class TradingSignal:
    """Represents a trading signal"""
    def __init__(self, 
                 symbol: str,
                 signal_type: str,  # 'buy', 'sell', 'hold'
                 confidence: float,
                 price_target: Optional[float] = None,
                 stop_loss: Optional[float] = None,
                 reasoning: str = ""):

        self.symbol = symbol
        self.signal_type = signal_type
        self.confidence = confidence
        self.price_target = price_target
        self.stop_loss = stop_loss
        self.reasoning = reasoning
        self.timestamp = datetime.now()

class BaseStrategy(ABC):
    """Base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name
        self.signals_history = []

    @abstractmethod
    def generate_signal(self, data: Dict[str, Any]) -> TradingSignal:
        pass

class AdvancedMLStrategy(BaseStrategy):
    """Advanced ML-based trading strategy using our neural architectures"""

    def __init__(self, 
                 model: torch.nn.Module,
                 data_processor: TradingDataProcessor,
                 text_processor: Optional[TextProcessor] = None,
                 confidence_threshold: float = 0.6,
                 position_sizing: str = 'fixed',  # 'fixed', 'volatility', 'kelly'
                 max_position_size: float = 0.1):  # 10% of portfolio

        super().__init__("AdvancedMLStrategy")

        self.model = model
        self.model.eval()

        self.data_processor = data_processor
        self.text_processor = text_processor
        self.confidence_threshold = confidence_threshold
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def preprocess_data(self, 
                       price_data: pd.DataFrame,
                       text_prompt: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Preprocess data for model inference"""

        # Process price data
        processed_data = self.data_processor.calculate_technical_indicators(price_data)
        processed_data = self.data_processor.calculate_advanced_features(processed_data)

        # Extract features
        feature_columns = self.data_processor.feature_names
        if not feature_columns:
            # Fallback to basic features
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        features_data = processed_data[feature_columns].fillna(method='ffill').fillna(0).values

        # Normalize using fitted scalers
        if self.data_processor.normalize_data and self.data_processor.is_fitted:
            # Apply same normalization as during training
            price_cols = [i for i, col in enumerate(feature_columns) if col in ['Open', 'High', 'Low', 'Close']]
            volume_cols = [i for i, col in enumerate(feature_columns) if col == 'Volume']
            technical_cols = [i for i, col in enumerate(feature_columns) if i not in price_cols + volume_cols]

            scaled_features = features_data.copy()

            if price_cols and self.data_processor.price_scaler is not None:
                scaled_features[:, price_cols] = self.data_processor.price_scaler.transform(features_data[:, price_cols])

            if volume_cols and self.data_processor.volume_scaler is not None:
                scaled_features[:, volume_cols] = self.data_processor.volume_scaler.transform(features_data[:, volume_cols])

            if technical_cols and self.data_processor.technical_scaler is not None:
                scaled_features[:, technical_cols] = self.data_processor.technical_scaler.transform(features_data[:, technical_cols])

            features_data = scaled_features

        # Get the last sequence_length points
        sequence_length = self.data_processor.sequence_length
        if len(features_data) >= sequence_length:
            sequence_data = features_data[-sequence_length:]
        else:
            # Pad if insufficient data
            padding_length = sequence_length - len(features_data)
            padding = np.zeros((padding_length, features_data.shape[1]))
            sequence_data = np.vstack([padding, features_data])

        # Convert to tensor
        price_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)  # Add batch dimension

        # Process text if provided
        text_tensor = None
        if text_prompt and self.text_processor:
            encoded_text = self.text_processor.encode_text(text_prompt)
            text_tensor = torch.LongTensor(encoded_text).unsqueeze(0).to(self.device)

        return price_tensor, text_tensor

    def predict(self, 
               price_data: pd.DataFrame,
               text_prompt: Optional[str] = None) -> Tuple[float, float]:
        """Make prediction using the model"""

        with torch.no_grad():
            price_tensor, text_tensor = self.preprocess_data(price_data, text_prompt)

            # Model prediction
            if text_tensor is not None and hasattr(self.model, 'forward'):
                # Check if model accepts text input
                try:
                    prediction = self.model(price_tensor, text_tensor)
                except:
                    # Fallback to price-only prediction
                    prediction = self.model(price_tensor)
            else:
                prediction = self.model(price_tensor)

            # Extract prediction value
            pred_value = prediction.cpu().numpy()[0]

            # Calculate confidence based on prediction magnitude
            # Higher absolute values indicate higher confidence
            confidence = min(1.0, abs(pred_value) / 0.1)  # Scale factor can be adjusted

            return float(pred_value), float(confidence)

    def generate_signal(self, data: Dict[str, Any]) -> TradingSignal:
        """Generate trading signal based on model prediction"""

        symbol = data.get('symbol', 'UNKNOWN')
        price_data = data.get('price_data')
        text_prompt = data.get('text_prompt', "")
        current_price = data.get('current_price', 0.0)

        if price_data is None:
            return TradingSignal(symbol, 'hold', 0.0, reasoning="No price data available")

        try:
            # Make prediction
            prediction, confidence = self.predict(price_data, text_prompt)

            # Determine signal type
            if confidence < self.confidence_threshold:
                signal_type = 'hold'
                reasoning = f"Low confidence ({confidence:.3f} < {self.confidence_threshold})"

            elif prediction > 0.02:  # Bullish threshold (2% expected return)
                signal_type = 'buy'
                price_target = current_price * (1 + abs(prediction))
                stop_loss = current_price * 0.95  # 5% stop loss
                reasoning = f"Bullish prediction: {prediction:.3f}, confidence: {confidence:.3f}"

            elif prediction < -0.02:  # Bearish threshold
                signal_type = 'sell'
                price_target = current_price * (1 - abs(prediction))
                stop_loss = current_price * 1.05  # 5% stop loss for short
                reasoning = f"Bearish prediction: {prediction:.3f}, confidence: {confidence:.3f}"

            else:
                signal_type = 'hold'
                reasoning = f"Neutral prediction: {prediction:.3f}"

            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price_target=price_target if signal_type != 'hold' else None,
                stop_loss=stop_loss if signal_type != 'hold' else None,
                reasoning=reasoning
            )

        except Exception as e:
            return TradingSignal(
                symbol=symbol,
                signal_type='hold',
                confidence=0.0,
                reasoning=f"Error generating signal: {str(e)}"
            )
import os
class TradingBot:
    """Main trading bot orchestrator"""

    def __init__(self,
                 strategy: BaseStrategy,
                 broker: LiveDataBroker,
                 symbols: List[str],
                 update_interval: int = 300,  # 5 minutes
                 max_positions: int = 5,
                 risk_management: bool = True):

        self.strategy = strategy
        self.broker = broker
        self.symbols = symbols
        self.update_interval = update_interval
        self.max_positions = max_positions
        self.risk_management = risk_management

        self.running = False
        self.last_update = {}
        try:
            os.makedirs('logs', exist_ok=True)
            os.makefile('logs/trading_bot.log', exist_ok=True)
        except Exception as e:
            print(f"Error creating log directory or file: {e}")
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def should_update_symbol(self, symbol: str) -> bool:
        """Check if symbol should be updated"""
        if symbol not in self.last_update:
            return True

        time_since_update = (datetime.now() - self.last_update[symbol]).total_seconds()
        return time_since_update >= self.update_interval

    def get_symbol_data(self, symbol: str, lookback_days: int = 60) -> Dict[str, Any]:
        """Get comprehensive data for a symbol"""
        try:
            # Get current price
            current_price = self.broker.get_current_price(symbol)

            # Get historical data
            price_data = self.broker.get_market_data(symbol, period=f"{lookback_days}d", interval="1d")

            if price_data.empty:
                raise ValueError("No price data available")

            # Get recent news for text prompt
            news = self.broker.get_market_news(symbol)
            text_prompt = ""
            if news:
                # Combine recent news headlines
                headlines = [article.get('title', '') for article in news[:3]]
                text_prompt = ". ".join(headlines)

            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_data': price_data,
                'text_prompt': text_prompt,
                'news': news
            }

        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {e}")
            return {}

    def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a trading signal"""
        if signal.signal_type == 'hold':
            return True

        try:
            portfolio_value = self.broker.get_portfolio_value()

            if signal.signal_type == 'buy':
                # Calculate position size
                position_value = portfolio_value * 0.1  # Default 10%

                current_price = self.broker.get_current_price(signal.symbol)
                quantity = position_value / current_price

                # Create and place order
                order = create_market_order(signal.symbol, quantity, 'buy')
                order_id = self.broker.place_order(order)

                self.logger.info(f"BUY order placed: {quantity:.2f} shares of {signal.symbol} | Order ID: {order_id}")
                return True

            elif signal.signal_type == 'sell':
                # Check if we have a position to sell
                positions = self.broker.get_positions()
                position_to_sell = None

                for pos in positions:
                    if pos.symbol == signal.symbol and pos.position_type == 'long':
                        position_to_sell = pos
                        break

                if position_to_sell:
                    # Sell existing position
                    order = create_market_order(signal.symbol, position_to_sell.quantity, 'sell')
                    order_id = self.broker.place_order(order)

                    self.logger.info(f"SELL order placed: {position_to_sell.quantity:.2f} shares of {signal.symbol} | Order ID: {order_id}")
                    return True
                else:
                    self.logger.warning(f"No position to sell for {signal.symbol}")
                    return False

        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return False

        return False

    def run_single_iteration(self):
        """Run a single iteration of the trading loop"""
        for symbol in self.symbols:
            if not self.should_update_symbol(symbol):
                continue

            try:
                # Get data
                data = self.get_symbol_data(symbol)
                if not data:
                    continue

                # Generate signal
                signal = self.strategy.generate_signal(data)

                # Log signal
                self.logger.info(f"Signal for {symbol}: {signal.signal_type.upper()} | "
                               f"Confidence: {signal.confidence:.3f} | "
                               f"Reasoning: {signal.reasoning}")

                # Execute if not hold
                if signal.signal_type != 'hold':
                    success = self.execute_signal(signal)
                    if success:
                        self.strategy.signals_history.append(signal)

                # Update last update time
                self.last_update[symbol] = datetime.now()

            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading bot performance summary"""
        portfolio = self.broker.get_portfolio_summary()

        signals_summary = {
            'total_signals': len(self.strategy.signals_history),
            'buy_signals': len([s for s in self.strategy.signals_history if s.signal_type == 'buy']),
            'sell_signals': len([s for s in self.strategy.signals_history if s.signal_type == 'sell']),
            'avg_confidence': np.mean([s.confidence for s in self.strategy.signals_history]) if self.strategy.signals_history else 0.0
        }

        return {
            'portfolio': portfolio,
            'signals': signals_summary,
            'last_update_times': self.last_update
        }

def create_trading_bot(strategy: BaseStrategy, broker: LiveDataBroker, symbols: List[str], update_interval: int = 300, max_positions: int = 5, risk_management: bool = True) -> TradingBot:
    """
    Factory function to create a TradingBot instance.

    Args:
        strategy (BaseStrategy): The trading strategy to use.
        broker (LiveDataBroker): The broker for live data and order execution.
        symbols (List[str]): List of trading symbols.
        update_interval (int): Interval in seconds for updates. Default is 300.
        max_positions (int): Maximum number of positions. Default is 5.
        risk_management (bool): Whether to enable risk management. Default is True.

    Returns:
        TradingBot: An instance of the TradingBot class.
    """
    return TradingBot(
        strategy=strategy,
        broker=broker,
        symbols=symbols,
        update_interval=update_interval,
        max_positions=max_positions,
        risk_management=risk_management
    )
