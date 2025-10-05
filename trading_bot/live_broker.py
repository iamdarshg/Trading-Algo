
import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import yfinance as yf
import time
from dataclasses import dataclass

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    position_type: str  # 'long' or 'short'

    @property
    def unrealized_pnl(self) -> float:
        if self.position_type == 'long':
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_percent(self) -> float:
        return (self.unrealized_pnl / (self.entry_price * abs(self.quantity))) * 100

@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    quantity: float
    order_type: str  # 'market', 'limit', 'stop'
    side: str  # 'buy' or 'sell'
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'DAY'  # 'DAY', 'GTC', 'IOC'
    order_id: Optional[str] = None
    status: str = 'pending'
    created_at: Optional[datetime] = None

class BaseBroker(ABC):
    """Abstract base class for all brokers"""

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def place_order(self, order: Order) -> str:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        pass

class LiveDataBroker(BaseBroker):
    """Broker with live internet data access but dummy trading functions"""

    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict] = []
        self.order_counter = 0

        # Price cache to avoid too many API calls
        self.price_cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_duration = timedelta(seconds=15)  # Cache prices for 15 seconds

    def get_current_price(self, symbol: str) -> float:
        """Get real-time price from Yahoo Finance"""
        current_time = datetime.now()

        # Check cache first
        if symbol in self.price_cache:
            price, timestamp = self.price_cache[symbol]
            if current_time - timestamp < self.cache_duration:
                return price

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Try different price fields
            price = None
            for price_field in ['currentPrice', 'regularMarketPrice', 'previousClose']:
                if price_field in info and info[price_field] is not None:
                    price = float(info[price_field])
                    break

            if price is None:
                # Fallback to recent data
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    price = float(data['Close'].iloc[-1])

            if price is not None:
                # Update cache
                self.price_cache[symbol] = (price, current_time)
                return price
            else:
                raise ValueError(f"Could not fetch price for {symbol}")

        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            # Return cached price if available, otherwise raise error
            if symbol in self.price_cache:
                return self.price_cache[symbol][0]
            raise

    def get_market_data(self, symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
        """Get historical market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()

    def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental financial data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract key financial metrics
            financial_data = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': info.get('returnOnEquity'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'profit_margin': info.get('profitMargins'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'average_volume': info.get('averageVolume')
            }

            return {k: v for k, v in financial_data.items() if v is not None}

        except Exception as e:
            print(f"Error fetching financial data for {symbol}: {e}")
            return {}

    def get_market_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Get recent market news for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            processed_news = []
            for article in news[:10]:  # Limit to 10 articles
                processed_news.append({
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'url': article.get('link', ''),
                    'published': article.get('providerPublishTime'),
                    'source': article.get('publisher', '')
                })

            return processed_news
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []

    def place_order(self, order: Order) -> str:
        """Place a trading order (DUMMY IMPLEMENTATION)"""
        print("\n" + "="*50)
        print("🚨 DUMMY TRADING FUNCTION CALLED 🚨")
        print("="*50)
        print(f"ORDER DETAILS:")
        print(f"Symbol: {order.symbol}")
        print(f"Side: {order.side.upper()}")
        print(f"Quantity: {order.quantity}")
        print(f"Type: {order.order_type}")

        if order.price:
            print(f"Price: ${order.price:.2f}")
        if order.stop_price:
            print(f"Stop Price: ${order.stop_price:.2f}")

        print(f"Time in Force: {order.time_in_force}")
        print("="*50)
        print("⚠️  REPLACE THIS FUNCTION WITH REAL BROKER API ⚠️")
        print("="*50)

        # Generate order ID
        self.order_counter += 1
        order_id = f"ORDER_{self.order_counter:06d}"
        order.order_id = order_id
        order.status = "submitted"
        order.created_at = datetime.now()

        # Store order
        self.orders[order_id] = order

        # For simulation purposes, execute market orders immediately
        if order.order_type == 'market':
            self._execute_order_simulation(order)

        return order_id

    def _execute_order_simulation(self, order: Order):
        """Simulate order execution for testing"""
        try:
            current_price = self.get_current_price(order.symbol)
            execution_price = current_price  # Simplified - no slippage

            if order.side == 'buy':
                cost = execution_price * order.quantity
                if cost <= self.current_balance:
                    self.current_balance -= cost

                    # Add/update position
                    if order.symbol in self.positions:
                        existing_pos = self.positions[order.symbol]
                        total_cost = (existing_pos.entry_price * existing_pos.quantity) + cost
                        total_quantity = existing_pos.quantity + order.quantity
                        avg_price = total_cost / total_quantity

                        self.positions[order.symbol] = Position(
                            symbol=order.symbol,
                            quantity=total_quantity,
                            entry_price=avg_price,
                            current_price=current_price,
                            entry_time=existing_pos.entry_time,
                            position_type='long'
                        )
                    else:
                        self.positions[order.symbol] = Position(
                            symbol=order.symbol,
                            quantity=order.quantity,
                            entry_price=execution_price,
                            current_price=current_price,
                            entry_time=datetime.now(),
                            position_type='long'
                        )

                    order.status = 'filled'
                    print(f"✅ BUY order filled: {order.quantity} shares of {order.symbol} at ${execution_price:.2f}")
                else:
                    order.status = 'rejected'
                    print(f"❌ BUY order rejected: Insufficient funds")

            elif order.side == 'sell':
                if order.symbol in self.positions:
                    position = self.positions[order.symbol]
                    if position.quantity >= order.quantity:
                        # Execute sell
                        proceeds = execution_price * order.quantity
                        self.current_balance += proceeds

                        # Update position
                        remaining_quantity = position.quantity - order.quantity
                        if remaining_quantity > 0:
                            self.positions[order.symbol].quantity = remaining_quantity
                        else:
                            del self.positions[order.symbol]

                        order.status = 'filled'
                        print(f"✅ SELL order filled: {order.quantity} shares of {order.symbol} at ${execution_price:.2f}")
                    else:
                        order.status = 'rejected'
                        print(f"❌ SELL order rejected: Insufficient shares")
                else:
                    order.status = 'rejected'
                    print(f"❌ SELL order rejected: No position in {order.symbol}")

            # Record trade
            if order.status == 'filled':
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': order.quantity,
                    'price': execution_price,
                    'order_id': order.order_id
                })

        except Exception as e:
            order.status = 'error'
            print(f"❌ Order execution error: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order (DUMMY IMPLEMENTATION)"""
        print(f"🚨 DUMMY CANCEL FUNCTION: Canceling order {order_id}")
        print("⚠️  REPLACE THIS FUNCTION WITH REAL BROKER API ⚠️")

        if order_id in self.orders:
            if self.orders[order_id].status == 'pending':
                self.orders[order_id].status = 'cancelled'
                return True
        return False

    def get_positions(self) -> List[Position]:
        """Get current positions"""
        # Update current prices for all positions
        for symbol, position in self.positions.items():
            try:
                position.current_price = self.get_current_price(symbol)
            except:
                pass  # Keep old price if update fails

        return list(self.positions.values())

    def get_account_balance(self) -> float:
        """Get current account balance"""
        return self.current_balance

    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        total_value = self.current_balance

        for position in self.get_positions():
            total_value += position.current_price * position.quantity

        return total_value

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        positions = self.get_positions()
        total_value = self.get_portfolio_value()

        total_pnl = sum(pos.unrealized_pnl for pos in positions)
        total_pnl_percent = ((total_value - self.initial_balance) / self.initial_balance) * 100

        return {
            'cash_balance': self.current_balance,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'num_positions': len(positions),
            'positions': [
                {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_percent': pos.unrealized_pnl_percent
                } for pos in positions
            ]
        }

# Helper functions for common trading operations
def create_market_order(symbol: str, quantity: float, side: str) -> Order:
    """Create a market order"""
    return Order(
        symbol=symbol,
        quantity=abs(quantity),
        order_type='market',
        side=side.lower()
    )

def create_limit_order(symbol: str, quantity: float, side: str, price: float) -> Order:
    """Create a limit order"""
    return Order(
        symbol=symbol,
        quantity=abs(quantity),
        order_type='limit',
        side=side.lower(),
        price=price
    )

def create_stop_loss_order(symbol: str, quantity: float, side: str, stop_price: float) -> Order:
    """Create a stop loss order"""
    return Order(
        symbol=symbol,
        quantity=abs(quantity),
        order_type='stop',
        side=side.lower(),
        stop_price=stop_price
    )
