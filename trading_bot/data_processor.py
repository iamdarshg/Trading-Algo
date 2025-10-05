
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import requests
import json
from datetime import datetime, timedelta
import yfinance as yf

class TradingDataProcessor:
    """Advanced data processor for trading models with technical indicators"""

    def __init__(self, 
                 sequence_length: int = 60,
                 prediction_horizon: int = 1,
                 include_technical_indicators: bool = True,
                 normalize_data: bool = True):

        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.include_technical_indicators = include_technical_indicators
        self.normalize_data = normalize_data

        self.price_scaler = StandardScaler() if normalize_data else None
        self.volume_scaler = StandardScaler() if normalize_data else None
        self.technical_scaler = StandardScaler() if normalize_data else None

        self.feature_names = []
        self.is_fitted = False

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        data = df.copy()

        # Moving averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()  # Requested 20d MA
        data['SMA_50'] = data['Close'].rolling(window=50).mean()  # Requested 50d MA
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()

        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_period = 20
        data['BB_Middle'] = data['Close'].rolling(window=bb_period).mean()
        bb_std = data['Close'].rolling(window=bb_period).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']

        # Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['Stoch_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

        # Average True Range (ATR)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data['ATR'] = true_range.rolling(window=14).mean()

        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']

        # Price momentum
        data['Momentum_1'] = data['Close'].pct_change(1)
        data['Momentum_5'] = data['Close'].pct_change(5)
        data['Momentum_10'] = data['Close'].pct_change(10)

        # Volatility
        data['Volatility_5'] = data['Close'].pct_change().rolling(window=5).std()
        data['Volatility_20'] = data['Close'].pct_change().rolling(window=20).std()

        # Support/Resistance levels (simplified)
        data['Support'] = data['Low'].rolling(window=20).min()
        data['Resistance'] = data['High'].rolling(window=20).max()
        data['Support_Distance'] = (data['Close'] - data['Support']) / data['Close']
        data['Resistance_Distance'] = (data['Resistance'] - data['Close']) / data['Close']

        return data

    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced features for non-differentiable analysis"""
        data = df.copy()

        # Local variation measures (non-differentiable friendly)
        data['Local_Range_5'] = data['High'].rolling(5).max() - data['Low'].rolling(5).min()
        data['Local_Range_20'] = data['High'].rolling(20).max() - data['Low'].rolling(20).min()

        # Jump detection (for discontinuities)
        returns = data['Close'].pct_change()
        rolling_std = returns.rolling(20).std()
        data['Jump_Indicator'] = np.abs(returns) > (3 * rolling_std)
        data['Jump_Magnitude'] = np.where(data['Jump_Indicator'], np.abs(returns), 0)

        # Regime change indicators
        data['Price_Regime'] = np.where(data['Close'] > data['SMA_50'], 1, 0)
        data['Volatility_Regime'] = np.where(data['Volatility_20'] > data['Volatility_20'].rolling(50).mean(), 1, 0)

        # Fractal dimension approximation
        def fractal_dimension(series, max_lag=10):
            """Approximate fractal dimension using box-counting method"""
            n = len(series)
            if n < max_lag * 2:
                return np.nan

            lags = range(2, min(max_lag, n//4))
            rs = []

            for lag in lags:
                # Calculate range for each lag
                mean_series = series.rolling(lag).mean()
                detrended = series - mean_series
                cumsum_detrended = detrended.cumsum()

                ranges = []
                for i in range(lag, len(series), lag):
                    segment = cumsum_detrended[i-lag:i]
                    if len(segment) > 0:
                        ranges.append(segment.max() - segment.min())

                if ranges:
                    mean_range = np.mean(ranges)
                    std_dev = series[lag:].rolling(lag).std().mean()
                    if std_dev > 0:
                        rs.append(mean_range / std_dev)

            if len(rs) < 2:
                return np.nan

            # Fit line to log(R/S) vs log(lag)
            log_lags = np.log(lags[:len(rs)])
            log_rs = np.log(rs)
            hurst = np.polyfit(log_lags, log_rs, 1)[0]

            # Convert to fractal dimension
            return 2 - hurst

        # Calculate fractal dimension over rolling windows
        data['Fractal_Dim_50'] = data['Close'].rolling(50).apply(lambda x: fractal_dimension(x), raw=False)

        return data

    def prepare_sequences(self, 
                         data: pd.DataFrame, 
                         target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training/inference"""

        # Select features
        feature_columns = []

        # Price features
        price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns.extend([col for col in price_features if col in data.columns])

        # Technical indicators
        if self.include_technical_indicators:
            technical_features = [
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI',
                'BB_Width', 'BB_Position', 'Stoch_K', 'Stoch_D', 'ATR',
                'Volume_Ratio', 'Momentum_1', 'Momentum_5', 'Momentum_10',
                'Volatility_5', 'Volatility_20', 'Support_Distance', 'Resistance_Distance',
                'Local_Range_5', 'Local_Range_20', 'Jump_Magnitude',
                'Price_Regime', 'Volatility_Regime', 'Fractal_Dim_50'
            ]
            feature_columns.extend([col for col in technical_features if col in data.columns])

        self.feature_names = feature_columns

        # Extract features and target
        features_data = data[feature_columns].fillna(method='ffill').fillna(0).values
        target_data = data[target_column].values

        # Normalize if required
        if self.normalize_data and not self.is_fitted:
            # Separate price, volume, and technical features for different scaling
            price_cols = [i for i, col in enumerate(feature_columns) if col in ['Open', 'High', 'Low', 'Close']]
            volume_cols = [i for i, col in enumerate(feature_columns) if col == 'Volume']
            technical_cols = [i for i, col in enumerate(feature_columns) if i not in price_cols + volume_cols]

            if price_cols:
                self.price_scaler.fit(features_data[:, price_cols])
            if volume_cols:
                self.volume_scaler.fit(features_data[:, volume_cols])
            if technical_cols:
                self.technical_scaler.fit(features_data[:, technical_cols])

            self.is_fitted = True

        if self.normalize_data:
            scaled_features = features_data.copy()

            if hasattr(self, 'price_scaler') and self.price_scaler is not None:
                price_cols = [i for i, col in enumerate(feature_columns) if col in ['Open', 'High', 'Low', 'Close']]
                if price_cols:
                    scaled_features[:, price_cols] = self.price_scaler.transform(features_data[:, price_cols])

            if hasattr(self, 'volume_scaler') and self.volume_scaler is not None:
                volume_cols = [i for i, col in enumerate(feature_columns) if col == 'Volume']
                if volume_cols:
                    scaled_features[:, volume_cols] = self.volume_scaler.transform(features_data[:, volume_cols])

            if hasattr(self, 'technical_scaler') and self.technical_scaler is not None:
                technical_cols = [i for i, col in enumerate(feature_columns) if i not in 
                                [j for j, col in enumerate(feature_columns) if col in ['Open', 'High', 'Low', 'Close', 'Volume']]]
                if technical_cols:
                    scaled_features[:, technical_cols] = self.technical_scaler.transform(features_data[:, technical_cols])

            features_data = scaled_features

        # Create sequences
        X, y = [], []

        for i in range(self.sequence_length, len(features_data) - self.prediction_horizon + 1):
            X.append(features_data[i-self.sequence_length:i])
            y.append(target_data[i + self.prediction_horizon - 1])

        return np.array(X), np.array(y)

    def process_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Complete data processing pipeline"""

        # Calculate technical indicators
        processed_data = self.calculate_technical_indicators(df)

        # Calculate advanced features
        processed_data = self.calculate_advanced_features(processed_data)

        # Prepare sequences
        X, y = self.prepare_sequences(processed_data)

        # Create train/validation split
        split_idx = int(0.8 * len(X))

        return {
            'X_train': X[:split_idx],
            'y_train': y[:split_idx],
            'X_val': X[split_idx:],
            'y_val': y[split_idx:],
            'feature_names': self.feature_names,
            'processed_data': processed_data
        }

class TextProcessor:
    """Simple text processor for trading prompts"""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.is_fitted = False

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from text corpus"""
        word_freq = {}

        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and take top vocab_size-2 words (reserve 0 for padding, 1 for unknown)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.vocab_size-2]

        # Build mappings
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}

        for i, (word, _) in enumerate(top_words):
            self.word_to_idx[word] = i + 2
            self.idx_to_word[i + 2] = word

        self.is_fitted = True

    def encode_text(self, text: str, max_length: int = 50) -> np.ndarray:
        """Encode text to token indices"""
        if not self.is_fitted:
            raise ValueError("Vocabulary not built. Call build_vocab first.")

        words = text.lower().split()
        tokens = []

        for word in words:
            tokens.append(self.word_to_idx.get(word, 1))  # 1 is <UNK>

        # Pad or truncate to max_length
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))  # 0 is <PAD>
        else:
            tokens = tokens[:max_length]

        return np.array(tokens)

def fetch_market_data(symbol: str, 
                     period: str = "2y", 
                     interval: str = "1d") -> pd.DataFrame:
    """Fetch market data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)

        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")

        # Reset index to make Date a column
        data.reset_index(inplace=True)

        return data

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def get_market_news(symbol: str, days_back: int = 7) -> List[str]:
    """Fetch recent market news (dummy implementation - replace with real news API)"""
    # This is a placeholder - in practice, you would use a news API like Alpha Vantage, NewsAPI, etc.
    dummy_news = [
        f"{symbol} shows strong momentum amid market volatility",
        f"Analysts upgrade {symbol} target price on positive earnings outlook",
        f"Market sentiment for {symbol} remains bullish despite sector headwinds",
        f"{symbol} technical indicators suggest potential breakout above resistance",
        f"Institutional buying pressure observed in {symbol} ahead of earnings"
    ]

    return dummy_news

# Example usage functions
def create_sample_dataset(symbol: str = "AAPL", period: str = "2y") -> Dict[str, any]:
    """Create a sample dataset for testing"""

    # Fetch market data
    data = fetch_market_data(symbol, period)

    if data.empty:
        return {}

    # Process data
    processor = TradingDataProcessor(
        sequence_length=60,
        prediction_horizon=1,
        include_technical_indicators=True,
        normalize_data=True
    )

    processed = processor.process_data(data)

    # Get sample news
    news = get_market_news(symbol)

    # Process text
    text_processor = TextProcessor()
    text_processor.build_vocab(news)

    # Encode news
    encoded_news = [text_processor.encode_text(text) for text in news]

    return {
        'processed_data': processed,
        'news': news,
        'encoded_news': encoded_news,
        'text_processor': text_processor,
        'data_processor': processor,
        'raw_data': data
    }
