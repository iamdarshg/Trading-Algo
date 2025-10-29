import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import yfinance as yf
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import hashlib
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler

# Absolute imports
from trading_bot.data_processor import TradingDataProcessor, TextProcessor
from config.config_manager import config_manager

@dataclass
class NewsArticle:
    """Enhanced news article structure"""
    title: str
    content: str
    published_at: datetime
    source: str
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    url: Optional[str] = None
    author: Optional[str] = None

class HistoricalNewsProvider:
    """Provider for historical news data with multiple sources"""
    
    def __init__(self):
        self.config = config_manager
        self._init_providers()
    
    def _init_providers(self):
        """Initialize available news providers"""
        self.providers = {
            'newsapi': self._newsapi_historical,
            'finnhub': self._finnhub_historical,
            'alpha_vantage': self._alpha_vantage_news,
        }
        
        # Check which providers are available
        self.available_providers = []
        for provider, func in self.providers.items():
            if self.config.is_service_enabled('news', provider):
                self.available_providers.append(provider)
    
    def get_historical_news(self, symbol: str, start_date: datetime, end_date: datetime, 
                           max_articles: int = 50) -> List[NewsArticle]:
        """Get historical news for a symbol between dates"""
        all_articles = []
        
        # Use multiple providers in parallel
        with ThreadPoolExecutor(max_workers=len(self.available_providers)) as executor:
            futures = {}
            
            for provider in self.available_providers:
                if provider in self.providers:
                    future = executor.submit(
                        self.providers[provider], symbol, start_date, end_date
                    )
                    futures[future] = provider
            
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    articles = future.result(timeout=30)
                    all_articles.extend(articles)
                except Exception as e:
                    print(f"Error fetching from {provider}: {e}")
        
        # Deduplicate and sort by date
        seen_titles = set()
        unique_articles = []
        
        for article in sorted(all_articles, key=lambda x: x.published_at, reverse=True):
            title_hash = hashlib.md5(article.title.lower().encode()).hexdigest()
            if title_hash not in seen_titles:
                seen_titles.add(title_hash)
                unique_articles.append(article)
        
        return unique_articles[:max_articles]
    
    def _newsapi_historical(self, symbol: str, start_date: datetime, end_date: datetime) -> List[NewsArticle]:
        """Fetch historical news from NewsAPI"""
        api_key = self.config.get_api_key('news', 'newsapi')
        if not api_key:
            return []
        
        # Get company name for better search
        try:
            ticker = yf.Ticker(symbol)
            company_name = ticker.info.get('longName', symbol)
        except:
            company_name = symbol
        
        articles = []
        
        # NewsAPI has a 1-month limit for historical data on free tier
        # Split into monthly chunks
        current_date = start_date
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=28), end_date)
            
            params = {
                'q': f'({company_name} OR {symbol}) AND (earnings OR revenue OR stock OR shares)',
                'from': current_date.strftime('%Y-%m-%d'),
                'to': chunk_end.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 100,
                'apiKey': api_key
            }
            
            try:
                response = requests.get(
                    'https://newsapi.org/v2/everything',
                    params=params,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok':
                        for article_data in data.get('articles', []):
                            try:
                                pub_date = datetime.fromisoformat(
                                    article_data['publishedAt'].replace('Z', '+00:00')
                                )
                                
                                article = NewsArticle(
                                    title=article_data.get('title', ''),
                                    content=article_data.get('description', ''),
                                    published_at=pub_date,
                                    source=article_data.get('source', {}).get('name', 'NewsAPI'),
                                    url=article_data.get('url'),
                                    author=article_data.get('author')
                                )
                                articles.append(article)
                            except Exception as e:
                                print(f"Error parsing article: {e}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching NewsAPI data: {e}")
            
            current_date = chunk_end
        
        return articles
    
    def _finnhub_historical(self, symbol: str, start_date: datetime, end_date: datetime) -> List[NewsArticle]:
        """Fetch historical news from Finnhub"""
        api_key = self.config.get_api_key('news', 'finnhub')
        if not api_key:
            return []
        
        articles = []
        
        try:
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': api_key
            }
            
            response = requests.get(
                'https://finnhub.io/api/v1/company-news',
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                for article_data in data:
                    try:
                        pub_date = datetime.fromtimestamp(article_data['datetime'])
                        
                        article = NewsArticle(
                            title=article_data.get('headline', ''),
                            content=article_data.get('summary', ''),
                            published_at=pub_date,
                            source='Finnhub',
                            url=article_data.get('url')
                        )
                        articles.append(article)
                    except Exception as e:
                        print(f"Error parsing Finnhub article: {e}")
        
        except Exception as e:
            print(f"Error fetching Finnhub data: {e}")
        
        return articles
    
    def _alpha_vantage_news(self, symbol: str, start_date: datetime, end_date: datetime) -> List[NewsArticle]:
        """Fetch news from Alpha Vantage"""
        api_key = self.config.get_api_key('news', 'alpha_vantage')
        if not api_key:
            return []
        
        articles = []
        
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'time_from': start_date.strftime('%Y%m%dT%H%M'),
                'time_to': end_date.strftime('%Y%m%dT%H%M'),
                'apikey': api_key
            }
            
            response = requests.get(
                'https://www.alphavantage.co/query',
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                for article_data in data.get('feed', []):
                    try:
                        pub_date = datetime.strptime(
                            article_data['time_published'], '%Y%m%dT%H%M%S'
                        )
                        
                        # Extract sentiment for the specific ticker
                        sentiment_score = None
                        for ticker_data in article_data.get('ticker_sentiment', []):
                            if ticker_data.get('ticker') == symbol:
                                sentiment_score = float(ticker_data.get('relevance_score', 0))
                                break
                        
                        article = NewsArticle(
                            title=article_data.get('title', ''),
                            content=article_data.get('summary', ''),
                            published_at=pub_date,
                            source=article_data.get('source', 'Alpha Vantage'),
                            sentiment_score=sentiment_score,
                            url=article_data.get('url')
                        )
                        articles.append(article)
                    except Exception as e:
                        print(f"Error parsing Alpha Vantage article: {e}")
        
        except Exception as e:
            print(f"Error fetching Alpha Vantage data: {e}")
        
        return articles

class EnhancedDataProcessor(TradingDataProcessor):
    """Enhanced data processor with optimizations and historical news"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.news_provider = HistoricalNewsProvider()
        self.feature_cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
        # Use RobustScaler for better outlier handling
        if self.normalize_data:
            self.price_scaler = RobustScaler()
            self.volume_scaler = RobustScaler()
            self.technical_scaler = RobustScaler()
    
    @lru_cache(maxsize=128)
    def _cached_technical_indicators(self, data_hash: str, df_pkl: bytes) -> pd.DataFrame:
        """Cached technical indicator calculation"""
        df = pd.read_pickle(df_pkl)
        return super().calculate_technical_indicators(df)
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced technical indicators with caching"""
        # Create hash for caching
        data_hash = hashlib.md5(df.to_json().encode()).hexdigest()[:16]
        
        # Check cache
        if data_hash in self.feature_cache:
            cached_time, cached_result = self.feature_cache[data_hash]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result
        
        # Calculate indicators
        result = super().calculate_technical_indicators(df)
        
        # Add advanced technical indicators
        result = self._add_advanced_indicators(result)
        
        # Cache result
        self.feature_cache[data_hash] = (time.time(), result)
        
        return result
    
    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators for better prediction"""
        data = df.copy()
        
        # Ichimoku Cloud
        high_9 = data['High'].rolling(window=9).max()
        low_9 = data['Low'].rolling(window=9).min()
        data['Ichimoku_Tenkan'] = (high_9 + low_9) / 2
        
        high_26 = data['High'].rolling(window=26).max()
        low_26 = data['Low'].rolling(window=26).min()
        data['Ichimoku_Kijun'] = (high_26 + low_26) / 2
        
        data['Ichimoku_Senkou_A'] = ((data['Ichimoku_Tenkan'] + data['Ichimoku_Kijun']) / 2).shift(26)
        
        high_52 = data['High'].rolling(window=52).max()
        low_52 = data['Low'].rolling(window=52).min()
        data['Ichimoku_Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
        
        # Williams %R
        data['Williams_R'] = (
            (data['High'].rolling(14).max() - data['Close']) /
            (data['High'].rolling(14).max() - data['Low'].rolling(14).min())
        ) * -100
        
        # Commodity Channel Index (CCI)
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        data['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # Chaikin Money Flow
        mfv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']) * data['Volume']
        data['CMF'] = mfv.rolling(20).sum() / data['Volume'].rolling(20).sum()
        
        # Rate of Change
        data['ROC_12'] = data['Close'].pct_change(12) * 100
        data['ROC_25'] = data['Close'].pct_change(25) * 100
        
        # Price relative to moving averages
        data['Price_vs_SMA20'] = data['Close'] / data['SMA_20'] - 1
        data['Price_vs_SMA50'] = data['Close'] / data['SMA_50'] - 1
        
        # Volume indicators
        data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        data['Volume_Price_Trend'] = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * data['Volume']).fillna(0).cumsum()
        
        # Volatility indicators
        data['Historical_Volatility'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        return data
    
    def get_historical_news_context(self, symbol: str, date: datetime, lookback_days: int = 7) -> List[NewsArticle]:
        """Get historical news context for a specific date"""
        start_date = date - timedelta(days=lookback_days)
        end_date = date + timedelta(days=1)
        
        return self.news_provider.get_historical_news(symbol, start_date, end_date)
    
    def create_news_features(self, articles: List[NewsArticle]) -> Dict[str, float]:
        """Create numerical features from news articles"""
        if not articles:
            return {
                'news_count': 0,
                'avg_sentiment': 0,
                'news_relevance': 0,
                'news_recency': 0
            }
        
        # Count of articles
        news_count = len(articles)
        
        # Average sentiment (if available)
        sentiment_scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Relevance based on title/content keywords
        relevance_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
            'buy', 'sell', 'upgrade', 'downgrade', 'target', 'price',
            'merger', 'acquisition', 'partnership', 'lawsuit', 'regulation'
        ]
        
        total_relevance = 0
        for article in articles:
            text = (article.title + ' ' + article.content).lower()
            relevance = sum(1 for keyword in relevance_keywords if keyword in text)
            total_relevance += relevance
        
        avg_relevance = total_relevance / len(articles) if articles else 0
        
        # Recency score (more recent = higher score)
        now = datetime.now()
        recency_scores = []
        for article in articles:
            hours_old = (now - article.published_at).total_seconds() / 3600
            recency_score = max(0, 1 - hours_old / (7 * 24))  # Decay over 7 days
            recency_scores.append(recency_score)
        
        avg_recency = np.mean(recency_scores) if recency_scores else 0
        
        return {
            'news_count': min(news_count / 10, 1),  # Normalize to 0-1
            'avg_sentiment': avg_sentiment,
            'news_relevance': min(avg_relevance / 5, 1),  # Normalize to 0-1
            'news_recency': avg_recency
        }
    
    def process_data_with_news(self, df: pd.DataFrame, symbol: str, 
                              include_historical_news: bool = True) -> Dict[str, np.ndarray]:
        """Enhanced data processing with news features"""
        # Base processing
        result = self.process_data(df)
        
        if include_historical_news:
            # Add news features for each date
            news_features_list = []
            
            for idx, row in df.iterrows():
                date = row.name if hasattr(row.name, 'date') else datetime.now()
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                
                # Get news for this date
                news_articles = self.get_historical_news_context(symbol, date)
                news_features = self.create_news_features(news_articles)
                news_features_list.append(list(news_features.values()))
            
            # Convert to numpy array
            news_array = np.array(news_features_list)
            
            # Pad/trim to match the processed data length
            if len(news_array) > len(result['X_train']) + len(result['X_val']):
                news_array = news_array[-len(result['X_train']) - len(result['X_val']):]
            elif len(news_array) < len(result['X_train']) + len(result['X_val']):
                # Pad with zeros
                padding = np.zeros((len(result['X_train']) + len(result['X_val']) - len(news_array), news_array.shape[1]))
                news_array = np.vstack([padding, news_array])
            
            # Split into train/val
            split_idx = len(result['X_train'])
            news_train = news_array[:split_idx]
            news_val = news_array[split_idx:]
            
            # Add to result
            result['news_features_train'] = news_train
            result['news_features_val'] = news_val
            result['news_feature_names'] = ['news_count', 'avg_sentiment', 'news_relevance', 'news_recency']
        
        return result
    
    def create_enhanced_dataset(self, symbol: str, period: str = "2y", 
                               interval: str = "1d", include_news: bool = True) -> Dict[str, any]:
        """Create enhanced dataset with optimized processing and news"""
        # Fetch market data
        from trading_bot.data_processor import fetch_market_data
        data = fetch_market_data(symbol, period, interval)
        
        if data.empty:
            return {}
        
        # Process with news if requested
        if include_news:
            processed = self.process_data_with_news(data, symbol)
        else:
            processed = self.process_data(data)
        
        # Enhanced text processor with more sophisticated tokenization
        from trading_bot.enhanced_text_processor import EnhancedTextProcessor
        text_processor = EnhancedTextProcessor()
        
        # Get sample news for vocabulary building
        recent_articles = self.news_provider.get_historical_news(
            symbol, 
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        
        if recent_articles:
            news_texts = [article.title + ' ' + article.content for article in recent_articles]
            text_processor.build_vocab(news_texts)
        
        return {
            'processed_data': processed,
            'news_articles': recent_articles if include_news else [],
            'text_processor': text_processor,
            'data_processor': self,
            'raw_data': data
        }