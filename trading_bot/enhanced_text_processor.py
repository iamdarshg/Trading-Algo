import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class EnhancedTextProcessor:
    """Advanced text processor with sentiment analysis and embeddings"""
    
    def __init__(self, vocab_size: int = 10000, use_pretrained: bool = False, model_name: str = 'distilbert-base-uncased'):
        self.vocab_size = vocab_size
        self.use_pretrained = use_pretrained
        self.model_name = model_name
        
        # Traditional tokenization
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.is_fitted = False
        
        # NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.stop_words = set()
            self.lemmatizer = None
            self.sentiment_analyzer = None
        
        # Financial keywords for domain-specific processing
        self.financial_keywords = {
            'positive': [
                'profit', 'revenue', 'growth', 'increase', 'rise', 'gain', 'bullish',
                'upgrade', 'buy', 'outperform', 'strong', 'beat', 'exceed', 'surge',
                'rally', 'breakthrough', 'expansion', 'acquisition', 'merger'
            ],
            'negative': [
                'loss', 'decline', 'decrease', 'fall', 'drop', 'bearish', 'downgrade',
                'sell', 'underperform', 'weak', 'miss', 'disappoint', 'crash', 'plunge',
                'recession', 'bankruptcy', 'lawsuit', 'investigation', 'scandal'
            ],
            'neutral': [
                'maintain', 'hold', 'neutral', 'stable', 'flat', 'unchanged',
                'meeting', 'conference', 'announcement', 'report', 'statement'
            ]
        }
        
        # Pretrained model components
        if use_pretrained:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
            except Exception as e:
                print(f"Could not load pretrained model {model_name}: {e}")
                self.use_pretrained = False
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, email addresses, and social media handles
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^a-zA-Z0-9\s\$\%]', ' ', text)
        
        # Normalize financial expressions
        text = re.sub(r'\$([0-9]+(?:\.[0-9]+)?[kmb]?)', r'dollar_\1', text)
        text = re.sub(r'([0-9]+(?:\.[0-9]+)?)%', r'\1_percent', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_financial_sentiment(self, text: str) -> Dict[str, float]:
        """Extract financial sentiment using domain-specific keywords"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.financial_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.financial_keywords['negative'] if word in text_lower)
        neutral_count = sum(1 for word in self.financial_keywords['neutral'] if word in text_lower)
        
        total_count = positive_count + negative_count + neutral_count
        
        if total_count == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        sentiment = {
            'positive': positive_count / total_count,
            'negative': negative_count / total_count,
            'neutral': neutral_count / total_count,
            'compound': (positive_count - negative_count) / total_count
        }
        
        # Use VADER if available for additional sentiment
        if self.sentiment_analyzer:
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            # Combine financial keywords with VADER (weighted average)
            for key in ['positive', 'negative', 'neutral', 'compound']:
                if key in vader_scores:
                    sentiment[key] = 0.7 * sentiment[key] + 0.3 * vader_scores[key]
        
        return sentiment
    
    def tokenize_advanced(self, text: str) -> List[str]:
        """Advanced tokenization with financial domain awareness"""
        text = self.preprocess_text(text)
        
        try:
            # Use NLTK tokenizer if available
            if word_tokenize:
                tokens = word_tokenize(text)
            else:
                tokens = text.split()
        except:
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                if self.lemmatizer:
                    try:
                        token = self.lemmatizer.lemmatize(token)
                    except:
                        pass
                processed_tokens.append(token)
        
        return processed_tokens
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary with enhanced processing"""
        word_freq = Counter()
        
        for text in texts:
            tokens = self.tokenize_advanced(text)
            word_freq.update(tokens)
        
        # Reserve special tokens
        self.word_to_idx = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3}
        self.idx_to_word = {0: '<pad>', 1: '<unk>', 2: '<start>', 3: '<end>'}
        
        # Add most frequent words
        most_common = word_freq.most_common(self.vocab_size - 4)
        for i, (word, freq) in enumerate(most_common):
            idx = i + 4
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.is_fitted = True
    
    def encode_text(self, text: str, max_length: int = 50, add_special_tokens: bool = True) -> np.ndarray:
        """Enhanced text encoding"""
        if not self.is_fitted:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        tokens = self.tokenize_advanced(text)
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.word_to_idx.get('<start>', 2))
        
        for token in tokens:
            token_ids.append(self.word_to_idx.get(token, 1))  # 1 is <unk>
        
        if add_special_tokens:
            token_ids.append(self.word_to_idx.get('<end>', 3))
        
        # Pad or truncate
        if len(token_ids) < max_length:
            token_ids.extend([0] * (max_length - len(token_ids)))  # 0 is <pad>
        else:
            token_ids = token_ids[:max_length]
        
        return np.array(token_ids)
    
    def encode_with_pretrained(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Encode text using pretrained transformer model"""
        if not self.use_pretrained:
            raise ValueError("Pretrained model not available")
        
        text = self.preprocess_text(text)
        
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get contextual embeddings from pretrained model"""
        if not self.use_pretrained:
            raise ValueError("Pretrained model not available")
        
        encoded = self.encode_with_pretrained(text)
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use CLS token embedding or mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        
        return embeddings
    
    def create_text_features(self, text: str) -> Dict[str, float]:
        """Create comprehensive text features"""
        if not text:
            return {
                'length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'sentiment_positive': 0,
                'sentiment_negative': 0,
                'sentiment_neutral': 0,
                'sentiment_compound': 0,
                'financial_keywords': 0,
                'readability': 0
            }
        
        # Basic features
        char_length = len(text)
        words = self.tokenize_advanced(text)
        word_count = len(words)
        
        try:
            sentences = sent_tokenize(text) if sent_tokenize else text.split('.')
            sentence_count = len(sentences)
        except:
            sentence_count = text.count('.') + 1
        
        # Sentiment features
        sentiment = self.extract_financial_sentiment(text)
        
        # Financial keyword density
        all_financial_words = []
        for word_list in self.financial_keywords.values():
            all_financial_words.extend(word_list)
        
        financial_count = sum(1 for word in words if word in all_financial_words)
        financial_density = financial_count / max(word_count, 1)
        
        # Simple readability score (average words per sentence)
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        readability = 1.0 / (1.0 + avg_words_per_sentence / 20.0)  # Normalize
        
        return {
            'length': min(char_length / 1000, 1.0),  # Normalize to 0-1
            'word_count': min(word_count / 100, 1.0),  # Normalize to 0-1
            'sentence_count': min(sentence_count / 10, 1.0),  # Normalize to 0-1
            'sentiment_positive': sentiment['positive'],
            'sentiment_negative': sentiment['negative'],
            'sentiment_neutral': sentiment['neutral'],
            'sentiment_compound': (sentiment['compound'] + 1) / 2,  # Convert -1,1 to 0,1
            'financial_keywords': min(financial_density * 10, 1.0),  # Scale up
            'readability': readability
        }
    
    def process_news_batch(self, articles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch of news articles efficiently"""
        encoded_articles = []
        text_features = []
        
        for article in articles:
            # Encode article
            encoded = self.encode_text(article)
            encoded_articles.append(encoded)
            
            # Extract features
            features = self.create_text_features(article)
            text_features.append(list(features.values()))
        
        return np.array(encoded_articles), np.array(text_features)

class FinancialNER:
    """Financial Named Entity Recognition"""
    
    def __init__(self):
        self.entity_patterns = {
            'stock_ticker': r'\b[A-Z]{1,5}\b',
            'currency': r'\$[0-9,]+(?:\.[0-9]{2})?',
            'percentage': r'[0-9]+(?:\.[0-9]+)?%',
            'date': r'\b(?:Q[1-4]|[A-Za-z]+ [0-9]{1,2}, [0-9]{4}|[0-9]{1,2}/[0-9]{1,2}/[0-9]{4})\b',
            'financial_metric': r'\b(?:EPS|P/E|ROE|ROI|EBITDA|revenue|earnings)\b'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities from text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = matches
        
        return entities

def create_enhanced_text_processor(vocab_size: int = 10000, use_pretrained: bool = False) -> EnhancedTextProcessor:
    """Factory function to create enhanced text processor"""
    return EnhancedTextProcessor(vocab_size=vocab_size, use_pretrained=use_pretrained)