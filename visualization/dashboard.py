import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import threading
import time
from queue import Queue

# Absolute imports
from trading_bot.portfolio_manager import PortfolioManager
from trading_bot.live_broker import LiveDataBroker
from trading_bot.ml_strategy import AdvancedMLStrategy
from config.config_manager import config_manager

class TradingDashboard:
    """Real-time trading dashboard with comprehensive visualizations"""
    
    def __init__(self):
        self.portfolio_manager = None
        self.update_queue = Queue()
        self.is_running = False
        self.data_cache = {
            'portfolio_history': [],
            'trade_history': [],
            'performance_metrics': {},
            'model_metrics': {}
        }
        
    def initialize_portfolio(self, symbols: List[str], initial_balance: float = 100000):
        """Initialize portfolio manager for dashboard"""
        self.portfolio_manager = PortfolioManager(tickers=symbols, vram_gb=8)
        self.portfolio_manager.spawn_bots(initial_balance=initial_balance)
        
    def create_portfolio_overview(self) -> go.Figure:
        """Create portfolio overview chart"""
        if not self.portfolio_manager:
            return go.Figure()
        
        # Get portfolio metrics
        metrics = self.portfolio_manager.compute_portfolio_metrics()
        
        # Create portfolio composition pie chart
        symbols = list(self.portfolio_manager.bots.keys())
        values = []
        colors = px.colors.qualitative.Set3
        
        for symbol in symbols:
            bot_state = self.portfolio_manager.bots[symbol]
            portfolio_value = bot_state.broker.get_portfolio_value()
            values.append(portfolio_value)
        
        fig = go.Figure(data=[
            go.Pie(
                labels=symbols,
                values=values,
                hole=0.3,
                marker_colors=colors[:len(symbols)]
            )
        ])
        
        fig.update_layout(
            title="Portfolio Composition",
            font=dict(size=14),
            showlegend=True
        )
        
        return fig
    
    def create_performance_chart(self) -> go.Figure:
        """Create performance tracking chart"""
        if not self.data_cache['portfolio_history']:
            return go.Figure()
        
        df = pd.DataFrame(self.data_cache['portfolio_history'])
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Portfolio Value', 'Daily Returns', 'Drawdown'),
            vertical_spacing=0.08
        )
        
        # Portfolio value over time
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['total_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Daily returns
        if 'daily_return' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['daily_return'],
                    name='Daily Returns',
                    marker_color=['red' if x < 0 else 'green' for x in df['daily_return']]
                ),
                row=2, col=1
            )
        
        # Drawdown
        if 'drawdown' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['drawdown'],
                    mode='lines',
                    name='Drawdown',
                    fill='tonexty',
                    line=dict(color='red', width=1)
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title="Portfolio Performance",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_signals_chart(self, symbol: str) -> go.Figure:
        """Create trading signals visualization"""
        if not self.portfolio_manager or symbol not in self.portfolio_manager.bots:
            return go.Figure()
        
        bot_state = self.portfolio_manager.bots[symbol]
        
        # Get recent price data
        price_data = bot_state.broker.get_market_data(symbol, period='1mo', interval='1d')
        
        if price_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name=f'{symbol} Price'
            )
        )
        
        # Add moving averages if available
        if hasattr(bot_state.strategy, 'data_processor'):
            processor = bot_state.strategy.data_processor
            processed_data = processor.calculate_technical_indicators(price_data)
            
            # Add SMA lines
            for sma_period in [20, 50]:
                sma_col = f'SMA_{sma_period}'
                if sma_col in processed_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=processed_data.index,
                            y=processed_data[sma_col],
                            mode='lines',
                            name=f'SMA {sma_period}',
                            line=dict(width=1)
                        )
                    )
        
        # Add trade signals from history
        trade_history = [t for t in self.data_cache['trade_history'] if t.get('symbol') == symbol]
        
        for trade in trade_history[-20:]:  # Last 20 trades
            color = 'green' if trade['signal_type'] == 'buy' else 'red'
            fig.add_annotation(
                x=trade['timestamp'],
                y=trade['price'],
                text=trade['signal_type'].upper(),
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                bgcolor=color,
                font=dict(color='white')
            )
        
        fig.update_layout(
            title=f"{symbol} Price and Signals",
            xaxis_title="Date",
            yaxis_title="Price",
            height=400
        )
        
        return fig
    
    def create_model_performance_chart(self) -> go.Figure:
        """Create model performance metrics chart"""
        if not self.data_cache['model_metrics']:
            return go.Figure()
        
        metrics_df = pd.DataFrame(self.data_cache['model_metrics'])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Accuracy', 'Inference Speed', 'Confidence Distribution', 'Feature Importance'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Prediction accuracy over time
        if 'accuracy' in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Inference speed
        if 'inference_time' in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['inference_time'],
                    mode='lines+markers',
                    name='Inference Time (ms)',
                    line=dict(color='orange')
                ),
                row=1, col=2
            )
        
        # Confidence distribution
        if 'confidence' in metrics_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=metrics_df['confidence'],
                    nbinsx=20,
                    name='Confidence Distribution'
                ),
                row=2, col=1
            )
        
        # Feature importance (placeholder)
        feature_names = ['Price', 'Volume', 'RSI', 'MACD', 'News Sentiment']
        importance_values = np.random.rand(5)  # Replace with actual feature importance
        
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=importance_values,
                name='Feature Importance'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Model Performance Metrics",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_risk_metrics_chart(self) -> go.Figure:
        """Create risk management metrics visualization"""
        if not self.portfolio_manager:
            return go.Figure()
        
        metrics = self.portfolio_manager.compute_portfolio_metrics()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Value at Risk (VaR)', 'Beta vs Market', 'Correlation Matrix', 'Sharpe Ratios'),
            specs=[[{"type": "indicator"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # VaR gauge
        var_value = metrics.get('portfolio', {}).get('var_95', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=abs(var_value) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "VaR 95% (%)"}
            ),
            row=1, col=1
        )
        
        # Beta scatter (placeholder)
        symbols = list(self.portfolio_manager.bots.keys())
        betas = np.random.rand(len(symbols))
        returns = np.random.randn(len(symbols)) * 0.02
        
        fig.add_trace(
            go.Scatter(
                x=betas,
                y=returns,
                mode='markers+text',
                text=symbols,
                textposition='top center',
                name='Symbol Beta vs Return'
            ),
            row=1, col=2
        )
        
        # Correlation matrix
        n_symbols = len(symbols)
        correlation_matrix = np.random.rand(n_symbols, n_symbols)
        np.fill_diagonal(correlation_matrix, 1)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix,
                x=symbols,
                y=symbols,
                colorscale='RdBu',
                zmid=0
            ),
            row=2, col=1
        )
        
        # Sharpe ratios
        sharpe_ratios = np.random.rand(len(symbols))
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=sharpe_ratios,
                name='Sharpe Ratio'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Risk Management Dashboard",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_news_sentiment_chart(self) -> go.Figure:
        """Create news sentiment analysis chart"""
        # Placeholder for news sentiment data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        sentiment_scores = np.random.randn(len(dates)) * 0.5
        news_count = np.random.randint(5, 20, len(dates))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('News Sentiment Over Time', 'News Volume'),
            vertical_spacing=0.1
        )
        
        # Sentiment over time
        colors = ['red' if x < 0 else 'green' for x in sentiment_scores]
        fig.add_trace(
            go.Bar(
                x=dates,
                y=sentiment_scores,
                name='Sentiment Score',
                marker_color=colors
            ),
            row=1, col=1
        )
        
        # News volume
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=news_count,
                mode='lines+markers',
                name='News Articles Count',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="News Sentiment Analysis",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def update_data(self):
        """Update dashboard data from portfolio manager"""
        if not self.portfolio_manager:
            return
        
        try:
            # Run portfolio iteration
            self.portfolio_manager.run_iteration()
            
            # Update portfolio history
            current_time = datetime.now()
            portfolio_value = sum(
                bot_state.broker.get_portfolio_value()
                for bot_state in self.portfolio_manager.bots.values()
            )
            
            self.data_cache['portfolio_history'].append({
                'timestamp': current_time,
                'total_value': portfolio_value,
                'daily_return': 0,  # Calculate actual daily return
                'drawdown': 0  # Calculate actual drawdown
            })
            
            # Keep only last 100 entries
            if len(self.data_cache['portfolio_history']) > 100:
                self.data_cache['portfolio_history'] = self.data_cache['portfolio_history'][-100:]
            
            # Update model metrics (placeholder)
            self.data_cache['model_metrics'] = {
                'timestamp': current_time,
                'accuracy': np.random.rand(),
                'inference_time': np.random.rand() * 100,
                'confidence': np.random.rand()
            }
            
        except Exception as e:
            st.error(f"Error updating data: {e}")

def create_streamlit_dashboard():
    """Create Streamlit dashboard interface"""
    st.set_page_config(
        page_title="Advanced Trading Bot Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🚀 Advanced Trading Bot Dashboard")
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = TradingDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Sidebar controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        # Portfolio setup
        if st.button("Initialize Portfolio"):
            symbols = st.text_input("Symbols (comma-separated)", "AAPL,MSFT,GOOGL").split(',')
            initial_balance = st.number_input("Initial Balance", value=100000.0)
            
            with st.spinner("Initializing portfolio..."):
                dashboard.initialize_portfolio([s.strip() for s in symbols], initial_balance)
            st.success("Portfolio initialized!")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        
        if st.button("Manual Refresh"):
            dashboard.update_data()
            st.rerun()
    
    # Main dashboard content
    if dashboard.portfolio_manager:
        # Portfolio Overview Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_value = sum(
                bot_state.broker.get_portfolio_value()
                for bot_state in dashboard.portfolio_manager.bots.values()
            )
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")
        
        with col2:
            st.metric("Daily P&L", "+$1,234.56", "+2.3%")  # Placeholder
        
        with col3:
            st.metric("Active Positions", len(dashboard.portfolio_manager.bots))
        
        with col4:
            st.metric("Win Rate", "67.3%")  # Placeholder
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(dashboard.create_portfolio_overview(), use_container_width=True)
        
        with col2:
            st.plotly_chart(dashboard.create_performance_chart(), use_container_width=True)
        
        # Charts Row 2
        st.plotly_chart(dashboard.create_model_performance_chart(), use_container_width=True)
        
        # Charts Row 3
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(dashboard.create_risk_metrics_chart(), use_container_width=True)
        
        with col2:
            st.plotly_chart(dashboard.create_news_sentiment_chart(), use_container_width=True)
        
        # Individual Symbol Analysis
        st.header("Individual Symbol Analysis")
        
        selected_symbol = st.selectbox(
            "Select Symbol",
            list(dashboard.portfolio_manager.bots.keys())
        )
        
        if selected_symbol:
            st.plotly_chart(
                dashboard.create_signals_chart(selected_symbol),
                use_container_width=True
            )
    
    else:
        st.info("Please initialize a portfolio to see the dashboard.")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        dashboard.update_data()
        st.rerun()

if __name__ == "__main__":
    create_streamlit_dashboard()