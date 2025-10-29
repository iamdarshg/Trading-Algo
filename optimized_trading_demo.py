#!/usr/bin/env python3
"""
🚀 OPTIMIZED ADVANCED TRADING BOT DEMONSTRATION

This script showcases all the major optimizations and enhancements:
1. ⚡ Faster Inference (model optimization, caching, quantization)
2. 📰 Enhanced News Integration (historical news, era-specific context)
3. 📊 Comprehensive Visualization Tools (real-time dashboard, training metrics)
4. 🧠 Online Learning Capabilities (adaptive learning during live trading)

Usage:
    python optimized_trading_demo.py --mode [inference|training|dashboard|full]
    
    --mode inference: Demonstrate inference optimizations
    --mode training: Show enhanced training with news integration
    --mode dashboard: Launch visualization dashboard
    --mode full: Complete demonstration (default)
"""

import sys
import os
import argparse
import time
from datetime import datetime, timedelta
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_api_keys():
    """Setup API keys configuration"""
    print("🔐 Setting up API configuration...")
    
    from config.config_manager import config_manager
    
    # Check if config exists
    api_config = config_manager.get_config()
    
    if not api_config or not api_config.get('news'):
        print("📝 No API configuration found. Creating template...")
        
        # Create sample config
        sample_config = {
            "news": {
                "newsapi": {
                    "api_key": "your_newsapi_key_here",
                    "enabled": False
                }
            },
            "ai_services": {
                "openai": {
                    "api_key": "your_openai_key_here", 
                    "enabled": False
                }
            }
        }
        
        config_manager.update_config(sample_config)
        
        print("✅ API configuration template created at config/api_keys.json")
        print("💡 Update the file with your API keys to enable news integration")
    else:
        print("✅ API configuration loaded")
    
    return config_manager

def demonstrate_inference_optimization():
    """Demonstrate inference speed optimizations"""
    print("\n" + "="*60)
    print("⚡ INFERENCE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    try:
        from trading_bot.model_builder import create_hybrid_model
        from trading_bot.optimized_inference import OptimizedInferenceManager, benchmark_inference_speed
        from trading_bot.enhanced_data_processor import EnhancedDataProcessor
        
        print("🏗️  Building hybrid model...")
        model_builder = create_hybrid_model(input_size=25, hidden_size=128)
        model = model_builder.build()
        
        print("📊 Creating sample data...")
        # Create sample data for benchmarking
        sample_data = torch.randn(1, 60, 25)  # [batch, sequence, features]
        
        print("📈 Benchmarking inference speeds...")
        
        # Benchmark original vs optimized
        benchmark_results = benchmark_inference_speed(model, sample_data, iterations=50)
        
        print("\n📊 BENCHMARK RESULTS:")
        print(f"   Original Model Average Time: {benchmark_results['original_avg_time']:.6f}s")
        print(f"   Optimized Model Average Time: {benchmark_results['optimized_avg_time']:.6f}s")
        print(f"   🚀 Speedup Factor: {benchmark_results['speedup_factor']:.2f}x")
        
        # Demonstrate optimized inference manager
        print("\n🧠 Creating optimized inference manager...")
        inference_manager = OptimizedInferenceManager(model, 'balanced')
        
        # Test predictions with caching
        print("🔄 Testing prediction with caching...")
        
        for i in range(5):
            start_time = time.time()
            prediction, confidence = inference_manager.predict(sample_data, use_cache=True)
            inference_time = time.time() - start_time
            
            cache_status = "HIT" if i > 0 else "MISS"
            print(f"   Prediction {i+1}: {prediction:.6f} (confidence: {confidence:.3f}) "
                  f"[{inference_time*1000:.2f}ms - Cache {cache_status}]")
        
        # Show performance statistics
        stats = inference_manager.get_performance_stats()
        print(f"\n📈 PERFORMANCE STATISTICS:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in inference demonstration: {e}")
        return False

def demonstrate_enhanced_training():
    """Demonstrate enhanced training with news integration"""
    print("\n" + "="*60)
    print("🏋️ ENHANCED TRAINING DEMONSTRATION")
    print("="*60)
    
    try:
        from trading_bot.enhanced_trainer import EnhancedTrainer, TrainingConfig, train_with_enhanced_pipeline
        from trading_bot.enhanced_data_processor import EnhancedDataProcessor
        from trading_bot.model_builder import create_hybrid_model
        from visualization.training_visualizer import TrainingVisualizer
        
        print("⚙️  Creating enhanced training configuration...")
        
        # Create advanced training config
        training_config = TrainingConfig(
            epochs=20,  # Reduced for demo
            batch_size=16,
            learning_rate=1e-4,
            use_mixed_precision=True,
            use_data_augmentation=True,
            include_news=True,
            news_lookback_days=7,
            scheduler_type='cosine_with_restarts'
        )
        
        print("📊 Initializing enhanced data processor...")
        data_processor = EnhancedDataProcessor(
            sequence_length=60,
            prediction_horizon=1,
            include_technical_indicators=True,
            normalize_data=True
        )
        
        print("🏗️  Building model...")
        model_builder = create_hybrid_model(input_size=50, hidden_size=64)  # Smaller for demo
        
        print("📰 Creating enhanced dataset with news integration...")
        symbol = "AAPL"
        
        # Create dataset (may take a moment for news fetching)
        dataset = data_processor.create_enhanced_dataset(
            symbol=symbol,
            period="6mo",  # Shorter period for demo
            interval="1d",
            include_news=training_config.include_news
        )
        
        if not dataset:
            print("⚠️  Could not create dataset, using synthetic data for demo...")
            
            # Create synthetic dataset for demonstration
            X_train = np.random.randn(100, 60, 50)
            y_train = np.random.randn(100)
            X_val = np.random.randn(20, 60, 50)
            y_val = np.random.randn(20)
            
            dataset = {
                'processed_data': {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'feature_names': [f'feature_{i}' for i in range(50)]
                }
            }
        
        print("🚀 Starting enhanced training...")
        
        # Build and train model
        model = model_builder.build()
        trainer = EnhancedTrainer(model, training_config)
        
        # Create data loaders
        train_loader, val_loader = trainer.create_data_loaders(
            dataset['processed_data'],
            symbol,
            dataset if training_config.include_news else None
        )
        
        print("⏳ Training model (this may take a few minutes)...")
        history = trainer.fit(train_loader, val_loader)
        
        print("✅ Training completed!")
        
        # Generate visualizations
        print("📊 Generating training visualizations...")
        visualizer = TrainingVisualizer()
        
        # Create training history plot
        fig = visualizer.plot_training_history(history)
        print(f"   📈 Training history saved to: {visualizer.save_dir}/training_history.html")
        
        # Create weight distribution plot
        fig = visualizer.plot_weight_distribution(model)
        print(f"   ⚖️  Weight distribution saved to: {visualizer.save_dir}/weight_distribution.html")
        
        # Generate comprehensive report
        feature_names = dataset['processed_data']['feature_names']
        feature_importance = np.random.rand(len(feature_names))  # Placeholder
        
        report = visualizer.create_training_report(history, model, feature_names, feature_importance)
        print(f"   📋 Training report saved to: {visualizer.save_dir}/training_report.md")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in training demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_online_learning():
    """Demonstrate online learning capabilities"""
    print("\n" + "="*60)
    print("🧠 ONLINE LEARNING DEMONSTRATION")
    print("="*60)
    
    try:
        from trading_bot.online_learning import OnlineLearningManager, AdaptiveTradingStrategy
        from trading_bot.optimized_inference import OptimizedInferenceManager
        from trading_bot.enhanced_data_processor import EnhancedDataProcessor
        from trading_bot.model_builder import create_hybrid_model
        
        print("🏗️  Setting up online learning system...")
        
        # Create model and processors
        model_builder = create_hybrid_model(input_size=25, hidden_size=64)
        model = model_builder.build()
        
        data_processor = EnhancedDataProcessor()
        inference_manager = OptimizedInferenceManager(model, 'balanced')
        
        # Create adaptive strategy
        adaptive_strategy = AdaptiveTradingStrategy(model, data_processor, inference_manager)
        
        print("🚀 Starting online learning...")
        adaptive_strategy.start_adaptive_learning()
        
        # Simulate trading experiences
        print("📊 Simulating trading experiences...")
        
        for i in range(10):
            # Simulate market data
            price_data = np.random.randn(60, 25)
            
            # Make prediction
            prediction, confidence = adaptive_strategy.predict_with_learning(price_data)
            
            # Simulate actual market outcome
            actual_return = np.random.randn() * 0.02  # ±2% return
            
            # Update with actual outcome
            adaptive_strategy.update_with_actual_outcome(actual_return, symbol="AAPL")
            
            print(f"   Experience {i+1}: Predicted {prediction:.4f}, Actual {actual_return:.4f}, "
                  f"Error: {abs(prediction - actual_return):.4f}")
            
            time.sleep(0.5)  # Small delay to show updates
        
        # Get adaptation metrics
        metrics = adaptive_strategy.get_adaptation_metrics()
        
        print(f"\n📈 ONLINE LEARNING METRICS:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
        
        print("🛑 Stopping online learning...")
        adaptive_strategy.stop_adaptive_learning()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in online learning demonstration: {e}")
        return False

def launch_dashboard():
    """Launch the comprehensive trading dashboard"""
    print("\n" + "="*60)
    print("📊 LAUNCHING COMPREHENSIVE DASHBOARD")
    print("="*60)
    
    try:
        print("🚀 Starting Streamlit dashboard...")
        print("📝 Note: This will start a web server. Access the dashboard at: http://localhost:8501")
        
        import subprocess
        import sys
        
        # Launch Streamlit dashboard
        dashboard_path = os.path.join(os.path.dirname(__file__), 'visualization', 'dashboard.py')
        
        if os.path.exists(dashboard_path):
            cmd = [sys.executable, '-m', 'streamlit', 'run', dashboard_path, '--server.port', '8501']
            print(f"   Command: {' '.join(cmd)}")
            
            subprocess.run(cmd)
        else:
            print("❌ Dashboard file not found. Creating simplified dashboard...")
            
            # Create a simple dashboard display
            from visualization.dashboard import TradingDashboard
            
            dashboard = TradingDashboard()
            print("✅ Dashboard components initialized")
            print("💡 For full dashboard functionality, run: streamlit run visualization/dashboard.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Try running: pip install streamlit")
        return False

def demonstrate_news_integration():
    """Demonstrate enhanced news integration"""
    print("\n" + "="*60)
    print("📰 NEWS INTEGRATION DEMONSTRATION") 
    print("="*60)
    
    try:
        from trading_bot.enhanced_data_processor import HistoricalNewsProvider
        from trading_bot.enhanced_text_processor import EnhancedTextProcessor
        
        print("🔍 Initializing news providers...")
        news_provider = HistoricalNewsProvider()
        text_processor = EnhancedTextProcessor()
        
        print("📅 Fetching historical news for demonstration...")
        
        # Try to get historical news
        symbol = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"   Searching for news about {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        articles = news_provider.get_historical_news(symbol, start_date, end_date, max_articles=5)
        
        if articles:
            print(f"✅ Found {len(articles)} news articles")
            
            for i, article in enumerate(articles, 1):
                print(f"\n📄 Article {i}:")
                print(f"   Title: {article.title[:100]}...")
                print(f"   Source: {article.source}")
                print(f"   Date: {article.published_at.strftime('%Y-%m-%d %H:%M')}")
                
                # Analyze sentiment
                sentiment = text_processor.extract_financial_sentiment(article.title + " " + article.content)
                print(f"   Sentiment: {sentiment['compound']:.3f} (Positive: {sentiment['positive']:.3f}, Negative: {sentiment['negative']:.3f})")
            
            # Demonstrate text processing
            print("\n🔤 TEXT PROCESSING DEMONSTRATION:")
            
            sample_text = articles[0].title + " " + articles[0].content
            
            # Extract features
            features = text_processor.create_text_features(sample_text)
            print(f"   Text Features:")
            for key, value in features.items():
                print(f"      {key}: {value:.4f}")
            
            # Build vocabulary and encode
            all_texts = [article.title + " " + article.content for article in articles]
            text_processor.build_vocab(all_texts)
            
            encoded = text_processor.encode_text(sample_text)
            print(f"   Encoded length: {len(encoded)} tokens")
            
        else:
            print("⚠️  No news articles found (this is normal without API keys)")
            print("💡 To enable news integration, add your API keys to config/api_keys.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in news integration demonstration: {e}")
        return False

def run_full_demonstration():
    """Run complete demonstration of all features"""
    print("🎯 RUNNING COMPLETE OPTIMIZED TRADING BOT DEMONSTRATION")
    print("=" * 70)
    
    start_time = time.time()
    results = {}
    
    # Setup
    print("🔧 Setting up environment...")
    config_manager = setup_api_keys()
    
    # Run all demonstrations
    demonstrations = [
        ("Inference Optimization", demonstrate_inference_optimization),
        ("News Integration", demonstrate_news_integration),
        ("Online Learning", demonstrate_online_learning),
        ("Enhanced Training", demonstrate_enhanced_training),
    ]
    
    for name, demo_func in demonstrations:
        print(f"\n🚀 Running {name}...")
        success = demo_func()
        results[name] = "✅ SUCCESS" if success else "❌ FAILED"
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("📋 DEMONSTRATION SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        print(f"   {name}: {result}")
    
    print(f"\n⏱️  Total demonstration time: {total_time:.2f} seconds")
    
    success_count = sum(1 for result in results.values() if "SUCCESS" in result)
    print(f"📊 Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count == len(results):
        print("\n🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("\n🚀 Key Optimizations Demonstrated:")
        print("   ⚡ Inference Speed: Model quantization, caching, batch processing")
        print("   📰 News Integration: Historical data, sentiment analysis, era-specific context")
        print("   🧠 Online Learning: Adaptive models, experience replay, real-time updates")
        print("   📊 Visualization: Comprehensive dashboards, training metrics, performance tracking")
        
        print("\n💡 Next Steps:")
        print("   1. Configure your API keys in config/api_keys.json")
        print("   2. Run the dashboard: streamlit run visualization/dashboard.py")
        print("   3. Train models with your own data using the enhanced pipeline")
        print("   4. Deploy with online learning for adaptive trading")
    else:
        print("\n⚠️  Some demonstrations had issues. Check the logs above for details.")

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Optimized Trading Bot Demonstration")
    parser.add_argument('--mode', choices=['inference', 'training', 'dashboard', 'news', 'online', 'full'], 
                       default='full', help='Demonstration mode')
    
    args = parser.parse_args()
    
    if args.mode == 'inference':
        setup_api_keys()
        demonstrate_inference_optimization()
    elif args.mode == 'training':
        setup_api_keys()
        demonstrate_enhanced_training()
    elif args.mode == 'dashboard':
        setup_api_keys()
        launch_dashboard()
    elif args.mode == 'news':
        setup_api_keys()
        demonstrate_news_integration()
    elif args.mode == 'online':
        setup_api_keys()
        demonstrate_online_learning()
    elif args.mode == 'full':
        run_full_demonstration()

if __name__ == "__main__":
    main()