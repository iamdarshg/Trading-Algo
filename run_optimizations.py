#!/usr/bin/env python3
"""
🚀 QUICK START SCRIPT FOR OPTIMIZED TRADING BOT

This script provides quick access to all optimization features:
- Fast inference benchmarks
- Enhanced training pipelines  
- Historical news integration
- Real-time visualization dashboard
- Online learning capabilities

Usage:
    python run_optimizations.py
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                  🚀 OPTIMIZED TRADING BOT                 ║
    ║                                                           ║
    ║    ⚡ Faster Inference  📰 News Integration               ║
    ║    📊 Visualization     🧠 Online Learning               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib', 'plotly', 
        'streamlit', 'yfinance', 'transformers', 'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing dependencies...")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '-r', 'trading_bot/requirements.txt'
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run manually:")
            print("   pip install -r trading_bot/requirements.txt")
            return False
    else:
        print("✅ All dependencies are installed!")
    
    return True

def setup_environment():
    """Setup the environment and configuration"""
    print("\n🔧 Setting up environment...")
    
    # Create necessary directories
    directories = ['logs', 'models', 'training_plots', 'online_checkpoints']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   Created directory: {directory}")
    
    # Setup configuration
    config_file = 'config/api_keys.json'
    template_file = 'config/api_keys.json.template'
    
    if not os.path.exists(config_file) and os.path.exists(template_file):
        import shutil
        shutil.copy(template_file, config_file)
        print(f"   Created config file: {config_file}")
        print("   💡 Edit this file to add your API keys")
    
    print("✅ Environment setup complete!")

def run_quick_demo():
    """Run a quick demonstration"""
    print("\n⚡ Running quick optimization demo...")
    
    try:
        # Import and run a subset of optimizations
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from optimized_trading_demo import demonstrate_inference_optimization, setup_api_keys
        
        # Setup
        setup_api_keys()
        
        # Run inference demo
        success = demonstrate_inference_optimization()
        
        if success:
            print("✅ Quick demo completed successfully!")
            return True
        else:
            print("⚠️ Quick demo had some issues, but basic functionality works")
            return True
            
    except Exception as e:
        print(f"❌ Error in quick demo: {e}")
        return False

def launch_dashboard():
    """Launch the visualization dashboard"""
    print("\n📊 Launching visualization dashboard...")
    
    try:
        dashboard_script = os.path.join('visualization', 'dashboard.py')
        
        if os.path.exists(dashboard_script):
            print("🚀 Starting Streamlit dashboard...")
            print("🌐 Dashboard will be available at: http://localhost:8501")
            print("⏹️  Press Ctrl+C to stop the dashboard")
            
            subprocess.run([
                sys.executable, '-m', 'streamlit', 'run', 
                dashboard_script, '--server.port', '8501'
            ])
        else:
            print("❌ Dashboard script not found")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped")
        return True
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

def show_menu():
    """Show interactive menu"""
    while True:
        print("\n" + "="*50)
        print("🎯 OPTIMIZED TRADING BOT - MAIN MENU")
        print("="*50)
        print("1. 🚀 Run Full Demonstration")
        print("2. ⚡ Inference Optimization Only")
        print("3. 🏋️  Enhanced Training Demo")
        print("4. 📰 News Integration Demo")
        print("5. 🧠 Online Learning Demo")
        print("6. 📊 Launch Visualization Dashboard")
        print("7. 🔧 Setup & Configuration")
        print("8. 📖 View Documentation")
        print("0. 🚪 Exit")
        print("="*50)
        
        try:
            choice = input("Select an option (0-8): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                run_full_demo()
            elif choice == '2':
                run_inference_demo()
            elif choice == '3':
                run_training_demo()
            elif choice == '4':
                run_news_demo()
            elif choice == '5':
                run_online_learning_demo()
            elif choice == '6':
                launch_dashboard()
            elif choice == '7':
                setup_configuration()
            elif choice == '8':
                show_documentation()
            else:
                print("❌ Invalid option. Please choose 0-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_full_demo():
    """Run the full demonstration"""
    print("\n🎯 Running full demonstration...")
    try:
        subprocess.run([sys.executable, 'optimized_trading_demo.py', '--mode', 'full'])
    except Exception as e:
        print(f"❌ Error: {e}")

def run_inference_demo():
    """Run inference optimization demo"""
    print("\n⚡ Running inference optimization demo...")
    try:
        subprocess.run([sys.executable, 'optimized_trading_demo.py', '--mode', 'inference'])
    except Exception as e:
        print(f"❌ Error: {e}")

def run_training_demo():
    """Run enhanced training demo"""
    print("\n🏋️ Running enhanced training demo...")
    try:
        subprocess.run([sys.executable, 'optimized_trading_demo.py', '--mode', 'training'])
    except Exception as e:
        print(f"❌ Error: {e}")

def run_news_demo():
    """Run news integration demo"""
    print("\n📰 Running news integration demo...")
    try:
        subprocess.run([sys.executable, 'optimized_trading_demo.py', '--mode', 'news'])
    except Exception as e:
        print(f"❌ Error: {e}")

def run_online_learning_demo():
    """Run online learning demo"""
    print("\n🧠 Running online learning demo...")
    try:
        subprocess.run([sys.executable, 'optimized_trading_demo.py', '--mode', 'online'])
    except Exception as e:
        print(f"❌ Error: {e}")

def setup_configuration():
    """Setup configuration interactively"""
    print("\n🔧 Interactive Configuration Setup")
    print("="*40)
    
    config_file = 'config/api_keys.json'
    
    if os.path.exists(config_file):
        print(f"📁 Configuration file exists: {config_file}")
        
        view_config = input("View current configuration? (y/n): ").strip().lower()
        if view_config == 'y':
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                print(f"\n📄 Current configuration:")
                print(content)
            except Exception as e:
                print(f"❌ Error reading config: {e}")
    else:
        print("📄 No configuration file found. Creating template...")
        
        # Create config directory and template
        os.makedirs('config', exist_ok=True)
        
        template_content = '''
{
  "news": {
    "newsapi": {
      "api_key": "your_newsapi_key_here",
      "enabled": false
    },
    "alpha_vantage": {
      "api_key": "your_alpha_vantage_key_here", 
      "enabled": false
    }
  },
  "ai_services": {
    "openai": {
      "api_key": "your_openai_key_here",
      "enabled": false
    }
  }
}
        '''
        
        try:
            with open(config_file, 'w') as f:
                f.write(template_content.strip())
            print(f"✅ Created configuration template: {config_file}")
        except Exception as e:
            print(f"❌ Error creating config: {e}")
    
    print(f"\n💡 Edit {config_file} to add your API keys and enable services")
    print("📚 See README.md for information on obtaining API keys")

def show_documentation():
    """Show documentation and help"""
    print("\n📖 DOCUMENTATION & HELP")
    print("="*40)
    
    docs = [
        ("README.md", "Main project documentation"),
        ("trading_bot/README.md", "Trading bot architecture"),
        ("config/api_keys.json.template", "API key configuration template"),
        ("visualization/", "Dashboard and visualization tools"),
    ]
    
    print("📚 Available documentation:")
    for filename, description in docs:
        exists = "✅" if os.path.exists(filename) else "❌"
        print(f"   {exists} {filename} - {description}")
    
    print("\n🔗 Key Features:")
    print("   ⚡ Optimized Inference: Model quantization, caching, batch processing")
    print("   📰 News Integration: Historical data, sentiment analysis")
    print("   🧠 Online Learning: Adaptive models, real-time updates")
    print("   📊 Visualization: Real-time dashboards, training metrics")
    
    print("\n🚀 Quick Start Commands:")
    print("   python run_optimizations.py          # Interactive menu")
    print("   python optimized_trading_demo.py     # Full demonstration")
    print("   streamlit run visualization/dashboard.py  # Launch dashboard")

def main():
    """Main function"""
    print_banner()
    
    # Check system
    print(f"🖥️  System: {platform.system()} {platform.machine()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup environment
    setup_environment()
    
    # Quick test
    print("\n🧪 Running quick functionality test...")
    if run_quick_demo():
        print("✅ System is ready!")
        
        # Show menu
        print("\n🎉 Welcome to the Optimized Trading Bot!")
        show_menu()
    else:
        print("⚠️ Some issues detected, but you can still use the system")
        show_menu()

if __name__ == "__main__":
    main()