import os
import importlib.util
proj_root = r"e:\Trading Algo"
os.chdir(proj_root)

# Load the data_processor module directly from file to avoid package import issues
module_path = os.path.join(proj_root, 'trading_bot', 'data_processor.py')
spec = importlib.util.spec_from_file_location('data_processor', module_path)
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)

print('Calling get_market_news("AAPL", days_back=3)')
news = dp.get_market_news('AAPL', days_back=3)
print('Got', len(news), 'headlines')
for i, h in enumerate(news[:10], 1):
    print(i, h)
