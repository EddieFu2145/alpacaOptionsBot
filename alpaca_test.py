from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
trading_client = TradingClient(api_key, secret_key, paper=paper)
