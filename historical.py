from alpaca.data import StockHistoricalDataClient, StockTradesRequest
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

data_client = StockHistoricalDataClient(api_key, secret_key)

request_params = StockTradesRequest(
    symbol_or_symbols="AAPL",
    start=datetime(2024, 1, 30, 14, 30),
    end=datetime(2024, 1, 30, 14, 45)
)

trades = data_client.get_stock_trades(request_params)
first_trade = trades["AAPL"][0]
print(first_trade)
