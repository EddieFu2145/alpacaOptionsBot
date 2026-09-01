"""Historical option chain and option bar data."""
import re
from datetime import date, datetime
from typing import Optional

import pandas as pd

from alpaca.data.requests import OptionBarsRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import AssetStatus
from alpaca.trading.requests import GetOptionContractsRequest

from .cache import cached_fetch
from .clients import option_client, trading_client

_OCC_SYMBOL_RE = re.compile(r"^(?P<underlying>[A-Z]+)(?P<expiration>\d{6})(?P<option_type>[CP])(?P<strike>\d{8})$")


def parse_occ_symbol(symbol: str) -> dict:
    """Break an OCC option symbol (e.g. 'AAPL240119C00190000') into its parts."""
    match = _OCC_SYMBOL_RE.match(symbol)
    if not match:
        raise ValueError(f"'{symbol}' is not a valid OCC option symbol")
    return {
        "underlying": match["underlying"],
        "expiration": datetime.strptime(match["expiration"], "%y%m%d").date(),
        "option_type": "call" if match["option_type"] == "C" else "put",
        "strike": int(match["strike"]) / 1000,
    }


def get_option_chain(
    underlying_symbol: str,
    expiration_date: Optional[date] = None,
    expiration_date_gte: Optional[date] = None,
    expiration_date_lte: Optional[date] = None,
    strike_price_gte: Optional[float] = None,
    strike_price_lte: Optional[float] = None,
) -> pd.DataFrame:
    """Current snapshot of an underlying's option chain, one row per contract."""
    cache_key = (
        f"option_chain_{underlying_symbol}_{expiration_date}_{expiration_date_gte}"
        f"_{expiration_date_lte}_{strike_price_gte}_{strike_price_lte}_{date.today():%Y%m%d}"
    )

    def fetch() -> pd.DataFrame:
        request = OptionChainRequest(
            underlying_symbol=underlying_symbol,
            expiration_date=expiration_date,
            expiration_date_gte=expiration_date_gte,
            expiration_date_lte=expiration_date_lte,
            strike_price_gte=strike_price_gte,
            strike_price_lte=strike_price_lte,
        )
        chain = option_client().get_option_chain(request)

        rows = []
        for symbol, snapshot in chain.items():
            row = {"symbol": symbol, **parse_occ_symbol(symbol)}
            if snapshot.latest_trade:
                row["last_trade_price"] = snapshot.latest_trade.price
                row["last_trade_size"] = snapshot.latest_trade.size
            if snapshot.latest_quote:
                row["bid_price"] = snapshot.latest_quote.bid_price
                row["bid_size"] = snapshot.latest_quote.bid_size
                row["ask_price"] = snapshot.latest_quote.ask_price
                row["ask_size"] = snapshot.latest_quote.ask_size
            row["implied_volatility"] = snapshot.implied_volatility
            if snapshot.greeks:
                row["delta"] = snapshot.greeks.delta
                row["gamma"] = snapshot.greeks.gamma
                row["theta"] = snapshot.greeks.theta
                row["vega"] = snapshot.greeks.vega
                row["rho"] = snapshot.greeks.rho
            rows.append(row)

        return pd.DataFrame(rows)

    return cached_fetch(cache_key, fetch)


def get_expired_contracts(
    underlying_symbol: str,
    expiration_date_gte: date,
    expiration_date_lte: date,
    strike_price_gte: Optional[float] = None,
    strike_price_lte: Optional[float] = None,
) -> pd.DataFrame:
    """Contracts that existed and have since expired/delisted, for reconstructing
    what was actually tradable on a past date. Alpaca's contract registry only
    goes back to when it launched options trading (~January 2024) - anything
    before that returns empty.
    """
    cache_key = (
        f"expired_contracts_{underlying_symbol}_{expiration_date_gte}_{expiration_date_lte}"
        f"_{strike_price_gte}_{strike_price_lte}"
    )

    def fetch() -> pd.DataFrame:
        rows = []
        page_token = None
        while True:
            request = GetOptionContractsRequest(
                underlying_symbols=[underlying_symbol],
                status=AssetStatus.INACTIVE,
                expiration_date_gte=expiration_date_gte,
                expiration_date_lte=expiration_date_lte,
                strike_price_gte=strike_price_gte,
                strike_price_lte=strike_price_lte,
                limit=10000,
                page_token=page_token,
            )
            response = trading_client().get_option_contracts(request)
            for contract in response.option_contracts:
                rows.append(
                    {
                        "symbol": contract.symbol,
                        "underlying": underlying_symbol,
                        "expiration": contract.expiration_date,
                        "option_type": contract.type.value,
                        "strike": float(contract.strike_price),
                    }
                )
            page_token = response.next_page_token
            if not page_token:
                break

        return pd.DataFrame(rows, columns=["symbol", "underlying", "expiration", "option_type", "strike"])

    return cached_fetch(cache_key, fetch)


def has_liquid_weekly_options(symbol: str, expiration: date) -> bool:
    """Cheap live check: does this contract's chain actually show trading
    activity, or is it a listed-but-dead expiration."""
    chain = get_option_chain(symbol, expiration_date=expiration)
    return not chain.empty and chain["last_trade_price"].notna().sum() >= 4


def get_option_bars(
    symbols: list[str] | str,
    start: datetime,
    end: datetime,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """Historical OHLCV bars for one or more specific option contract symbols."""
    key_symbols = symbols if isinstance(symbols, str) else "-".join(sorted(symbols))
    cache_key = f"option_bars_{key_symbols}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}"

    def fetch() -> pd.DataFrame:
        request = OptionBarsRequest(
            symbol_or_symbols=symbols,
            start=start,
            end=end,
            timeframe=timeframe,
        )
        return option_client().get_option_bars(request).df

    return cached_fetch(cache_key, fetch)
