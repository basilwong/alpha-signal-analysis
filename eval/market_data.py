"""
Market Data Provider

Downloads and caches historical price data from Yahoo Finance.
Computes daily returns and constructs the quantum sector basket.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance: pip install yfinance")


# Tickers in our universe
QUANTUM_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]
MARKET_TICKER = "SPY"
SECTOR_BASKET_TICKERS = ["IONQ", "RGTI", "QBTS"]  # Equal-weighted pure-play basket

CACHE_DIR = Path("data/market")


class MarketDataProvider:
    """Downloads and caches market data from Yahoo Finance."""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._price_cache = {}
        self._return_cache = {}

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.parquet"

    def download_prices(self, tickers: list = None, start: str = "2024-01-01", end: str = None):
        """
        Download and cache daily prices for all tickers.
        Uses cached data if available and recent.
        """
        if tickers is None:
            tickers = QUANTUM_TICKERS + [MARKET_TICKER]

        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        print(f"Downloading market data for {len(tickers)} tickers...")
        print(f"  Date range: {start} to {end}")

        for ticker in tickers:
            cache_path = self._cache_path(ticker)

            # Check if cache exists and is recent (within 1 day)
            if cache_path.exists():
                cached = pd.read_parquet(cache_path)
                last_date = cached.index.max()
                if last_date >= pd.Timestamp(end) - pd.Timedelta(days=3):
                    print(f"  {ticker}: Using cached data (last: {last_date.date()})")
                    self._price_cache[ticker] = cached
                    continue

            # Download from Yahoo Finance
            try:
                print(f"  {ticker}: Downloading...", end=" ")
                data = yf.download(ticker, start=start, end=end, progress=False)
                if data.empty:
                    print(f"WARNING: No data returned for {ticker}")
                    continue

                # Handle multi-level columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                # Save to cache
                data.to_parquet(cache_path)
                self._price_cache[ticker] = data
                print(f"OK ({len(data)} days)")
            except Exception as e:
                print(f"ERROR: {e}")

        print(f"  Done. {len(self._price_cache)} tickers loaded.")

    def get_prices(self, ticker: str) -> pd.DataFrame:
        """Get cached price data for a ticker."""
        if ticker not in self._price_cache:
            cache_path = self._cache_path(ticker)
            if cache_path.exists():
                self._price_cache[ticker] = pd.read_parquet(cache_path)
            else:
                raise ValueError(f"No data for {ticker}. Run download_prices() first.")
        return self._price_cache[ticker]

    def get_daily_returns(self, ticker: str) -> pd.Series:
        """Compute daily percentage returns from adjusted close."""
        if ticker not in self._return_cache:
            prices = self.get_prices(ticker)
            close_col = "Adj Close" if "Adj Close" in prices.columns else "Close"
            self._return_cache[ticker] = prices[close_col].pct_change().dropna()
        return self._return_cache[ticker]

    def get_market_returns(self) -> pd.Series:
        """Get S&P 500 (SPY) daily returns."""
        return self.get_daily_returns(MARKET_TICKER)

    def get_sector_basket_returns(self, exclude_ticker: str = None) -> pd.Series:
        """
        Compute equal-weighted quantum sector basket returns.
        Optionally exclude a ticker (to avoid including the stock being evaluated).
        """
        basket_tickers = [t for t in SECTOR_BASKET_TICKERS if t != exclude_ticker]

        returns_list = []
        for ticker in basket_tickers:
            try:
                returns_list.append(self.get_daily_returns(ticker))
            except ValueError:
                continue

        if not returns_list:
            raise ValueError("No basket tickers available")

        # Equal-weighted average
        basket_df = pd.concat(returns_list, axis=1)
        return basket_df.mean(axis=1)

    def get_all_returns_panel(self) -> pd.DataFrame:
        """Get a panel of daily returns for all quantum tickers."""
        returns_dict = {}
        for ticker in QUANTUM_TICKERS:
            try:
                returns_dict[ticker] = self.get_daily_returns(ticker)
            except ValueError:
                continue
        return pd.DataFrame(returns_dict)

    def get_all_prices_panel(self) -> pd.DataFrame:
        """Get a panel of adjusted close prices for all quantum tickers."""
        prices_dict = {}
        for ticker in QUANTUM_TICKERS:
            try:
                prices = self.get_prices(ticker)
                close_col = "Adj Close" if "Adj Close" in prices.columns else "Close"
                prices_dict[ticker] = prices[close_col]
            except ValueError:
                continue
        return pd.DataFrame(prices_dict)


def download_all_market_data(start: str = "2024-01-01"):
    """Convenience function to download all required market data."""
    provider = MarketDataProvider()
    all_tickers = QUANTUM_TICKERS + [MARKET_TICKER]
    provider.download_prices(tickers=all_tickers, start=start)
    return provider


if __name__ == "__main__":
    print("Downloading all market data for Quantum Alpha evaluation...")
    provider = download_all_market_data(start="2024-01-01")

    # Print summary
    print("\n" + "=" * 60)
    print("MARKET DATA SUMMARY")
    print("=" * 60)

    for ticker in QUANTUM_TICKERS + [MARKET_TICKER]:
        try:
            returns = provider.get_daily_returns(ticker)
            print(f"  {ticker:6s}: {len(returns)} trading days | "
                  f"{returns.index.min().date()} to {returns.index.max().date()} | "
                  f"Mean: {returns.mean()*100:.3f}%/day | Std: {returns.std()*100:.2f}%/day")
        except ValueError as e:
            print(f"  {ticker:6s}: {e}")

    # Sector basket
    basket = provider.get_sector_basket_returns()
    print(f"\n  Sector Basket (equal-weight IONQ+RGTI+QBTS):")
    print(f"  {len(basket)} days | Mean: {basket.mean()*100:.3f}%/day | Std: {basket.std()*100:.2f}%/day")
