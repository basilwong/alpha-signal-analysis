"""
Market Context Provider (Fix 3, Fix 12)

Computes market context for teacher pipeline and inference:
- 5-day and 30-day returns for active tickers
- 52-week high/low position
- Liquidity tier
- Market regime (bull/bear/neutral + volatility)

Usage:
    from src.market_context import get_market_context
    context = get_market_context("2025-03-15")
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import config - handle both running from project root and from scripts/
try:
    from src.config import LIQUIDITY_TIERS, ACTIVE_TICKERS
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.config import LIQUIDITY_TIERS, ACTIVE_TICKERS

MARKET_DIR = Path(__file__).parent.parent / "data" / "market"


def get_market_context(date: str, tickers: list = None, market_dir: Path = None) -> str:
    """
    Compute market context block for a given date.
    
    Returns formatted string for prompt injection, or empty string if data unavailable.
    
    Args:
        date: Article date in YYYY-MM-DD format
        tickers: List of tickers to include (defaults to ACTIVE_TICKERS)
        market_dir: Path to market data directory
    """
    if tickers is None:
        tickers = ACTIVE_TICKERS
    if market_dir is None:
        market_dir = MARKET_DIR
    
    if not market_dir.exists():
        return ""
    
    target_date = pd.Timestamp(date)
    rows = []
    
    for ticker in tickers:
        path = market_dir / f"{ticker}.parquet"
        if not path.exists():
            continue
        
        try:
            df = pd.read_parquet(path)
            close = df["Close"]
        except Exception:
            continue
        
        # Find the most recent trading day on or before target_date
        available = close.loc[:target_date]
        if len(available) < 30:
            continue
        
        current_price = available.iloc[-1]
        
        # 5-day return
        if len(available) >= 6:
            ret_5d = (available.iloc[-1] / available.iloc[-6] - 1) * 100
        else:
            ret_5d = None
        
        # 30-day return
        if len(available) >= 31:
            ret_30d = (available.iloc[-1] / available.iloc[-31] - 1) * 100
        else:
            ret_30d = None
        
        # 52-week position
        lookback = min(252, len(available))
        window = available.iloc[-lookback:]
        high_52w = window.max()
        low_52w = window.min()
        
        range_52w = high_52w - low_52w
        if range_52w > 0:
            position = (current_price - low_52w) / range_52w
            if position > 0.8:
                pos_label = "Near high"
            elif position < 0.2:
                pos_label = "Near low"
            else:
                pos_label = "Mid-range"
        else:
            pos_label = "N/A"
        
        # Liquidity tier
        liq_info = LIQUIDITY_TIERS.get(ticker, {})
        liq_tier = liq_info.get("tier", "unknown")
        
        rows.append({
            "ticker": ticker,
            "ret_5d": f"{ret_5d:+.1f}%" if ret_5d is not None else "N/A",
            "ret_30d": f"{ret_30d:+.1f}%" if ret_30d is not None else "N/A",
            "position": pos_label,
            "liquidity": liq_tier.replace("_", " ").title(),
        })
    
    if not rows:
        return ""
    
    # Format as markdown table
    lines = [f"**Market Context (as of {date}):**"]
    lines.append("| Ticker | 5d Ret | 30d Ret | 52w Position | Liquidity |")
    lines.append("|--------|--------|---------|--------------|-----------|")
    for r in rows:
        lines.append(f"| {r['ticker']} | {r['ret_5d']} | {r['ret_30d']} | {r['position']} | {r['liquidity']} |")
    
    # Add regime (Fix 12)
    regime = get_market_regime(date, market_dir)
    if regime:
        lines.append(f"\n**Market regime:** {regime}")
    
    return "\n".join(lines)


def get_market_regime(date: str, market_dir: Path = None) -> str:
    """
    Fix 12: Compute market regime for a given date.
    
    Uses SPY 30-day return and quantum basket 30-day realized volatility.
    Returns regime string like "Bull / Low Volatility" or "Bear / High Volatility".
    """
    if market_dir is None:
        market_dir = MARKET_DIR
    
    spy_path = market_dir / "SPY.parquet"
    if not spy_path.exists():
        return ""
    
    try:
        spy = pd.read_parquet(spy_path)["Close"]
    except Exception:
        return ""
    
    target_date = pd.Timestamp(date)
    available = spy.loc[:target_date]
    
    if len(available) < 31:
        return ""
    
    spy_30d = (available.iloc[-1] / available.iloc[-31] - 1)
    
    # Quantum basket volatility (IONQ, RGTI, QBTS)
    basket_tickers = ["IONQ", "RGTI", "QBTS"]
    basket_rets = []
    for t in basket_tickers:
        path = market_dir / f"{t}.parquet"
        if path.exists():
            try:
                close = pd.read_parquet(path)["Close"]
                rets = close.pct_change().loc[:target_date].tail(30).dropna()
                if len(rets) > 0:
                    basket_rets.append(rets)
            except Exception:
                continue
    
    if basket_rets:
        # Average annualized volatility across basket
        vols = [r.std() * (252**0.5) for r in basket_rets]
        basket_vol = np.mean(vols)
    else:
        basket_vol = 0.5  # Default to mid-range if no data
    
    # Classify regime
    if spy_30d > 0.05:
        regime = "Bull"
    elif spy_30d < -0.05:
        regime = "Bear"
    else:
        regime = "Neutral"
    
    if basket_vol > 0.80:
        regime += " / High Volatility"
    elif basket_vol < 0.40:
        regime += " / Low Volatility"
    else:
        regime += " / Normal Volatility"
    
    return regime


def get_5d_forward_returns(date: str, tickers: list = None, market_dir: Path = None) -> dict:
    """
    Compute 5-day forward returns from a given date.
    Used by Fix 4 (teacher accuracy metadata).
    
    Returns dict of {ticker: 5d_return} or None if data unavailable.
    """
    if tickers is None:
        tickers = ACTIVE_TICKERS
    if market_dir is None:
        market_dir = MARKET_DIR
    
    target_date = pd.Timestamp(date)
    returns = {}
    
    for ticker in tickers:
        path = market_dir / f"{ticker}.parquet"
        if not path.exists():
            continue
        
        try:
            df = pd.read_parquet(path)
            close = df["Close"]
        except Exception:
            continue
        
        # Find the trading day on or after target_date
        future = close.loc[target_date:]
        if len(future) < 6:  # Need at least 6 days (today + 5 forward)
            continue
        
        # 5-day forward return
        ret_5d = (future.iloc[5] / future.iloc[0] - 1)
        returns[ticker] = ret_5d
    
    return returns if returns else None
