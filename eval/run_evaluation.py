"""
Evaluation Pipeline: Compute IC, Abnormal Returns, Signal Decay

Uses the 215+ successful predictions and market data to compute:
1. Abnormal returns (market model: OLS regression against SPY)
2. Information Coefficient (Spearman rank correlation)
3. Signal decay curve (IC at multiple horizons)
4. Direction accuracy
5. IC by subset (source, ticker, event type)

Usage:
    python eval/run_evaluation.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from datetime import timedelta

# Paths
DATA_DIR = Path("data")
EVAL_DIR = DATA_DIR / "eval"
MARKET_DIR = DATA_DIR / "market"

QUANTUM_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]
MARKET_TICKER = "SPY"
SECTOR_BASKET = ["IONQ", "RGTI", "QBTS"]


def load_predictions():
    """Load successful predictions."""
    predictions = []
    pred_path = EVAL_DIR / "predictions_v2_final.jsonl"
    if not pred_path.exists():
        pred_path = EVAL_DIR / "predictions_v2.jsonl"

    with open(pred_path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p.get("status") == "success":
                    predictions.append(p)
    return predictions


def load_returns():
    """Load daily returns for all tickers."""
    returns = {}
    for ticker in QUANTUM_TICKERS + [MARKET_TICKER]:
        path = MARKET_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            returns[ticker] = df[close_col].pct_change().dropna()
    return pd.DataFrame(returns)


def estimate_market_model(stock_returns, market_returns, estimation_window=180):
    """
    Estimate alpha and beta using OLS regression.
    Returns (alpha, beta, r_squared) or None if insufficient data.
    """
    # Align dates
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ["stock", "market"]

    if len(aligned) < 60:  # Minimum 60 observations
        return None

    # Use last `estimation_window` observations
    aligned = aligned.tail(estimation_window)

    X = sm.add_constant(aligned["market"])
    y = aligned["stock"]

    try:
        model = sm.OLS(y, X).fit()
        return {
            "alpha": model.params.iloc[0],
            "beta": model.params.iloc[1],
            "r_squared": model.rsquared,
            "n_obs": len(aligned),
        }
    except:
        return None


def compute_abnormal_returns(predictions, returns_df, horizons=[1, 2, 5, 10, 20]):
    """
    Compute abnormal returns for each (prediction, ticker) pair.
    Returns a DataFrame with columns: article_idx, ticker, predicted_score, horizon, car
    """
    results = []

    for pred in predictions:
        date_str = pred.get("date", "")
        signal = pred.get("signal", {})
        signal_vector = signal.get("signal_vector", {})

        if not date_str:
            continue

        try:
            event_date = pd.Timestamp(date_str)
        except:
            continue

        for ticker in QUANTUM_TICKERS:
            if ticker not in signal_vector:
                continue
            if ticker not in returns_df.columns:
                continue

            predicted_score = signal_vector[ticker].get("score", 0)

            stock_returns = returns_df[ticker]
            market_returns = returns_df[MARKET_TICKER]

            # Get returns before event for estimation
            pre_event = stock_returns[stock_returns.index < event_date]
            pre_market = market_returns[market_returns.index < event_date]

            if len(pre_event) < 60:
                continue

            # Estimate market model using 180 days before event (with 10-day gap)
            gap_date = event_date - timedelta(days=14)  # ~10 trading days gap
            est_stock = pre_event[pre_event.index < gap_date].tail(180)
            est_market = pre_market[pre_market.index < gap_date].tail(180)

            params = estimate_market_model(est_stock, est_market)
            if params is None:
                continue

            alpha = params["alpha"]
            beta = params["beta"]

            # Compute CAR for each horizon
            post_event_stock = stock_returns[stock_returns.index > event_date]
            post_event_market = market_returns[market_returns.index > event_date]

            for horizon in horizons:
                if len(post_event_stock) < horizon:
                    continue

                # Get returns for the window
                window_stock = post_event_stock.iloc[:horizon]
                window_market = post_event_market.iloc[:horizon]

                # Align
                aligned_idx = window_stock.index.intersection(window_market.index)
                if len(aligned_idx) < 1:
                    continue

                ws = window_stock.loc[aligned_idx]
                wm = window_market.loc[aligned_idx]

                # Compute abnormal returns
                expected_returns = alpha + beta * wm
                abnormal_returns = ws - expected_returns
                car = abnormal_returns.sum()

                results.append({
                    "article_idx": pred.get("article_idx"),
                    "date": date_str,
                    "ticker": ticker,
                    "predicted_score": predicted_score,
                    "horizon": horizon,
                    "car": car,
                    "source": pred.get("source", ""),
                    "event_type": signal.get("event_type", ""),
                    "beta": beta,
                })

    return pd.DataFrame(results)


def compute_ic(predicted_scores, realized_returns):
    """Compute Spearman IC with p-value."""
    # Remove NaN
    mask = ~(np.isnan(predicted_scores) | np.isnan(realized_returns))
    pred = predicted_scores[mask]
    real = realized_returns[mask]

    if len(pred) < 10:
        return {"ic": np.nan, "p_value": np.nan, "n": len(pred)}

    ic, p_value = stats.spearmanr(pred, real)
    return {"ic": ic, "p_value": p_value, "n": len(pred)}


def compute_direction_accuracy(predicted_scores, realized_returns):
    """Compute what percentage of predictions got the direction right."""
    mask = ~(np.isnan(predicted_scores) | np.isnan(realized_returns))
    pred = predicted_scores[mask]
    real = realized_returns[mask]

    # Only count non-zero predictions
    nonzero = np.abs(pred) > 0.05
    if nonzero.sum() < 5:
        return {"accuracy": np.nan, "n": 0}

    pred_direction = np.sign(pred[nonzero])
    real_direction = np.sign(real[nonzero])
    correct = (pred_direction == real_direction).sum()
    total = nonzero.sum()

    return {"accuracy": correct / total, "correct": int(correct), "total": int(total)}


def bootstrap_ic(predicted_scores, realized_returns, n_bootstrap=1000):
    """Compute 95% bootstrap confidence interval for IC."""
    mask = ~(np.isnan(predicted_scores) | np.isnan(realized_returns))
    pred = predicted_scores[mask]
    real = realized_returns[mask]
    n = len(pred)

    if n < 10:
        return {"ci_lower": np.nan, "ci_upper": np.nan}

    ics = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        ic, _ = stats.spearmanr(pred[idx], real[idx])
        ics.append(ic)

    return {
        "ci_lower": np.percentile(ics, 2.5),
        "ci_upper": np.percentile(ics, 97.5),
    }


def run_evaluation():
    """Run the full evaluation pipeline."""
    print("=" * 60)
    print("QUANTUM ALPHA EVALUATION PIPELINE")
    print("=" * 60)

    # Load data
    print("\nLoading predictions...")
    predictions = load_predictions()
    print(f"  {len(predictions)} successful predictions loaded")

    print("Loading market data...")
    returns_df = load_returns()
    print(f"  {returns_df.shape[0]} trading days, {returns_df.shape[1]} tickers")

    # Compute abnormal returns
    print("\nComputing abnormal returns...")
    horizons = [1, 2, 5, 10, 20]
    ar_df = compute_abnormal_returns(predictions, returns_df, horizons=horizons)
    print(f"  {len(ar_df)} (event, ticker, horizon) observations computed")

    if ar_df.empty:
        print("ERROR: No abnormal returns computed. Check date alignment.")
        return

    # Overall IC at each horizon
    print("\n" + "=" * 60)
    print("INFORMATION COEFFICIENT (Overall)")
    print("=" * 60)

    decay_curve = []
    for horizon in horizons:
        subset = ar_df[ar_df["horizon"] == horizon]
        ic_result = compute_ic(
            subset["predicted_score"].values,
            subset["car"].values
        )
        bootstrap = bootstrap_ic(
            subset["predicted_score"].values,
            subset["car"].values
        )
        ic_result.update(bootstrap)
        ic_result["horizon"] = horizon
        decay_curve.append(ic_result)

        sig = "***" if ic_result["p_value"] < 0.01 else "**" if ic_result["p_value"] < 0.05 else "*" if ic_result["p_value"] < 0.10 else ""
        print(f"  Horizon +{horizon:2d} days: IC = {ic_result['ic']:+.4f} (p={ic_result['p_value']:.4f}) {sig} "
              f"[{ic_result['ci_lower']:+.4f}, {ic_result['ci_upper']:+.4f}] n={ic_result['n']}")

    # Direction accuracy
    print("\n" + "=" * 60)
    print("DIRECTION ACCURACY")
    print("=" * 60)

    for horizon in [1, 5, 10]:
        subset = ar_df[ar_df["horizon"] == horizon]
        dir_result = compute_direction_accuracy(
            subset["predicted_score"].values,
            subset["car"].values
        )
        print(f"  Horizon +{horizon:2d} days: {dir_result['accuracy']*100:.1f}% "
              f"({dir_result.get('correct', 0)}/{dir_result.get('total', 0)})")

    # IC by source type
    print("\n" + "=" * 60)
    print("IC BY SOURCE TYPE (Horizon +5 days)")
    print("=" * 60)

    h5 = ar_df[ar_df["horizon"] == 5]
    ic_by_source = {}
    for source in h5["source"].unique():
        subset = h5[h5["source"] == source]
        ic_result = compute_ic(subset["predicted_score"].values, subset["car"].values)
        ic_by_source[source] = ic_result
        print(f"  {source:15s}: IC = {ic_result['ic']:+.4f} (p={ic_result['p_value']:.4f}) n={ic_result['n']}")

    # IC by ticker
    print("\n" + "=" * 60)
    print("IC BY TICKER (Horizon +5 days)")
    print("=" * 60)

    ic_by_ticker = {}
    for ticker in QUANTUM_TICKERS:
        subset = h5[h5["ticker"] == ticker]
        if len(subset) < 10:
            continue
        ic_result = compute_ic(subset["predicted_score"].values, subset["car"].values)
        ic_by_ticker[ticker] = ic_result
        print(f"  {ticker:6s}: IC = {ic_result['ic']:+.4f} (p={ic_result['p_value']:.4f}) n={ic_result['n']}")

    # IC by event type
    print("\n" + "=" * 60)
    print("IC BY EVENT TYPE (Horizon +5 days)")
    print("=" * 60)

    ic_by_event = {}
    for event_type in h5["event_type"].unique():
        subset = h5[h5["event_type"] == event_type]
        if len(subset) < 20:
            continue
        ic_result = compute_ic(subset["predicted_score"].values, subset["car"].values)
        ic_by_event[event_type] = ic_result
        print(f"  {event_type:30s}: IC = {ic_result['ic']:+.4f} (p={ic_result['p_value']:.4f}) n={ic_result['n']}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    results = {
        "summary": {
            "n_predictions": len(predictions),
            "n_observations": len(ar_df),
            "date_range": f"{ar_df['date'].min()} to {ar_df['date'].max()}",
        },
        "decay_curve": decay_curve,
        "direction_accuracy": {
            h: compute_direction_accuracy(
                ar_df[ar_df["horizon"] == h]["predicted_score"].values,
                ar_df[ar_df["horizon"] == h]["car"].values
            ) for h in [1, 5, 10]
        },
        "ic_by_source": ic_by_source,
        "ic_by_ticker": ic_by_ticker,
        "ic_by_event_type": ic_by_event,
    }

    results_path = EVAL_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {results_path}")

    # Save abnormal returns
    ar_path = EVAL_DIR / "abnormal_returns.csv"
    ar_df.to_csv(ar_path, index=False)
    print(f"  Abnormal returns saved to: {ar_path}")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    results = run_evaluation()
