"""
Is the model just a hype machine? Analyze:
1. Distribution of positive vs negative predictions
2. IC for bullish predictions only vs bearish predictions only
3. Does the model ever predict negative? When?
4. Average CAR when model says bullish vs bearish
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from datetime import timedelta

DATA_DIR = Path("data")
EVAL_DIR = DATA_DIR / "eval"
MARKET_DIR = DATA_DIR / "market"

QUANTUM_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]
MARKET_TICKER = "SPY"


def load_returns():
    returns = {}
    for ticker in QUANTUM_TICKERS + [MARKET_TICKER]:
        path = MARKET_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            returns[ticker] = df[close_col].pct_change().dropna()
    return pd.DataFrame(returns)


def load_predictions(path):
    preds = []
    with open(path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p.get("status") == "success" or p.get("success") == True:
                    preds.append(p)
    return preds


def compute_observations(predictions, returns_df, horizon=10):
    results = []
    for pred in predictions:
        date_str = pred.get("date", "")
        signal = pred.get("signal", {})
        signal_vector = signal.get("signal_vector", {})
        source = pred.get("source", "unknown")
        title = pred.get("title", "")
        if not date_str or not signal_vector:
            continue
        try:
            event_date = pd.Timestamp(date_str)
        except:
            continue

        for ticker in QUANTUM_TICKERS:
            if ticker not in signal_vector or ticker not in returns_df.columns:
                continue
            predicted_score = signal_vector[ticker].get("score", 0)
            if predicted_score == 0:
                continue

            stock_returns = returns_df[ticker]
            market_returns = returns_df[MARKET_TICKER]

            pre_event = stock_returns[stock_returns.index < event_date]
            if len(pre_event) < 60:
                continue

            gap_date = event_date - timedelta(days=14)
            est_stock = pre_event[pre_event.index < gap_date].tail(180)
            est_market = market_returns[market_returns.index < gap_date].tail(180)

            aligned = pd.concat([est_stock, est_market], axis=1).dropna()
            aligned.columns = ["stock", "market"]
            if len(aligned) < 60:
                continue

            X = sm.add_constant(aligned["market"])
            try:
                model = sm.OLS(aligned["stock"], X).fit()
                alpha, beta = model.params.iloc[0], model.params.iloc[1]
            except:
                continue

            post_stock = stock_returns[stock_returns.index > event_date]
            post_market = market_returns[market_returns.index > event_date]

            if len(post_stock) < horizon:
                continue
            ws = post_stock.iloc[:horizon]
            wm = post_market.iloc[:horizon]
            idx = ws.index.intersection(wm.index)
            if len(idx) < 1:
                continue
            expected = alpha + beta * wm.loc[idx]
            ar = ws.loc[idx] - expected
            car = ar.sum()

            results.append({
                "source": source,
                "ticker": ticker,
                "predicted_score": predicted_score,
                "car": car,
                "date": date_str,
                "title": title,
                "article_idx": pred.get("article_idx"),
            })

    return pd.DataFrame(results)


# Load
returns_df = load_returns()
nemotron_preds = load_predictions(EVAL_DIR / "predictions_openreasoning7b_v4.jsonl")
obs = compute_observations(nemotron_preds, returns_df, horizon=10)

print("=" * 70)
print("QUESTION: Is the model just a hype machine (always bullish)?")
print("=" * 70)

bullish = obs[obs["predicted_score"] > 0]
bearish = obs[obs["predicted_score"] < 0]

print(f"\n  Total non-zero predictions: {len(obs)}")
print(f"  Bullish (score > 0): {len(bullish)} ({len(bullish)/len(obs)*100:.1f}%)")
print(f"  Bearish (score < 0): {len(bearish)} ({len(bearish)/len(obs)*100:.1f}%)")

print(f"\n  Average bullish score: {bullish['predicted_score'].mean():+.3f}")
print(f"  Average bearish score: {bearish['predicted_score'].mean():+.3f}")

print("\n" + "=" * 70)
print("IC SPLIT: Bullish predictions vs Bearish predictions")
print("=" * 70)

if len(bullish) >= 10:
    corr_b, pval_b = stats.pearsonr(bullish["predicted_score"], bullish["car"])
    avg_car_b = bullish["car"].mean()
    print(f"\n  BULLISH predictions (N={len(bullish)}):")
    print(f"    IC: {corr_b:+.4f} (p={pval_b:.4f})")
    print(f"    Avg CAR when model says bullish: {avg_car_b:+.4f} ({avg_car_b*100:+.2f}%)")
    print(f"    % of time stock actually went up: {(bullish['car'] > 0).mean()*100:.1f}%")

if len(bearish) >= 10:
    corr_bear, pval_bear = stats.pearsonr(bearish["predicted_score"], bearish["car"])
    avg_car_bear = bearish["car"].mean()
    print(f"\n  BEARISH predictions (N={len(bearish)}):")
    print(f"    IC: {corr_bear:+.4f} (p={pval_bear:.4f})")
    print(f"    Avg CAR when model says bearish: {avg_car_bear:+.4f} ({avg_car_bear*100:+.2f}%)")
    print(f"    % of time stock actually went down: {(bearish['car'] < 0).mean()*100:.1f}%")
else:
    print(f"\n  BEARISH predictions: only {len(bearish)} observations (too few for IC)")

print("\n" + "=" * 70)
print("HYPE MACHINE TEST: What happens if we just predict +1.0 for everything?")
print("=" * 70)

# Simulate a naive "always bullish" model
all_cars = obs["car"]
naive_ic = all_cars.mean()  # If all predictions are +1, IC = mean(CAR)
print(f"\n  If model always predicted +1.0:")
print(f"    Mean CAR across all observations: {naive_ic:+.4f} ({naive_ic*100:+.2f}%)")
print(f"    This would be the 'IC' of a constant bullish model")
print(f"    Actual Nemotron IC: +0.151")
print(f"    Difference: Nemotron adds {0.151 - naive_ic:.4f} beyond naive bullish")

print("\n" + "=" * 70)
print("DIRECTIONAL ACCURACY: Bullish vs Bearish")
print("=" * 70)

if len(bullish) > 0:
    bull_correct = (bullish["car"] > 0).sum()
    print(f"\n  Bullish predictions: {bull_correct}/{len(bullish)} correct ({bull_correct/len(bullish)*100:.1f}%)")

if len(bearish) > 0:
    bear_correct = (bearish["car"] < 0).sum()
    print(f"  Bearish predictions: {bear_correct}/{len(bearish)} correct ({bear_correct/len(bearish)*100:.1f}%)")

print("\n" + "=" * 70)
print("SAMPLE BEARISH PREDICTIONS (if any)")
print("=" * 70)

# Show some bearish predictions
bearish_preds = []
for pred in nemotron_preds:
    sv = pred.get("signal", {}).get("signal_vector", {})
    for ticker in ["IONQ", "RGTI", "QBTS", "QUBT"]:
        if ticker in sv:
            score = sv[ticker].get("score", 0)
            if isinstance(score, (int, float)) and score < -0.3:
                bearish_preds.append({
                    "title": pred.get("title", "")[:60],
                    "ticker": ticker,
                    "score": score,
                    "reasoning": sv[ticker].get("reasoning", "")[:80],
                })

print(f"\n  Found {len(bearish_preds)} bearish predictions (score < -0.3) on pure-play tickers:")
for bp in bearish_preds[:10]:
    print(f"    {bp['ticker']} {bp['score']:+.1f} | {bp['title']}")
    print(f"      Reasoning: {bp['reasoning']}")
