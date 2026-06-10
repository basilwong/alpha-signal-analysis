"""
Multi-Model Evaluation: Run IC analysis on all completed models and produce comparison.

Usage:
    python eval/run_multi_model_eval.py
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

# Models to evaluate
MODELS = {
    "Qwen3-8B Fine-tuned (LoRA)": EVAL_DIR / "predictions_finetuned_all.jsonl",
    "Qwen3-8B Base": EVAL_DIR / "predictions_qwen3_8b_base.jsonl",
    "Qwen3.7-Max Base": EVAL_DIR / "predictions_qwen37_max_base.jsonl",
}


def load_predictions(path):
    predictions = []
    with open(path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p.get("status") == "success":
                    predictions.append(p)
    return predictions


def load_returns():
    returns = {}
    for ticker in QUANTUM_TICKERS + [MARKET_TICKER]:
        path = MARKET_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            returns[ticker] = df[close_col].pct_change().dropna()
    return pd.DataFrame(returns)


def compute_abnormal_returns(predictions, returns_df, horizons=[1, 2, 5, 10, 20]):
    results = []
    for pred in predictions:
        date_str = pred.get("date", "")
        signal = pred.get("signal", {})
        signal_vector = signal.get("signal_vector", {})
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

            for horizon in horizons:
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
                    "article_idx": pred.get("article_idx"),
                    "date": date_str,
                    "ticker": ticker,
                    "predicted_score": predicted_score,
                    "horizon": horizon,
                    "car": car,
                })

    return pd.DataFrame(results)


def compute_ic(predicted, realized):
    mask = ~(np.isnan(predicted) | np.isnan(realized))
    p, r = predicted[mask], realized[mask]
    if len(p) < 10:
        return np.nan, np.nan, len(p)
    ic, pval = stats.spearmanr(p, r)
    return ic, pval, len(p)


def compute_direction_accuracy(predicted, realized):
    mask = ~(np.isnan(predicted) | np.isnan(realized))
    p, r = predicted[mask], realized[mask]
    nonzero = np.abs(p) > 0.05
    if nonzero.sum() < 5:
        return np.nan, 0
    correct = (np.sign(p[nonzero]) == np.sign(r[nonzero])).sum()
    return correct / nonzero.sum(), int(nonzero.sum())


def main():
    print("=" * 70)
    print("MULTI-MODEL EVALUATION")
    print("=" * 70)

    returns_df = load_returns()
    print(f"Market data: {returns_df.shape[0]} days, {returns_df.shape[1]} tickers")

    horizons = [1, 2, 5, 10, 20]
    all_results = {}

    for model_name, pred_path in MODELS.items():
        if not pred_path.exists():
            print(f"\n  SKIP {model_name}: file not found ({pred_path})")
            continue

        predictions = load_predictions(pred_path)
        print(f"\n{'='*70}")
        print(f"  {model_name}: {len(predictions)} predictions")
        print(f"{'='*70}")

        ar_df = compute_abnormal_returns(predictions, returns_df, horizons)
        print(f"  Observations: {len(ar_df)}")

        if ar_df.empty:
            all_results[model_name] = {"error": "no observations"}
            continue

        model_results = {"n_predictions": len(predictions), "n_observations": len(ar_df)}

        # IC at each horizon
        decay = []
        for h in horizons:
            subset = ar_df[ar_df["horizon"] == h]
            ic, pval, n = compute_ic(subset["predicted_score"].values, subset["car"].values)
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            decay.append({"horizon": h, "ic": ic, "p_value": pval, "n": n})
            print(f"    IC +{h:2d}d: {ic:+.4f} (p={pval:.4f}) {sig} n={n}")

        model_results["decay_curve"] = decay

        # Direction accuracy at +5d
        h5 = ar_df[ar_df["horizon"] == 5]
        acc, n_acc = compute_direction_accuracy(h5["predicted_score"].values, h5["car"].values)
        model_results["direction_accuracy_5d"] = {"accuracy": acc, "n": n_acc}
        print(f"    Dir Acc +5d: {acc*100:.1f}% (n={n_acc})")

        all_results[model_name] = model_results

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'IC+1d':>8} {'IC+5d':>8} {'IC+10d':>8} {'IC+20d':>8} {'DirAcc':>8} {'N':>6}")
    print("-" * 70)

    for model_name, results in all_results.items():
        if "error" in results:
            print(f"{model_name:<30} {'ERROR':>8}")
            continue
        decay = {d["horizon"]: d["ic"] for d in results["decay_curve"]}
        acc = results.get("direction_accuracy_5d", {}).get("accuracy", 0)
        n = results.get("n_predictions", 0)
        print(f"{model_name:<30} {decay.get(1,0):>+8.4f} {decay.get(5,0):>+8.4f} "
              f"{decay.get(10,0):>+8.4f} {decay.get(20,0):>+8.4f} {acc*100:>7.1f}% {n:>6}")

    # Save results
    output_path = EVAL_DIR / "results_multi_model.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
