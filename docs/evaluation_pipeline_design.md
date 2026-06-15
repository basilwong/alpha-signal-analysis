# Evaluation Pipeline: Engineering Design Document

## 1. Objective

Build a reproducible evaluation pipeline that quantifies the predictive power of the Quantum Alpha model by computing Abnormal Returns, Information Coefficient, and Signal Decay metrics against historical market data. The pipeline must also produce visualizations suitable for embedding in the Gradio app and the hackathon submission materials.

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PIPELINE                                     │
│                                                                                 │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐                │
│  │  DATA LAYER   │────▶│  COMPUTE LAYER│────▶│  OUTPUT LAYER │                │
│  └───────────────┘     └───────────────┘     └───────────────┘                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

DATA LAYER:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐           │
│  │  Article Dataset  │   │  Model Predictions│   │  Market Data     │           │
│  │                  │   │                  │   │                  │           │
│  │  data/raw/       │   │  data/eval/      │   │  data/market/    │           │
│  │  articles.jsonl  │   │  predictions.jsonl│   │  prices.parquet  │           │
│  │                  │   │                  │   │                  │           │
│  │  Fields:         │   │  Fields:         │   │  Fields:         │           │
│  │  - text          │   │  - article_id    │   │  - date          │           │
│  │  - date          │   │  - ticker        │   │  - ticker        │           │
│  │  - source        │   │  - sentiment     │   │  - open          │           │
│  │  - title         │   │  - magnitude     │   │  - high          │           │
│  │                  │   │  - pct_range     │   │  - low           │           │
│  │                  │   │  - time_horizon  │   │  - close         │           │
│  │                  │   │  - decay         │   │  - volume        │           │
│  │                  │   │  - cross_assets  │   │  - adj_close     │           │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

COMPUTE LAYER:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Step 1: MARKET MODEL ESTIMATION                                          │  │
│  │                                                                          │  │
│  │  For each (ticker, event_date) pair:                                     │  │
│  │    1. Define estimation window: [event_date - 250d, event_date - 10d]    │  │
│  │    2. Pull daily returns for ticker and market (SPY) in that window      │  │
│  │    3. Run OLS regression: R_stock = alpha + beta * R_market              │  │
│  │    4. Store alpha, beta, R_squared for this ticker-event pair            │  │
│  │                                                                          │  │
│  │  Also estimate sector beta using quantum basket:                         │  │
│  │    R_stock = alpha + beta_mkt * R_SPY + beta_sector * R_quantum_basket   │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│                                    ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Step 2: ABNORMAL RETURN CALCULATION                                      │  │
│  │                                                                          │  │
│  │  For each event:                                                         │  │
│  │    1. Define event window: [event_date + 1, event_date + N]              │  │
│  │    2. For each day t in event window:                                    │  │
│  │       AR_t = R_stock_t - (alpha + beta * R_market_t)                     │  │
│  │    3. Compute CAR for multiple windows:                                  │  │
│  │       CAR(+1,+1), CAR(+1,+2), CAR(+1,+5), CAR(+1,+10), CAR(+1,+20)    │  │
│  │    4. Also compute sector-adjusted AR using two-factor model             │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│                                    ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Step 3: INFORMATION COEFFICIENT                                          │  │
│  │                                                                          │  │
│  │  1. Map model sentiment to numeric score:                                │  │
│  │     strongly_bearish=-2, bearish=-1, neutral=0, bullish=+1,              │  │
│  │     strongly_bullish=+2                                                  │  │
│  │                                                                          │  │
│  │  2. Compute overall IC:                                                  │  │
│  │     IC = SpearmanRankCorrelation(predicted_scores, CAR(+1,+5))           │  │
│  │                                                                          │  │
│  │  3. Compute IC by subset:                                                │  │
│  │     IC_by_source = {news: IC, arxiv: IC, sec_filing: IC, ...}            │  │
│  │     IC_by_ticker = {IONQ: IC, RGTI: IC, QBTS: IC, ...}                  │  │
│  │     IC_by_event_type = {qubit_milestone: IC, earnings: IC, ...}          │  │
│  │                                                                          │  │
│  │  4. Statistical significance:                                            │  │
│  │     t = IC * sqrt(n-2) / sqrt(1 - IC^2)                                 │  │
│  │     p_value = 2 * (1 - t_distribution.cdf(abs(t), df=n-2))              │  │
│  │     Apply Bonferroni correction for subset tests:                        │  │
│  │       adjusted_alpha = 0.05 / num_subsets_tested                         │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│                                    ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Step 4: SIGNAL DECAY ANALYSIS                                            │  │
│  │                                                                          │  │
│  │  Compute IC at multiple horizons:                                        │  │
│  │    horizons = [1, 2, 3, 5, 10, 20, 40, 60]                              │  │
│  │    for h in horizons:                                                    │  │
│  │      IC_h = SpearmanRankCorrelation(predicted_scores, CAR(+1, +h))       │  │
│  │                                                                          │  │
│  │  Compute decay by source type:                                           │  │
│  │    For each source in [news, arxiv, sec_filing, press_release]:          │  │
│  │      decay_curve_source = [IC_h for h in horizons]                       │  │
│  │                                                                          │  │
│  │  Identify optimal holding period:                                        │  │
│  │    optimal_horizon = argmax(IC_h)                                        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│                                    ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Step 5: CROSS-ASSET VALIDATION                                           │  │
│  │                                                                          │  │
│  │  For each prediction that includes cross_asset_signals:                  │  │
│  │    1. Extract predicted direction for secondary tickers                  │  │
│  │    2. Compute CAR for those tickers over the same window                 │  │
│  │    3. Check if direction matches:                                        │  │
│  │       cross_asset_accuracy = correct_directions / total_predictions      │  │
│  │    4. Compute cross-asset IC separately                                  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

OUTPUT LAYER:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Output 1: EVALUATION REPORT (Markdown/PDF)                               │  │
│  │                                                                          │  │
│  │  - Summary statistics table                                              │  │
│  │  - IC values with confidence intervals and p-values                      │  │
│  │  - Comparison: fine-tuned model vs base model                            │  │
│  │  - Limitations and caveats section                                       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Output 2: VISUALIZATIONS (PNG/Interactive)                               │  │
│  │                                                                          │  │
│  │  Chart 1: Signal Decay Curve                                             │  │
│  │    X-axis: Holding period (days)                                         │  │
│  │    Y-axis: IC value                                                      │  │
│  │    Lines: Overall, by source type, by event type                         │  │
│  │                                                                          │  │
│  │  Chart 2: CAR Event Study Plot                                           │  │
│  │    X-axis: Days relative to event [-5, +20]                              │  │
│  │    Y-axis: Average CAR (%)                                               │  │
│  │    Lines: Bullish signals, Bearish signals, All signals                  │  │
│  │                                                                          │  │
│  │  Chart 3: IC by Subset (Bar Chart)                                       │  │
│  │    Grouped bars: by source, by ticker, by event type                     │  │
│  │    Error bars: 95% bootstrap confidence intervals                        │  │
│  │                                                                          │  │
│  │  Chart 4: Predicted vs Realized Scatter                                  │  │
│  │    X-axis: Model predicted score                                         │  │
│  │    Y-axis: Realized CAR(+1,+5)                                           │  │
│  │    Color: by source type                                                 │  │
│  │    Annotation: Spearman rho and p-value                                  │  │
│  │                                                                          │  │
│  │  Chart 5: Confusion Matrix (Direction Accuracy)                          │  │
│  │    Predicted direction vs actual direction                               │  │
│  │    Cells: count and percentage                                           │  │
│  │                                                                          │  │
│  │  Chart 6: Model Comparison (Fine-tuned vs Base)                          │  │
│  │    Side-by-side IC values across all metrics                             │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Output 3: STRUCTURED RESULTS (JSON)                                      │  │
│  │                                                                          │  │
│  │  Machine-readable results for embedding in the Gradio app:               │  │
│  │  {                                                                       │  │
│  │    "overall_ic": 0.07,                                                   │  │
│  │    "overall_ic_pvalue": 0.03,                                            │  │
│  │    "directional_accuracy": 0.68,                                         │  │
│  │    "optimal_holding_period_days": 5,                                     │  │
│  │    "n_events_evaluated": 187,                                            │  │
│  │    "ic_by_source": {...},                                                │  │
│  │    "ic_by_ticker": {...},                                                │  │
│  │    "decay_curve": [...],                                                 │  │
│  │    "model_comparison": {...}                                              │  │
│  │  }                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 3. Module Breakdown

### Module 1: `eval/market_data.py`

Responsible for downloading and caching historical price data.

```python
class MarketDataProvider:
    """Downloads and caches market data from Yahoo Finance."""

    def __init__(self, cache_dir: str = "data/market/"):
        ...

    def get_daily_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Returns daily OHLCV data for a ticker."""
        ...

    def get_daily_returns(self, ticker: str, start: str, end: str) -> pd.Series:
        """Returns daily percentage returns (close-to-close)."""
        ...

    def get_market_returns(self, start: str, end: str) -> pd.Series:
        """Returns S&P 500 daily returns."""
        ...

    def get_sector_returns(self, start: str, end: str, exclude_ticker: str = None) -> pd.Series:
        """Returns equal-weighted quantum sector basket returns."""
        ...
```

**Tickers to download:**
- Primary: IONQ, RGTI, QBTS, QUBT, IBM, GOOGL, MSFT, HON, NVDA
- Market: SPY (S&P 500 ETF)
- Sector basket: Equal-weighted average of IONQ, RGTI, QBTS

**Caching strategy:** Download once, store as Parquet files in `data/market/`. Check if data exists before re-downloading. Append new data if cache is stale.

### Module 2: `eval/market_model.py`

Estimates alpha and beta for each ticker-event pair.

```python
class MarketModel:
    """Estimates expected returns using OLS regression."""

    def __init__(self, estimation_window: int = 180, gap: int = 10):
        ...

    def estimate(self, stock_returns: pd.Series, market_returns: pd.Series,
                 sector_returns: pd.Series, event_date: str) -> dict:
        """
        Returns:
            {
                "alpha": float,
                "beta_market": float,
                "beta_sector": float,  # from two-factor model
                "r_squared": float,
                "estimation_start": str,
                "estimation_end": str,
                "n_observations": int,
            }
        """
        ...

    def expected_return(self, params: dict, market_return: float,
                       sector_return: float = None) -> float:
        """Compute expected return given market model parameters."""
        ...
```

**Design decisions:**
- Estimation window: 180 trading days (approximately 9 months)
- Gap: 10 days between estimation window end and event date (prevents event contamination)
- Minimum observations: Require at least 100 valid trading days in estimation window. Skip events where the stock was recently IPO'd or had insufficient history.
- Two models estimated: single-factor (market only) and two-factor (market + sector)

### Module 3: `eval/abnormal_returns.py`

Computes AR and CAR for each event.

```python
class AbnormalReturnCalculator:
    """Computes abnormal and cumulative abnormal returns."""

    def __init__(self, market_model: MarketModel, market_data: MarketDataProvider):
        ...

    def compute_ar(self, ticker: str, event_date: str,
                   window: tuple = (1, 5)) -> dict:
        """
        Compute abnormal returns for a single event.

        Returns:
            {
                "ticker": str,
                "event_date": str,
                "daily_ar": [float, ...],  # AR for each day in window
                "car": float,  # Cumulative AR over window
                "car_windows": {
                    "(+1,+1)": float,
                    "(+1,+2)": float,
                    "(+1,+5)": float,
                    "(+1,+10)": float,
                    "(+1,+20)": float,
                },
                "model_params": dict,  # alpha, beta used
                "sector_adjusted_car": dict,  # same windows, two-factor model
            }
        """
        ...

    def compute_batch(self, events: list[dict]) -> pd.DataFrame:
        """Compute AR for all events. Returns a DataFrame."""
        ...
```

**Edge cases handled:**
- Event on non-trading day: Roll forward to next trading day
- Ticker not yet public on event date: Skip event, log warning
- Insufficient estimation window: Skip event, log warning
- Overlapping events (same ticker within 5 days): Flag but include both; note in limitations

### Module 4: `eval/information_coefficient.py`

Computes IC and statistical tests.

```python
class InformationCoefficientCalculator:
    """Computes IC, significance tests, and subset analysis."""

    def __init__(self):
        ...

    def compute_ic(self, predicted_scores: np.array,
                   realized_returns: np.array) -> dict:
        """
        Returns:
            {
                "ic": float,
                "p_value": float,
                "t_statistic": float,
                "n_observations": int,
                "ci_lower": float,  # 95% bootstrap CI
                "ci_upper": float,
            }
        """
        ...

    def compute_ic_by_subset(self, predictions_df: pd.DataFrame,
                            group_col: str) -> dict:
        """
        Compute IC for each group (source type, ticker, event type).
        Applies Bonferroni correction.

        Returns:
            {
                "group_name_1": {"ic": float, "p_value": float, "n": int, ...},
                "group_name_2": {...},
                "bonferroni_threshold": float,
            }
        """
        ...

    def compute_decay_curve(self, predicted_scores: np.array,
                           car_by_horizon: dict) -> list:
        """
        Returns:
            [
                {"horizon": 1, "ic": float, "p_value": float},
                {"horizon": 2, "ic": float, "p_value": float},
                ...
                {"horizon": 60, "ic": float, "p_value": float},
            ]
        """
        ...

    def bootstrap_ci(self, predicted_scores: np.array,
                     realized_returns: np.array,
                     n_bootstrap: int = 1000) -> tuple:
        """Returns (ci_lower, ci_upper) at 95% confidence."""
        ...
```

### Module 5: `eval/visualizations.py`

Generates all charts.

```python
class EvaluationVisualizer:
    """Generates publication-quality evaluation charts."""

    def __init__(self, output_dir: str = "data/eval/charts/"):
        ...

    def plot_decay_curve(self, decay_data: list, by_source: dict = None) -> str:
        """Signal decay curve. Returns path to saved PNG."""
        ...

    def plot_event_study(self, car_data: pd.DataFrame) -> str:
        """Average CAR around event date [-5, +20]. Returns path."""
        ...

    def plot_ic_by_subset(self, ic_subsets: dict) -> str:
        """Bar chart of IC by source/ticker/event type. Returns path."""
        ...

    def plot_predicted_vs_realized(self, predictions_df: pd.DataFrame) -> str:
        """Scatter plot with regression line. Returns path."""
        ...

    def plot_confusion_matrix(self, predictions_df: pd.DataFrame) -> str:
        """Direction accuracy confusion matrix. Returns path."""
        ...

    def plot_model_comparison(self, finetuned_results: dict,
                            base_results: dict) -> str:
        """Side-by-side comparison of fine-tuned vs base model. Returns path."""
        ...
```

### Module 6: `eval/pipeline.py`

Orchestrates the full evaluation.

```python
class EvaluationPipeline:
    """End-to-end evaluation orchestrator."""

    def __init__(self, config: dict):
        self.market_data = MarketDataProvider()
        self.market_model = MarketModel()
        self.ar_calculator = AbnormalReturnCalculator(...)
        self.ic_calculator = InformationCoefficientCalculator()
        self.visualizer = EvaluationVisualizer()

    def run(self, predictions_path: str, articles_path: str) -> dict:
        """
        Full evaluation pipeline.

        Steps:
            1. Load predictions and articles
            2. Download/cache market data for all tickers and date range
            3. For each event: estimate market model, compute AR and CAR
            4. Compute overall IC and IC by subset
            5. Compute signal decay curve
            6. Validate cross-asset predictions
            7. Generate all visualizations
            8. Compile results into structured output

        Returns:
            Full results dictionary (saved to data/eval/results.json)
        """
        ...

    def generate_report(self, results: dict) -> str:
        """Generate markdown evaluation report. Returns path."""
        ...
```

## 4. Data Flow Sequence

```
1. GENERATE PREDICTIONS
   ┌─────────────────────────────────────────────────────────────────┐
   │  Input: data/raw/articles.jsonl (611 articles with dates)        │
   │  Process: Run model on each article (Modal or Qwen Cloud API)   │
   │  Output: data/eval/predictions.jsonl                            │
   │                                                                 │
   │  Each prediction contains:                                      │
   │  {                                                              │
   │    "article_id": 0,                                             │
   │    "article_date": "2025-03-15",                                │
   │    "primary_ticker": "IONQ",                                    │
   │    "sentiment": "bullish",                                      │
   │    "sentiment_score": 1,  # numeric mapping                     │
   │    "expected_move_magnitude": "significant",                    │
   │    "expected_move_pct_range": [5.0, 12.0],                      │
   │    "time_horizon": "2-5 days",                                  │
   │    "signal_decay": "slow",                                      │
   │    "cross_asset_signals": [...],                                │
   │    "source": "arxiv",                                           │
   │    "event_type": "logical_qubit_breakthrough"                   │
   │  }                                                              │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. DOWNLOAD MARKET DATA
   ┌─────────────────────────────────────────────────────────────────┐
   │  For each unique ticker in predictions:                         │
   │    Download daily prices from Yahoo Finance                     │
   │    Date range: earliest_event - 300d to latest_event + 60d     │
   │  Also download: SPY (market factor)                             │
   │  Construct: quantum sector basket (equal-weight IONQ+RGTI+QBTS)│
   │  Cache: data/market/{ticker}.parquet                            │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. COMPUTE ABNORMAL RETURNS
   ┌─────────────────────────────────────────────────────────────────┐
   │  For each prediction:                                           │
   │    1. Estimate market model (180-day window before event)       │
   │    2. Compute daily AR for days +1 through +60                  │
   │    3. Compute CAR at windows: +1, +2, +5, +10, +20, +60        │
   │    4. Also compute sector-adjusted CAR (two-factor)             │
   │  Output: data/eval/abnormal_returns.parquet                     │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
4. COMPUTE METRICS
   ┌─────────────────────────────────────────────────────────────────┐
   │  Overall IC (Spearman rank correlation)                         │
   │  IC by source type (with Bonferroni correction)                 │
   │  IC by ticker (with Bonferroni correction)                      │
   │  IC by event type (with Bonferroni correction)                  │
   │  Signal decay curve (IC at horizons 1,2,3,5,10,20,40,60)       │
   │  Directional accuracy (% correct direction)                     │
   │  Magnitude calibration (predicted range vs realized)            │
   │  Cross-asset accuracy                                           │
   │  Bootstrap confidence intervals (1000 resamples)                │
   │  Output: data/eval/results.json                                 │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
5. GENERATE OUTPUTS
   ┌─────────────────────────────────────────────────────────────────┐
   │  Visualizations: data/eval/charts/*.png                         │
   │  Report: data/eval/evaluation_report.md                         │
   │  Structured results: data/eval/results.json                     │
   │  App integration: results embedded in Gradio "Evaluation" tab   │
   └─────────────────────────────────────────────────────────────────┘
```

## 5. Pitfalls and Mitigations

| Pitfall | Risk | Mitigation | Documentation |
|---------|------|------------|---------------|
| **Look-ahead bias** | Using event-day close as baseline when article published during market hours | Use previous day's close as baseline for all events. Document that intraday timing is not accounted for. | Limitations section: "Article publication timestamps are daily granularity. Intraday timing effects are not captured." |
| **Survivorship bias** | Only evaluating companies that still exist | For our 2-year window, no major quantum companies have delisted. Document this explicitly. | Limitations section: "Evaluation covers June 2024 to June 2026. No quantum computing companies in our universe delisted during this period." |
| **Overlapping events** | Two articles about same ticker within event window | Flag overlapping events. Report results both including and excluding overlaps. | Results section: "X events had overlapping windows. Results with overlaps excluded: IC = Y." |
| **Multiple testing** | Testing IC across many subsets inflates false positive rate | Apply Bonferroni correction. Report both raw and corrected p-values. | Methods section: "Bonferroni-corrected significance threshold: alpha/k where k = number of subsets tested." |
| **Small sample size** | 200 events may not achieve statistical significance | Report confidence intervals. Acknowledge power limitations. Compute minimum detectable IC. | Limitations section: "With n=X observations, the minimum detectable IC at 5% significance is Y." |
| **Non-stationarity** | Market regime changes over 2 years | Report IC by sub-period (H1 2025, H2 2025, H1 2026). Note any regime shifts. | Results section: "IC by sub-period: [table]." |
| **Model data leakage** | Training data overlapping with evaluation data | Use strict temporal split. Training data generated from first 200 articles. Evaluate on remaining 411. | Methods section: "Strict temporal separation between training (articles 1-200) and evaluation (articles 201-611)." |
| **Sector basket construction** | Equal-weight basket may not represent sector well | Report results with and without sector adjustment. Acknowledge basket limitations. | Methods section: "Sector basket: equal-weighted IONQ, RGTI, QBTS. Alternative: market-cap weighted." |
| **Yahoo Finance data quality** | Adjusted close may have errors around splits/dividends | Use adjusted close for returns. Manually verify any returns > 50% in a single day. | Data section: "Returns computed from adjusted close prices. Outliers > 50% daily return flagged for manual review." |
| **Publication timing uncertainty** | RSS feed dates may not match actual publication time | Use conservative t+1 window start (never include event day in CAR). | Methods section: "Event window starts t+1 to avoid any same-day contamination." |

## 6. Gradio App Integration

The evaluation results will be displayed in a new "Model Performance" tab in the Gradio app:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Tab: Model Performance                                             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Summary Metrics (top row)                                   │   │
│  │                                                             │   │
│  │  IC: 0.07 (p=0.03)  |  Direction Accuracy: 68%  |          │   │
│  │  Optimal Horizon: 5d  |  Events Evaluated: 187              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐   │
│  │  Signal Decay Curve      │  │  Predicted vs Realized        │   │
│  │  (interactive plot)      │  │  (scatter plot)               │   │
│  └──────────────────────────┘  └──────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐   │
│  │  IC by Source Type       │  │  Event Study (avg CAR)        │   │
│  │  (bar chart)             │  │  (line chart)                 │   │
│  └──────────────────────────┘  └──────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Limitations & Caveats (collapsible)                         │   │
│  │  - Sample size: 187 events                                  │   │
│  │  - Period: Jun 2024 - Jun 2026                              │   │
│  │  - No intraday timing                                       │   │
│  │  - Bonferroni-corrected subset tests                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 7. File Structure

```
alpha-signal-analysis/
├── eval/
│   ├── __init__.py
│   ├── market_data.py          # Yahoo Finance data provider
│   ├── market_model.py         # OLS estimation (alpha, beta)
│   ├── abnormal_returns.py     # AR and CAR computation
│   ├── information_coefficient.py  # IC, significance, bootstrap
│   ├── visualizations.py       # All chart generation
│   ├── pipeline.py             # Orchestrator
│   └── config.py               # Evaluation parameters
├── data/
│   ├── eval/
│   │   ├── predictions.jsonl   # Model outputs for all articles
│   │   ├── abnormal_returns.parquet
│   │   ├── results.json        # Final structured results
│   │   └── charts/             # Generated visualizations
│   └── market/
│       ├── IONQ.parquet        # Cached price data
│       ├── RGTI.parquet
│       ├── SPY.parquet
│       └── ...
└── scripts/
    └── run_evaluation.py       # CLI entry point
```

## 8. Execution

```bash
# Step 1: Generate predictions for all articles (uses Modal or Qwen API)
python scripts/generate_predictions.py \
    --input data/raw/articles.jsonl \
    --output data/eval/predictions.jsonl \
    --model basilwong/quantum-alpha-qwen3-8b

# Step 2: Run the full evaluation pipeline
python scripts/run_evaluation.py \
    --predictions data/eval/predictions.jsonl \
    --output-dir data/eval/ \
    --generate-charts \
    --generate-report

# Step 3 (optional): Compare against base model
python scripts/generate_predictions.py \
    --input data/raw/articles.jsonl \
    --output data/eval/predictions_base.jsonl \
    --model Qwen/Qwen3-8B

python scripts/run_evaluation.py \
    --predictions data/eval/predictions_base.jsonl \
    --output-dir data/eval/base/ \
    --compare-to data/eval/results.json
```

## 9. Dependencies

```
yfinance          # Yahoo Finance data
pandas            # DataFrames
numpy             # Numerics
scipy             # Statistical tests (t-distribution, Spearman)
statsmodels       # OLS regression
matplotlib        # Static charts
plotly            # Interactive charts (for Gradio embedding)
```

All available via pip, no API keys required.
