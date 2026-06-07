# Background Report: Alphalens and EventStudy Libraries

## Introduction

This report provides deep context on two open source Python libraries that can handle the majority of our evaluation pipeline. Understanding what these tools are, where they came from, and what makes them the standard choices will help us make informed decisions about how to integrate them.

## Part 1: Alphalens (and Alphalens-Reloaded)

### Origin and History

Alphalens was created by **Quantopian**, a company that operated a crowd-sourced quantitative hedge fund from 2011 to 2020. Quantopian's business model was unique: they provided a free platform where anyone could write trading algorithms, and the best-performing algorithms received real capital allocation from Quantopian's fund. At its peak, Quantopian had over 200,000 registered users writing quantitative strategies.

To support this ecosystem, Quantopian built and open-sourced a suite of tools:
- **Zipline**: A backtesting engine for running strategies against historical data
- **Pyfolio**: Portfolio performance and risk analysis
- **Alphalens**: Alpha factor analysis (the tool we care about)

These three tools became the de facto standard in the Python quant community because they were battle-tested against thousands of real strategies submitted by Quantopian's users. The tools were designed to answer the exact question every quant researcher asks: "Is my signal actually predictive, or am I fooling myself?"

Quantopian shut down in 2020 (their fund underperformed), but their open source tools lived on. **Stefan Jansen**, author of the book "Machine Learning for Algorithmic Trading" (published by Packt, widely considered the best practical ML-for-trading textbook), forked and maintains the updated version as `alphalens-reloaded`. This fork has 594 stars, is actively maintained (latest release July 2025), and is compatible with modern pandas/numpy versions.

### What Alphalens Actually Does

Alphalens takes two inputs:
1. **A factor signal**: Your predictions, indexed by (date, ticker). For us, this is the sentiment score our model assigns to each article.
2. **Pricing data**: Historical prices for the relevant tickers.

From these two inputs, it automatically computes and visualizes:

**Returns Analysis:**
- Mean return by quantile (if you split signals into 5 buckets from most bearish to most bullish, does the top bucket actually outperform the bottom bucket?)
- Cumulative return spread (long top quantile, short bottom quantile over time)
- Return by holding period (1 day, 5 days, 10 days, etc.)

**Information Coefficient Analysis:**
- Daily IC (Spearman rank correlation between signal and forward returns for each cross-section)
- Mean IC over time
- IC by month (to detect regime changes)
- IC heatmap (by month and year)
- IC distribution histogram

**Turnover Analysis:**
- How frequently does the signal change? (High turnover = high transaction costs)
- Autocorrelation of factor ranks (is the signal stable or noisy?)

**Grouped Analysis:**
- IC by sector/group (for us: by source type, by ticker, by event type)
- Return spread by group

All of this is produced by calling a single function: `create_full_tear_sheet()`. The output is a comprehensive set of charts and tables that would take hundreds of lines of custom code to replicate.

### Why It's Considered the Gold Standard

1. **Provenance**: Built by a team that managed real capital based on these exact metrics. The tool was designed to prevent self-deception, not to make signals look good.

2. **Quantile analysis**: Instead of just computing one IC number, Alphalens shows you the full distribution of returns across signal quantiles. This reveals whether your signal is monotonic (top quintile beats 4th beats 3rd beats 2nd beats bottom) or just binary (top vs. bottom with noise in between).

3. **Forward return computation**: Alphalens correctly handles the computation of forward returns, including adjustments for dividends and splits. Getting this wrong is a common source of bugs in custom implementations.

4. **Statistical rigor**: It computes proper standard errors, handles overlapping observations correctly, and reports confidence intervals.

5. **Community adoption**: Because Quantopian had 200,000+ users, Alphalens became the lingua franca. When quant researchers share results, they often share Alphalens tearsheets. Judges who have a quant background will immediately recognize and trust this format.

### Alternatives to Alphalens

| Library | Strengths | Weaknesses | Stars |
|---------|-----------|------------|-------|
| **alphalens-reloaded** | Full-featured, well-maintained, industry standard | Requires specific data format (MultiIndex DataFrame) | 594 |
| **QuantStats** | Beautiful reports, Sharpe/Sortino/drawdown analysis | More focused on portfolio returns than factor analysis | 4.8K |
| **vectorbt** | Extremely fast backtesting, GPU-accelerated | Overkill for factor evaluation, steep learning curve | 4.2K |
| **empyrical** (by Quantopian) | Risk metrics (Sharpe, max drawdown, etc.) | Narrow scope, doesn't do IC analysis | 1.2K |
| **Custom scipy.stats** | Full control, no dependencies | Have to implement everything yourself, easy to make mistakes | N/A |

**Verdict**: For IC analysis and factor evaluation specifically, alphalens-reloaded has no real competitor. QuantStats is excellent for portfolio-level analysis but doesn't compute IC or quantile returns. Vectorbt is a backtesting engine, not a factor evaluator. For our use case (evaluating an NLP signal's predictive power), alphalens-reloaded is the clear choice.

### Limitations of Alphalens

1. **Assumes cross-sectional signals**: Alphalens is designed for signals that rank stocks against each other at each point in time. Our signal is more event-driven (one article about one stock). We'll need to adapt our data format slightly.

2. **Doesn't do event studies**: Alphalens computes forward returns from the signal date, but it doesn't estimate a market model or compute abnormal returns. It uses raw returns (or excess returns over a benchmark), not regression-residual abnormal returns.

3. **Requires sufficient cross-sectional breadth**: Alphalens works best when you have signals for many stocks at each time point. With only 9 quantum tickers, some of its analyses (like quantile returns) will be noisy.

4. **No causal inference**: Alphalens tells you "the signal is correlated with future returns" but cannot prove causation. The signal might be correlated with another factor (like momentum) that is the true driver.

## Part 2: EventStudy

### Origin and History

The `eventstudy` Python package was created by **Jean-Baptiste Lemaire**, a French quantitative finance researcher. It was published in 2020 as an open-source implementation of the classical event study methodology that has been the standard in academic finance since the 1960s.

The event study methodology itself was formalized by **Ball and Brown (1968)** and **Fama, Fisher, Jensen, and Roll (1969)**. It has been used in thousands of academic papers to measure the market impact of events like earnings announcements, mergers, regulatory changes, and CEO departures. The methodology is so well-established that it is taught in every MBA finance program and is the standard approach used by the SEC and DOJ to estimate damages in securities fraud cases.

### What EventStudy Actually Does

The library implements the complete event study workflow:

1. **Model estimation**: Runs OLS regression of stock returns against market returns (or Fama-French factors) over an estimation window to establish the "normal" relationship.

2. **Abnormal return computation**: For each day in the event window, computes the difference between actual return and expected return (based on the estimated model).

3. **Statistical testing**: Computes t-statistics and p-values for each day's abnormal return and for the cumulative abnormal return over the full window.

4. **Aggregation**: When you have multiple events, it computes Average Abnormal Returns (AAR) and Cumulative Average Abnormal Returns (CAAR) across all events, with proper variance estimation.

5. **Visualization**: Produces publication-ready plots showing CAR over time with confidence intervals.

**Supported models:**
- Market model (single factor: stock vs. S&P 500)
- Fama-French 3-factor model (market + size + value)
- Fama-French 5-factor model (market + size + value + profitability + investment)
- Custom user-defined models

### Example Output

For a single event, the library produces a table like:

| Day | AR | Variance AR | CAR | T-stat | P-value |
|-----|------|-------------|-------|--------|---------|
| -2 | 0.004 | 0.00048 | -0.051 | -1.15 | 0.13 |
| -1 | 0.000 | 0.00048 | -0.051 | -1.03 | 0.15 |
| 0 | -0.077 | 0.00048 | -0.128** | -2.37 | 0.01 |
| +1 | -0.039 | 0.00048 | -0.167*** | -2.88 | 0.00 |
| +2 | 0.027 | 0.00048 | -0.140** | -2.26 | 0.01 |

Asterisks indicate statistical significance (*** at 99%, ** at 95%, * at 90%).

For multiple events, it aggregates across all events and reports the average market reaction with proper statistical tests.

### Why It's Useful (But Not "Gold Standard")

The `eventstudy` package is useful because it saves you from implementing the OLS regression, variance estimation, and statistical testing yourself. However, it's important to note that it's a much smaller project than Alphalens:

- 69 stars (vs. Alphalens' 4,300+)
- 7 watchers
- Still in "alpha" according to its own documentation
- Last commit activity is sparse
- GPL-3.0 license (more restrictive than Alphalens' Apache-2.0)

It's a convenience tool, not an industry standard. The methodology it implements IS the gold standard (event studies are universally accepted in academic finance), but the specific library is just one implementation of that methodology.

### Alternatives to EventStudy

| Library | Strengths | Weaknesses | Stars |
|---------|-----------|------------|-------|
| **eventstudy** | Simple API, supports Fama-French, handles aggregation | Small project, alpha status, GPL license, limited maintenance | 69 |
| **event-study-toolkit** (PyPI) | Similar functionality | Very small, limited documentation | ~10 |
| **EasyEventStudies** | YouTube tutorial available, simple interface | Newer, less proven | ~20 |
| **Custom implementation** (statsmodels + pandas) | Full control, no GPL concerns, exactly what we need | 50-100 lines of code to write | N/A |
| **R packages** (eventstudies, EventStudy) | More mature, more features | Requires R, not Python | N/A |

**Verdict**: The `eventstudy` library is convenient but not irreplaceable. The event study methodology is straightforward enough (it's just OLS regression + subtraction) that a custom implementation in 50-100 lines using `statsmodels` and `pandas` would give us more control and avoid the GPL license concern. However, for speed of implementation, the library handles edge cases (missing data, error reporting) that would take time to implement from scratch.

### Limitations of EventStudy

1. **Data input format**: Requires pre-formatted CSV files or specific data structures. Doesn't integrate directly with Yahoo Finance.

2. **Alpha status**: The author explicitly states the API may change. Not production-grade.

3. **GPL-3.0 license**: This is a copyleft license. If we use it in our project and distribute the code, we'd need to license our code under GPL as well. For a hackathon this is fine, but for a commercial product it would be problematic.

4. **Limited to standard models**: Only supports market model and Fama-French. Our two-factor model (market + quantum sector basket) would need to be implemented as a custom model function.

5. **No sector-adjusted returns out of the box**: We'd need to extend it to support our quantum sector basket as a second factor.

## Part 3: Practical Recommendation

### What to Use Where

| Task | Recommended Approach | Reasoning |
|------|---------------------|-----------|
| **IC computation** | alphalens-reloaded | Industry standard, produces publication-quality tearsheets, handles all the statistics correctly |
| **Signal decay curve** | alphalens-reloaded (multi-period forward returns) | Built-in support for multiple holding periods |
| **Abnormal returns (CAR)** | Custom implementation (50 lines with statsmodels) | More control than eventstudy library, avoids GPL, supports our custom two-factor model |
| **Market data download** | yfinance | Free, reliable, no API key needed |
| **Statistical tests** | scipy.stats | Already available, handles Spearman correlation and t-tests |
| **Visualizations** | matplotlib + alphalens built-in charts | Alphalens charts for IC analysis, custom matplotlib for event study plots |

### The Hybrid Approach

Use alphalens-reloaded for the IC and factor evaluation (where it truly shines and saves us hundreds of lines of code), but write a lightweight custom implementation for the abnormal return calculation (where the eventstudy library adds complexity without proportional value).

This gives us:
- The credibility of alphalens tearsheets (judges will recognize them)
- Full control over the market model estimation (custom two-factor model)
- No GPL license concerns
- Approximately 300 total lines of custom code instead of 1000+

### Installation

```bash
pip install alphalens-reloaded yfinance scipy statsmodels matplotlib
```

All free, all permissively licensed (Apache-2.0 or BSD), all actively maintained.
