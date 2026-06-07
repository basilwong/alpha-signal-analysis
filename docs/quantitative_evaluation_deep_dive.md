# Quantitative Evaluation of NLP Trading Signals: A Deep Dive

## Introduction

When a language model reads a news article and outputs "bullish, expected move +5% to +12%," how do we know if that prediction has any real value? This document explains the quantitative finance frameworks used to rigorously evaluate whether an NLP signal has genuine predictive power or is simply generating plausible-sounding noise.

The central challenge is **attribution**. On any given day, a stock's price movement is influenced by dozens of factors simultaneously: the overall market direction, sector-wide sentiment, interest rate expectations, earnings season dynamics, and the specific company news we're trying to measure. To evaluate our model, we need to mathematically isolate the portion of a stock's movement that is attributable to the specific piece of news our model analyzed, separate from everything else.

This document covers three interconnected frameworks: Event Study Methodology (Abnormal Returns), the Information Coefficient, and Signal Decay Analysis. Together, they answer the questions: "Does the model predict correctly?", "How strong is the predictive power?", and "How quickly does the information get priced into the market?"

## Part 1: Event Study Methodology and Abnormal Returns

### The Fundamental Idea

An event study asks a simple question: "What would this stock have done on this day if the news event had NOT occurred?" The difference between what actually happened and what would have happened is the **abnormal return**, which represents the market's reaction to the specific piece of information.

### Step 1: Calculating Raw Returns

The daily return for stock *i* on day *t* is:

```
R_i,t = (P_i,t - P_i,t-1) / P_i,t-1
```

Where `P_i,t` is the closing price on day *t*. For example, if IONQ closed at $32.00 yesterday and $34.56 today, the daily return is:

```
R = (34.56 - 32.00) / 32.00 = 0.08 = 8%
```

### Step 2: Estimating the Expected Return (The Market Model)

The market model assumes that a stock's return on any given day can be decomposed into two parts: (1) the portion explained by the overall market's movement, and (2) the portion specific to the company.

```
R_i,t = alpha_i + beta_i * R_m,t + epsilon_i,t
```

Where:
- `R_i,t` is the stock's return on day *t*
- `R_m,t` is the market return on day *t* (typically the S&P 500 or QQQ)
- `alpha_i` is the stock's average excess return independent of the market (often close to zero)
- `beta_i` is the stock's sensitivity to market movements (a beta of 2.0 means the stock moves 2% for every 1% the market moves)
- `epsilon_i,t` is the residual (the unexplained portion, which includes our news signal)

To estimate `alpha_i` and `beta_i`, we run an Ordinary Least Squares (OLS) regression using historical data from an **estimation window** that does NOT overlap with the event. Typically, this is 120 to 250 trading days ending 10 days before the event.

**Example**: To evaluate a news article published on March 15, 2026, we would estimate IONQ's alpha and beta using daily returns from approximately June 2025 through February 2025 (the 180 trading days ending 10 days before the event).

### Step 3: Calculating the Abnormal Return

Once we have alpha and beta, the abnormal return on the event day is:

```
AR_i,t = R_i,t - (alpha_i + beta_i * R_m,t)
```

**Concrete Example:**

Suppose on the day IonQ announces 35 algorithmic qubits:
- IONQ's actual return: +8%
- S&P 500 return that day: +1.2%
- IONQ's estimated beta: 2.5
- IONQ's estimated alpha: 0.001 (0.1%)

Then:
```
Expected Return = 0.001 + 2.5 * 0.012 = 0.001 + 0.030 = 0.031 = 3.1%
Abnormal Return = 0.08 - 0.031 = 0.049 = 4.9%
```

Interpretation: Of IONQ's 8% gain that day, approximately 3.1% was explained by the overall market being up (since IONQ is a high-beta stock that amplifies market moves). The remaining 4.9% is the **abnormal return** attributable to the company-specific news.

### Step 4: Cumulative Abnormal Returns (CAR)

A single day often doesn't capture the full market reaction. Information diffuses gradually, especially for technically complex announcements that analysts need time to digest. The Cumulative Abnormal Return sums the daily abnormal returns over a window:

```
CAR_i(t1, t2) = AR_i,t1 + AR_i,t1+1 + ... + AR_i,t2
```

Common windows used in academic research:
- **CAR(0, 0)**: Same-day reaction only
- **CAR(0, +1)**: Event day plus next day (captures overnight reaction)
- **CAR(+1, +5)**: Next 5 trading days (excludes same-day to avoid look-ahead bias if the article was published during market hours)
- **CAR(+1, +20)**: Next month of trading (captures slow diffusion)

For our evaluation, we would primarily use **CAR(+1, +5)** as the main metric, since our model predicts a "2-5 day" time horizon for most signals. We would also compute CAR at other windows to measure signal decay.

### Step 5: Controlling for Sector-Wide Events

The basic market model controls for broad market movements (S&P 500), but it doesn't control for sector-specific events. If Jensen Huang says "quantum computing is 20 years away" and all quantum stocks drop 15%, the market model would attribute most of that drop to each individual stock's "abnormal return" even though it wasn't company-specific news.

**Solution: A Two-Factor Model**

```
R_i,t = alpha_i + beta_market * R_market,t + beta_sector * R_sector,t + epsilon_i,t
```

Where `R_sector,t` is the return of a quantum computing sector basket (equal-weighted average of IONQ, RGTI, QBTS, IBM quantum division proxy, etc.) excluding the stock being evaluated.

This isolates the truly company-specific signal. If the whole quantum sector drops 15% and IONQ drops 20%, the IONQ-specific abnormal return is approximately -5% (the extra drop beyond what the sector did).

### Practical Considerations

**Thin trading and volatility**: Quantum computing stocks (especially IONQ, RGTI, QBTS) are highly volatile with wide bid-ask spreads. This means abnormal returns will be noisy. A single evaluation will have high variance. We need many events (50+) to get statistically meaningful results.

**Overlapping events**: If two news articles about IONQ are published within 3 days of each other, their event windows overlap and we cannot cleanly attribute the price movement to either one. Standard practice is to either exclude overlapping events or use shorter windows.

**Publication timing**: If an article is published at 2pm ET (during market hours), some of the reaction may already be in the same-day close. If published at 8pm ET, the reaction starts the next trading day. We need to account for this.

## Part 2: The Information Coefficient (IC)

### What It Measures

The Information Coefficient answers: "Across all of our predictions, how well do our predicted rankings correspond to the actual outcome rankings?" It's not asking "did we get the direction right?" (that's accuracy). It's asking "when we said something would move MORE, did it actually move MORE than the things we said would move LESS?"

This is a more demanding test than simple directional accuracy. A model could be 70% accurate on direction but have an IC of 0.02 (nearly useless) if it can't distinguish between a +1% move and a +15% move.

### The Formula

```
IC = SpearmanRankCorrelation(predicted_scores, realized_abnormal_returns)
```

**Spearman Rank Correlation** works by:
1. Rank all predicted scores from lowest to highest (e.g., strongly_bearish = 1, bearish = 2, ..., strongly_bullish = 5)
2. Rank all realized abnormal returns from lowest to highest
3. Compute the Pearson correlation between the two rank vectors

The formula for Spearman's rho:

```
rho = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
```

Where `d_i` is the difference between the predicted rank and the realized rank for observation *i*, and *n* is the number of observations.

### Interpreting IC Values

| IC Value | Interpretation | Practical Implication |
|----------|---------------|---------------------|
| < 0.02 | No signal | Model is guessing. Not useful. |
| 0.02 - 0.05 | Weak signal | Marginally useful. Might add value in a large portfolio with many bets. |
| 0.05 - 0.10 | Moderate signal | Genuinely useful. Most successful quant signals fall in this range. |
| 0.10 - 0.15 | Strong signal | Exceptional. Very few signals sustain this level. |
| > 0.15 | Extremely strong | Suspicious. Likely overfitting, look-ahead bias, or survivorship bias. |

Context: A signal with IC = 0.05 applied across 100 stocks with daily rebalancing can generate a Sharpe ratio above 2.0 in a well-constructed portfolio. Even seemingly small IC values translate into significant economic value at scale.

### Why Rank Correlation (Not Pearson)?

Spearman rank correlation is preferred over Pearson correlation because:

1. It's robust to outliers. A single stock that moves +50% on a day won't dominate the metric.
2. It doesn't assume a linear relationship between predicted score and realized return.
3. It matches how portfolio construction actually works: you overweight the top-ranked signals and underweight the bottom-ranked ones. You care about the ranking being correct, not the exact magnitude.

### Computing IC in Practice

For our evaluation, we would:

1. Collect all articles where the model produced a signal (say 200 articles over 2 years)
2. For each article, record the model's predicted score (map sentiment to a numeric scale)
3. For each article, compute the realized CAR over the relevant window (e.g., +1 to +5 days)
4. Compute the Spearman rank correlation between the 200 predicted scores and the 200 realized CARs

We can also compute IC on subsets: "What's the IC for news articles only?", "What's the IC for arXiv papers only?", "What's the IC for IONQ specifically?" This tells us where the model is strongest and weakest.

## Part 3: Signal Decay Analysis

### What It Measures

Signal decay answers: "How quickly does the market incorporate the information that our model identified?" This is crucial for a trading system because it determines the **optimal holding period**.

If a signal decays fast (fully priced in within 1 day), you need to trade immediately upon receiving the signal. If it decays slowly (takes 2 weeks to be fully priced in), you have more time to enter the position and can hold it longer.

### How to Measure It

Compute the IC at multiple time horizons:

```
IC(t+1)  = SpearmanRankCorrelation(predicted_scores, CAR(+1, +1))
IC(t+2)  = SpearmanRankCorrelation(predicted_scores, CAR(+1, +2))
IC(t+5)  = SpearmanRankCorrelation(predicted_scores, CAR(+1, +5))
IC(t+10) = SpearmanRankCorrelation(predicted_scores, CAR(+1, +10))
IC(t+20) = SpearmanRankCorrelation(predicted_scores, CAR(+1, +20))
IC(t+60) = SpearmanRankCorrelation(predicted_scores, CAR(+1, +60))
```

Then plot IC as a function of the holding period. The shape of this curve tells you everything:

**Fast decay pattern** (typical for earnings announcements, analyst ratings):
```
IC(t+1) = 0.08
IC(t+2) = 0.06
IC(t+5) = 0.03
IC(t+10) = 0.01
IC(t+20) = 0.00
```
The signal is strongest on day 1 and essentially gone by day 10. You need to trade immediately.

**Slow decay pattern** (typical for technical breakthroughs, academic publications):
```
IC(t+1) = 0.03
IC(t+2) = 0.04
IC(t+5) = 0.06
IC(t+10) = 0.07
IC(t+20) = 0.05
IC(t+60) = 0.02
```
The signal actually *increases* over the first week as analysts gradually digest the information, then slowly fades. This is the "gradual information diffusion" pattern that the Lopez-Lira paper identified for complex, technical news.

**No decay (momentum) pattern**:
```
IC(t+1) = 0.05
IC(t+5) = 0.05
IC(t+20) = 0.05
IC(t+60) = 0.04
```
The signal persists indefinitely. This suggests the market is systematically underreacting to this type of information.

### Why This Matters for Our Project

The quantum computing sector is highly technical. Most market participants (retail investors, generalist analysts) do not understand the difference between physical qubits and logical qubits, or why error correction below threshold is a bigger deal than a raw qubit count increase. Our hypothesis is that NLP signals derived from technical quantum computing content will exhibit **slow decay** because the market is slow to incorporate information it doesn't fully understand.

If our evaluation confirms slow decay for arXiv-sourced signals but fast decay for news headlines, that validates our entire thesis: the model adds the most value precisely where human analysts are slowest to react (highly technical content).

## Part 4: Statistical Significance

### The Problem of Small Samples

With 200 articles over 2 years, we have a relatively small sample. An IC of 0.05 computed from 200 observations might be statistically significant, or it might be noise. We need to test this.

### T-Test for IC Significance

The standard test for whether an IC is significantly different from zero:

```
t = IC * sqrt(n - 2) / sqrt(1 - IC^2)
```

Where *n* is the number of observations. This follows a t-distribution with n-2 degrees of freedom.

For IC = 0.05 with n = 200:
```
t = 0.05 * sqrt(198) / sqrt(1 - 0.0025)
t = 0.05 * 14.07 / 0.9987
t = 0.704
```

A t-value of 0.704 corresponds to a p-value of approximately 0.24, which is NOT statistically significant at the 5% level. We would need either a higher IC or more observations.

For IC = 0.10 with n = 200:
```
t = 0.10 * 14.07 / 0.995 = 1.41
```
Still not significant at 5% (p ≈ 0.08).

For IC = 0.15 with n = 200:
```
t = 0.15 * 14.07 / 0.989 = 2.13
```
This IS significant at 5% (p ≈ 0.017).

### Implications for Our Evaluation

With 200 articles, we need an IC of approximately 0.14 or higher to achieve statistical significance at the 5% level. This is a high bar. Options to improve statistical power:

1. **More observations**: If we can generate signals for 500+ articles, we only need IC > 0.09 for significance.
2. **Daily IC**: Instead of one IC across all events, compute daily IC (across all stocks with signals on a given day) and then average. This gives more observations.
3. **Bootstrap confidence intervals**: Resample the data with replacement 1000 times, compute IC each time, and report the 95% confidence interval.

## Part 5: Practical Implementation Considerations

### Data Requirements

| Data | Source | Cost | Granularity |
|------|--------|------|-------------|
| Stock prices (IONQ, RGTI, QBTS, etc.) | Yahoo Finance (`yfinance`) | Free | Daily OHLCV |
| S&P 500 index | Yahoo Finance (`^GSPC`) | Free | Daily |
| QQQ / NASDAQ | Yahoo Finance (`QQQ`) | Free | Daily |
| Article dates | Our dataset | Already have | Per-article |
| Model predictions | Our model | Already have | Per-article |

### Potential Pitfalls

**Look-ahead bias**: If an article was published at 3pm and we use that day's closing price as the "pre-event" price, we're including 1 hour of post-event trading in our baseline. Solution: use the previous day's close as the baseline, or use intraday data.

**Survivorship bias**: We're only looking at companies that still exist and are publicly traded. Companies that went bankrupt or were acquired are excluded. For our 2-year window, this is less of a concern since no major quantum companies have failed in this period.

**Multiple testing**: If we compute IC for 10 different subsets (by source type, by ticker, by event type), some will appear significant by chance. We should apply a Bonferroni correction or report all results transparently.

**Non-stationarity**: Market regimes change. An IC computed over 2024-2025 might not hold in 2026. The quantum sector has gone through dramatic sentiment shifts (the "quantum winter" scare after Jensen Huang's comments, followed by a recovery). Our evaluation should report IC by sub-period.

### What "Good" Looks Like for a Hackathon

For a hackathon submission, we don't need to prove that our model can run a profitable hedge fund. We need to demonstrate:

1. The IC is positive (the model has some predictive power, even if not statistically significant with our sample size)
2. The signal decay pattern matches our hypothesis (slow decay for technical content, fast decay for news)
3. The fine-tuned model has higher IC than the base model (fine-tuning added value)
4. The cross-asset signals are directionally correct (when we predict bearish for competitors, they actually underperform)

Even an IC of 0.03-0.05 with the right decay pattern would be a compelling demonstration for judges, especially if we can show it improves after fine-tuning.

## Summary

The evaluation framework consists of three layers:

1. **Abnormal Returns** isolate the company-specific price impact from market and sector noise
2. **Information Coefficient** quantifies the overall predictive power of the model's signals
3. **Signal Decay** reveals how quickly the market incorporates the information, validating our thesis about slow diffusion of technical content

Together, these provide a rigorous, quantitative answer to the question: "Is this model actually useful for trading, or is it just generating plausible-sounding analysis?"
