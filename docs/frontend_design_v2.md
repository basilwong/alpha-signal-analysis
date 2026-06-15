# Frontend Design V2: Alpha Signal Analysis Platform

## Design Philosophy

The interface follows an **information hierarchy** principle: simplest, highest-impact metrics are presented first, with progressive disclosure of more sophisticated breakdowns as the user scrolls down or expands sections. Every non-obvious metric includes a small "i" (info) tooltip explaining what the user is seeing.

## Tab Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│  [ Signal Explorer ]  [ Evaluation Dashboard ]  [ Sector Map ]      │
└─────────────────────────────────────────────────────────────────────┘
```

## Tab 1: Signal Explorer

This is the primary view. It serves two modes:

**Mode A: Historical Browse** — Navigate through pre-computed events from our dataset, see model predictions alongside actual price outcomes.

**Mode B: Live Analysis** — Paste a URL or raw text, get the same signal view in real-time (with price data cut off at the last trading day).

### Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL SELECTOR                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  [ Base Qwen3-8B ▼ ] [ LoRA Fine-tuned ▼ ] [ DoRA Fine-tuned ▼ ]   │   │
│  │  Toggle between published models on HuggingFace                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  INPUT SECTION                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  [ Browse Historical Events ◀ Event 47/611 ▶ ]  OR  [ Paste New ]   │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Article: "IonQ Achieves 35 Algorithmic Qubits..."           │   │   │
│  │  │  Source: news | Date: 2026-05-15 | Ticker: IONQ              │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│─────────────────────────────────────────────────────────────────────────────│
│                                                                             │
│  LEVEL 1: SIGNAL VECTOR (Hero Element)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ℹ️ "Cross-sectional signal: predicted price impact across all      │   │
│  │     quantum tickers from this single event"                         │   │
│  │                                                                     │   │
│  │  IONQ  ████████████████████  +2.0  (strongly bullish)              │   │
│  │  HON   ████████████         +1.2  (bullish, Quantinuum)            │   │
│  │  NVDA  ███                  +0.3  (slight, HPC demand)             │   │
│  │  MSFT  ▏                     0.0  (neutral, topological)           │   │
│  │  QBTS  ▎▎▎                  -0.3  (slight bearish)                 │   │
│  │  IBM   ▎▎▎▎                 -0.4  (bearish, superconducting)       │   │
│  │  RGTI  ▎▎▎▎▎▎▎             -0.7  (bearish, competitive gap)       │   │
│  │  GOOGL ▎                    -0.1  (negligible, <0.1% revenue)      │   │
│  │                                                                     │   │
│  │  Time Horizon: 2-5 days | Decay: slow | Novelty: high              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│─────────────────────────────────────────────────────────────────────────────│
│                                                                             │
│  LEVEL 2: PREDICTED vs ACTUAL (Time Series Overlay)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ℹ️ "Compares the model's predicted signal decay against actual     │   │
│  │     abnormal returns (market-adjusted) over the following days"     │   │
│  │                                                                     │   │
│  │  [Chart: X-axis = days after event (0 to 20)]                      │   │
│  │  [Y-axis = cumulative abnormal return (%)]                         │   │
│  │                                                                     │   │
│  │  Lines:                                                             │   │
│  │  ── IONQ predicted (dashed, from model's pct_range)                │   │
│  │  ── IONQ actual (solid, from market data)                          │   │
│  │  ── RGTI predicted (dashed)                                        │   │
│  │  ── RGTI actual (solid)                                            │   │
│  │  (toggleable per ticker)                                           │   │
│  │                                                                     │   │
│  │  Shaded region: model's predicted range [low%, high%]              │   │
│  │  Vertical line: model's predicted time horizon                     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│─────────────────────────────────────────────────────────────────────────────│
│                                                                             │
│  LEVEL 3: EVENT METRICS                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ℹ️ "Performance metrics for this specific event"                   │   │
│  │                                                                     │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐│   │
│  │  │Direction     │ │Magnitude     │ │Decay Match   │ │Cross-Asset ││   │
│  │  │Accuracy      │ │Accuracy      │ │              │ │Accuracy    ││   │
│  │  │              │ │              │ │              │ │            ││   │
│  │  │  7/9 ✓       │ │ 5/9 within   │ │ Predicted:   │ │ 6/8 ✓      ││   │
│  │  │  tickers     │ │ predicted    │ │ slow         │ │ secondary  ││   │
│  │  │  correct     │ │ range        │ │ Actual: slow │ │ correct    ││   │
│  │  │  direction   │ │              │ │ ✓ Match      │ │            ││   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └────────────┘│   │
│  │                                                                     │   │
│  │  Forward Returns Table (from alphalens):                            │   │
│  │  ┌────────┬────────┬────────┬────────┬────────┐                   │   │
│  │  │ Ticker │ +1 day │ +5 day │ +10 day│ +20 day│                   │   │
│  │  ├────────┼────────┼────────┼────────┼────────┤                   │   │
│  │  │ IONQ   │ +2.1%  │ +6.8%  │ +8.2%  │ +5.1%  │                   │   │
│  │  │ RGTI   │ -0.5%  │ -3.2%  │ -4.1%  │ -2.8%  │                   │   │
│  │  │ ...    │ ...    │ ...    │ ...    │ ...    │                   │   │
│  │  └────────┴────────┴────────┴────────┴────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│─────────────────────────────────────────────────────────────────────────────│
│                                                                             │
│  LEVEL 4: MODEL REASONING (Expandable)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ▶ Show Model Reasoning Trace                                       │   │
│  │                                                                     │   │
│  │  (When expanded:)                                                   │   │
│  │  "This article describes a trapped-ion qubit milestone. IonQ is     │   │
│  │   the primary beneficiary because they are the leading trapped-ion  │   │
│  │   company. Quantinuum (HON) also benefits as a trapped-ion player.  │   │
│  │   Rigetti and IBM use superconducting qubits, so a trapped-ion      │   │
│  │   advance widens the competitive gap against them. Google is        │   │
│  │   minimally affected because quantum computing represents less      │   │
│  │   than 0.1% of their revenue. Signal decay is slow because most     │   │
│  │   analysts cannot assess the technical significance of algorithmic  │   │
│  │   qubit counts without domain expertise..."                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│─────────────────────────────────────────────────────────────────────────────│
│                                                                             │
│  LEVEL 5: RAW JSON (Expandable)                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ▶ Show Raw API Response                                            │   │
│  │  (Full JSON signal + metadata + latency)                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Navigation Behavior

The event navigator at the top uses a horizontal slider with event markers:

```
┌─────────────────────────────────────────────────────────────────────┐
│  ◀  │  2024-08 ──●──●─────●──── 2025-03 ──●●●──── 2026-06  │  ▶  │
│     │            ↑                          ↑                │     │
│     │         Event 3                    Event 47            │     │
│     │         (current)                                      │     │
└─────────────────────────────────────────────────────────────────────┘
│  Showing: Event 3 of 611 | "IonQ Achieves 35 Algorithmic Qubits"   │
│  Filter: [All Sources ▼] [All Tickers ▼] [All Event Types ▼]      │
└─────────────────────────────────────────────────────────────────────┘
```

Users can also filter events by source type, ticker, or event type to find specific categories.

### Live Analysis Mode

When the user clicks "Paste New," the input section expands to a text area + URL field. After analysis:
- The signal vector renders immediately
- The "Predicted vs Actual" chart shows the prediction envelope (dashed lines + shaded range) but actual price data only up to the last available trading day
- If the event is recent (within 20 days), the chart shows partial actual data with a "still in progress" indicator
- Event metrics show "N/A" for metrics that require future data

### Model Selector

A dropdown at the top allows toggling between published models:
- Base Qwen3-8B (no fine-tuning, for comparison)
- LoRA Fine-tuned Qwen3-8B (current model)
- DoRA Fine-tuned Qwen3-8B (future experiment)
- Qwen3-32B LoRA (future, final version)

When toggling, the signal vector and metrics update to show the selected model's output. In historical mode, pre-computed results for each model are loaded. In live mode, inference runs against the selected model.

## Tab 2: Evaluation Dashboard

Aggregate performance metrics across the full historical dataset.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  SUMMARY METRICS (Top Row)                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ Overall  │ │Direction │ │ Optimal  │ │ Events   │ │ Avg      │        │
│  │ IC       │ │ Accuracy │ │ Horizon  │ │ Evaluated│ │ Latency  │        │
│  │          │ │          │ │          │ │          │ │          │        │
│  │ 0.07     │ │ 68%      │ │ 5 days   │ │ 411      │ │ 4.2s     │        │
│  │ (p=0.03) │ │ (612/900)│ │          │ │          │ │          │        │
│  │ ℹ️        │ │ ℹ️        │ │ ℹ️        │ │ ℹ️        │ │ ℹ️        │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
│                                                                             │
│─────────────────────────────────────────────────────────────────────────────│
│                                                                             │
│  SIGNAL DECAY CURVE                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ℹ️ "Shows how predictive power changes over different holding       │   │
│  │     periods. Peak IC indicates the optimal time to hold a position" │   │
│  │                                                                     │   │
│  │  [Line chart: IC on Y-axis, Holding Period (days) on X-axis]       │   │
│  │  Lines: Overall, News only, arXiv only, Press Release only         │   │
│  │  Shaded: 95% confidence interval (bootstrap)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  IC BY SUBSET                                                               │
│  ┌──────────────────────────────┐ ┌──────────────────────────────────┐    │
│  │  IC by Source Type           │ │  IC by Ticker                     │    │
│  │  ℹ️                           │ │  ℹ️                               │    │
│  │  [Horizontal bar chart]      │ │  [Horizontal bar chart]           │    │
│  │  arxiv:  ████████ 0.12      │ │  IONQ: ████████ 0.09             │    │
│  │  news:   █████ 0.06         │ │  RGTI: ██████ 0.07               │    │
│  │  sec:    ████ 0.05          │ │  QBTS: ████ 0.04                 │    │
│  │  social: ██ 0.02            │ │  IBM:  ██ 0.02                   │    │
│  │                              │ │                                   │    │
│  │  * = Bonferroni significant  │ │  * = Bonferroni significant       │    │
│  └──────────────────────────────┘ └──────────────────────────────────┘    │
│                                                                             │
│  PREDICTED vs REALIZED (Scatter)                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ℹ️ "Each dot is one event. X = model's predicted score,            │   │
│  │     Y = actual abnormal return. Positive correlation = signal works"│   │
│  │                                                                     │   │
│  │  [Scatter plot with regression line]                                │   │
│  │  Color: by source type                                              │   │
│  │  Annotation: Spearman rho = 0.07, p = 0.03                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  MODEL COMPARISON (if multiple models evaluated)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ℹ️ "Compares fine-tuned model against base model to quantify       │   │
│  │     the value added by domain-specific training"                    │   │
│  │                                                                     │   │
│  │  [Grouped bar chart: IC, Direction Accuracy, Magnitude Accuracy]   │   │
│  │  Bars: Base Qwen3-8B vs LoRA Fine-tuned vs DoRA Fine-tuned        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LIMITATIONS (Collapsible, always visible)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ▶ View Limitations & Methodology Notes                             │   │
│  │                                                                     │   │
│  │  • Sample size: 411 events (statistical power limited)              │   │
│  │  • Period: Aug 2024 - Jun 2026 (single market regime)               │   │
│  │  • No intraday timing (daily granularity only)                      │   │
│  │  • Bonferroni correction applied to subset tests                    │   │
│  │  • Abnormal returns use single-factor market model (SPY)            │   │
│  │  • Quantum sector basket: equal-weighted IONQ+RGTI+QBTS            │   │
│  │  • Revenue exposure weights are static estimates                    │   │
│  │  • Correlation ≠ causation                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tab 3: Sector Map

A structural visualization of the quantum computing competitive landscape.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  TECHNOLOGY CLUSTERS                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ℹ️ "Companies grouped by their primary quantum computing approach. │   │
│  │     Lines show competitive relationships. Click a cluster to see    │   │
│  │     how signals propagate."                                         │   │
│  │                                                                     │   │
│  │         ┌─────────────────┐                                        │   │
│  │         │  TRAPPED ION    │                                        │   │
│  │         │  ┌────┐ ┌────┐ │                                        │   │
│  │         │  │IONQ│ │HON │ │                                        │   │
│  │         │  │100%│ │~5% │ │                                        │   │
│  │         │  └────┘ └────┘ │                                        │   │
│  │         └────────┬────────┘                                        │   │
│  │                  │ competes                                         │   │
│  │         ┌────────┴────────┐                                        │   │
│  │         │ SUPERCONDUCTING │                                        │   │
│  │         │ ┌────┐┌────┐┌─────┐                                     │   │
│  │         │ │RGTI││IBM ││GOOGL│                                     │   │
│  │         │ │100%││~2% ││<.1%│                                      │   │
│  │         │ └────┘└────┘└─────┘                                     │   │
│  │         └────────┬────────┘                                        │   │
│  │                  │                                                  │   │
│  │    ┌─────────────┼─────────────┐                                   │   │
│  │    │             │             │                                    │   │
│  │  ┌─┴───────┐ ┌──┴─────┐ ┌───┴──────┐                            │   │
│  │  │ANNEALING│ │TOPOLOG.│ │NEUTRAL AT│                              │   │
│  │  │ ┌────┐ │ │ ┌────┐ │ │ ┌────┐   │                              │   │
│  │  │ │QBTS│ │ │ │MSFT│ │ │ │INFQ│   │                              │   │
│  │  │ │100%│ │ │ │<.1%│ │ │ │100%│   │                              │   │
│  │  │ └────┘ │ │ └────┘ │ │ └────┘   │                              │   │
│  │  └────────┘ └────────┘ └──────────┘                               │   │
│  │                                                                     │   │
│  │  ┌──────────────────────┐                                          │   │
│  │  │ ADJACENT / ENABLERS  │                                          │   │
│  │  │ ┌────┐               │                                          │   │
│  │  │ │NVDA│ (simulation)  │                                          │   │
│  │  │ │~1% │               │                                          │   │
│  │  │ └────┘               │                                          │   │
│  │  └──────────────────────┘                                          │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SIGNAL PROPAGATION SIMULATOR                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ℹ️ "Select an event type to see how signals propagate across the   │   │
│  │     sector. Node color = signal direction. Size = magnitude."       │   │
│  │                                                                     │   │
│  │  Event Type: [ Trapped-ion breakthrough ▼ ]                        │   │
│  │                                                                     │   │
│  │  Result: IONQ (🟢 +2.0), HON (🟢 +1.2), RGTI (🔴 -0.7),          │   │
│  │          IBM (🔴 -0.4), QBTS (🔴 -0.3), NVDA (🟢 +0.3),          │   │
│  │          GOOGL (⚪ -0.01), MSFT (⚪ 0.0)                           │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  COMPANY PROFILES (Expandable cards)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ▶ IONQ | Trapped Ion | 100% quantum revenue | Market cap: $X.XB  │   │
│  │  ▶ RGTI | Superconducting | 100% quantum | Market cap: $X.XB      │   │
│  │  ▶ QBTS | Quantum Annealing | 100% quantum | Market cap: $X.XB    │   │
│  │  ▶ IBM  | Superconducting | ~2% quantum | Market cap: $XXXB       │   │
│  │  ...                                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Info Tooltips (ℹ️ Buttons)

Every metric and chart includes a tooltip. Examples:

| Element | Tooltip Text |
|---------|-------------|
| IC value | "Information Coefficient: Spearman rank correlation between model predictions and actual returns. Range: -1 to +1. Values above 0.05 indicate meaningful predictive power." |
| Direction Accuracy | "Percentage of events where the model correctly predicted whether the stock would go up or down (after adjusting for market movements)." |
| Signal Decay | "How quickly the market incorporates the information. 'Fast' means priced in same day. 'Slow' means the market takes days/weeks to fully react." |
| Abnormal Return | "Stock return minus what we'd expect based on overall market movements. Isolates the company-specific reaction to the news event." |
| Bonferroni correction | "Statistical adjustment for testing multiple hypotheses simultaneously. Prevents false positives when evaluating many subgroups." |
| Revenue Exposure | "Percentage of the company's total revenue derived from quantum computing. Higher exposure means quantum news has larger impact on stock price." |

## Implementation Technology

| Component | Technology | Reasoning |
|-----------|-----------|-----------|
| Framework | Gradio Blocks | Required by hackathon, supports tabs and custom layouts |
| Charts | Plotly (interactive) | Hover tooltips, zoom, pan. Embeds natively in Gradio |
| Sector Map | Plotly network graph or D3.js embed | Interactive node-link diagram |
| Data | Pre-computed JSON files | Historical results loaded from files, not computed on-the-fly |
| Live inference | @spaces.GPU | On-demand GPU for new article analysis |

## Data Dependencies

For this frontend to work, we need:

1. **Pre-computed predictions** for all 611 articles (signal vectors for all tickers)
2. **Historical price data** for all tickers (from yfinance)
3. **Pre-computed abnormal returns and CARs** for all events
4. **Pre-computed IC metrics and decay curves** (from alphalens)
5. **Static sector map data** (company profiles, technology clusters, revenue exposures)

All of this can be pre-computed once and stored as JSON/Parquet files that the Gradio app loads at startup. Only the "Live Analysis" mode requires real-time inference.
