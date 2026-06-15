# Chart Verification Results

All three price movement charts are rendering correctly:

1. **Raw Price Movement** - Shows cumulative returns for quantum tickers with SPY as a dashed benchmark line. Title: "Cumulative Return (%) | Dashed = SPY Benchmark"

2. **Abnormal Returns vs. Market (SPY)** - Shows stock return minus SPY return. Title: "Abnormal Return (%) = Stock Return − SPY Return". Has info tooltip explaining this is what IC is evaluated against.

3. **Abnormal Returns vs. Sector (Equal-Weight Quantum Basket)** - Shows stock return minus the average of pure-play quantum tickers (IONQ, RGTI, QBTS, QUBT). Title: "Abnormal Return (%) = Stock Return − Quantum Sector Avg". Has info tooltip explaining relative winners within sector.

All charts are interactive Plotly charts with proper legends, axis labels, and dark theme styling.
