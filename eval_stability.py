import json, pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from datetime import timedelta

MARKET_DIR = Path('data/market')
QUANTUM_TICKERS = ['IONQ','RGTI','QBTS','QUBT','IBM','GOOGL','MSFT','HON','NVDA']

returns = {}
for t in QUANTUM_TICKERS + ['SPY']:
    p = MARKET_DIR / f'{t}.parquet'
    if p.exists():
        df = pd.read_parquet(p)
        c = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        returns[t] = df[c].pct_change().dropna()
returns_df = pd.DataFrame(returns)

print('7B GRPO STABILITY CHECK')
print('='*70)

for model_file, model_name in [
    ('predictions_7b_grpo_run1_temp03.jsonl', 'Run 1 (temp=0.3, seed=42)'),
    ('predictions_7b_grpo_run2_temp03.jsonl', 'Run 2 (temp=0.3, seed=123)'),
    ('predictions_7b_grpo_run3_temp01.jsonl', 'Run 3 (temp=0.1, seed=42)'),
    ('predictions_v7d_grpo_clean.jsonl', 'Original run'),
]:
    preds = []
    errs = 0
    with open(f'data/eval/{model_file}') as f:
        for line in f:
            d = json.loads(line)
            if d.get('status') == 'success': preds.append(d)
            elif d.get('status') == 'error': errs += 1

    print(f'{model_name}: {len(preds)} success, {errs} errors')
    for horizon in [5, 10, 20]:
        obs = []
        for pred in preds:
            date_str = pred.get('date','')
            sv = pred.get('signal',{}).get('signal_vector',{})
            if not date_str or not sv or not isinstance(sv, dict): continue
            try: event_date = pd.Timestamp(date_str)
            except: continue
            for ticker in QUANTUM_TICKERS:
                if ticker not in sv or ticker not in returns_df.columns: continue
                val = sv[ticker]
                if isinstance(val, dict): score = val.get('score', 0)
                elif isinstance(val, (int, float)): score = val
                else: continue
                if not isinstance(score, (int, float)) or score == 0: continue
                sr = returns_df[ticker]; mr = returns_df['SPY']
                pre = sr[sr.index < event_date]
                if len(pre) < 60: continue
                gap = event_date - timedelta(days=14)
                es = pre[pre.index < gap].tail(180); em = mr[mr.index < gap].tail(180)
                al = pd.concat([es,em],axis=1).dropna(); al.columns=['s','m']
                if len(al)<60: continue
                X = sm.add_constant(al['m'])
                try:
                    mdl = sm.OLS(al['s'],X).fit(); a,b = mdl.params.iloc[0], mdl.params.iloc[1]
                except: continue
                ps = sr[sr.index > event_date]; pm = mr[mr.index > event_date]
                if len(ps)<horizon: continue
                ws=ps.iloc[:horizon]; wm=pm.iloc[:horizon]
                idx=ws.index.intersection(wm.index)
                if len(idx)<1: continue
                car = (ws.loc[idx] - (a + b*wm.loc[idx])).sum()
                obs.append({'score': float(score), 'car': float(car)})
        if len(obs) >= 10:
            df_obs = pd.DataFrame(obs)
            sp_ic, sp_p = stats.spearmanr(df_obs['score'], df_obs['car'])
            sig = '***' if sp_p<0.01 else '**' if sp_p<0.05 else '*' if sp_p<0.1 else ''
            print(f'  +{horizon}d: IC={sp_ic:+.4f}{sig} (p={sp_p:.4f}) N={len(obs)}')
    print()
