"""
Iterative Memory Loop with IC Evaluation.

Processes articles in batches of 50. After each batch:
1. Runs the feedback loop (records outcomes, generates rules)
2. Computes IC for that batch
3. Compares IC across batches to measure improvement

The hypothesis: IC should improve in later batches because the agent
has accumulated more memory and learned behavioral rules from earlier mistakes.
"""
import json
import os
import sys
import time
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from datetime import datetime

sys.path.insert(0, '.')

from openai import OpenAI
from agent.memory import MemoryStore
from agent.memory_loop import (
    EpisodicMemory, ProceduralMemory, FeedbackLoop, EnhancedRetriever, Episode
)
from agent.seed_data import SEED_FACTS
from agent.config import QUANTUM_TICKERS, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

# Config
MODEL = "qwen-plus-2025-07-28"
BATCH_SIZE = 50
MAX_ARTICLES = 200  # Process 200 articles total (4 batches of 50)
RATE_LIMIT = 2  # seconds between API calls

# DashScope client
client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

# Setup fresh memory
DB_PATH = "data/iterative_memory.db"
OUTPUT_FILE = "data/eval/predictions_iterative_memory.jsonl"
RESULTS_FILE = "data/eval/results_iterative_memory.json"

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

memory = MemoryStore(DB_PATH)
episodic = EpisodicMemory(memory)
procedural = ProceduralMemory(memory)
retriever = EnhancedRetriever(memory)
feedback = FeedbackLoop(memory, llm_client=client, model_name=MODEL)

# Seed baseline knowledge
for fact in SEED_FACTS:
    memory.store_knowledge(fact['ticker'], fact['type'], fact['content'], 'seed')

# Load market data
def load_market_data():
    market = {}
    for ticker in QUANTUM_TICKERS + ["SPY"]:
        path = f"data/market/{ticker}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            if close_col in df.columns:
                series = df[close_col].dropna()
                market[ticker] = {
                    "dates": [str(d.date()) for d in series.index],
                    "values": [float(v) for v in series.values]
                }
    return market

def get_forward_return(market, ticker, event_date, horizon=5):
    if ticker not in market:
        return None
    dates = market[ticker]["dates"]
    values = market[ticker]["values"]
    try:
        start_idx = next(i for i, d in enumerate(dates) if d >= event_date)
    except StopIteration:
        return None
    end_idx = min(start_idx + horizon, len(values) - 1)
    if end_idx <= start_idx or values[start_idx] == 0:
        return None
    return (values[end_idx] - values[start_idx]) / values[start_idx]

def compute_batch_ic(predictions, market, horizon=5):
    """Compute IC for a batch of predictions."""
    scores = []
    returns = []
    for pred in predictions:
        if pred.get("status") != "success":
            continue
        date = pred.get("date", "")
        sv = pred.get("signal_vector_clean", {})
        if not sv or not date:
            continue
        for ticker in QUANTUM_TICKERS[:5]:
            score = sv.get(ticker, 0)
            if abs(score) < 0.3:
                continue
            ret = get_forward_return(market, ticker, date, horizon)
            if ret is not None:
                scores.append(score)
                returns.append(ret)
    
    if len(scores) >= 10:
        ic, p_value = spearmanr(scores, returns)
        return {"ic": round(ic, 4), "p_value": round(p_value, 4), "n": len(scores)}
    return {"ic": None, "n": len(scores), "note": "insufficient data"}

def process_article(article, i, total):
    """Process a single article through the memory-augmented agent."""
    title = article.get('title', 'Untitled')[:50]
    date = article.get('date', '')
    source = article.get('source', 'news')
    
    context = retriever.build_full_context(article.get('text', ''), source_type=source)
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"""You are a quantitative signal generator for quantum computing stocks with persistent memory.

{context}

IMPORTANT: Follow your learned behavioral rules. Reference past experiences.
Generate a signal vector for: IONQ, RGTI, QBTS, QUBT, QNT, IBM, GOOGL, MSFT, HON, NVDA.
Score range: -2.0 to +2.0. GOOGL/MSFT/NVDA always 0.0.
Include chain_of_thought.
Output ONLY valid JSON with signal_vector (dict format: {{"IONQ": 1.5, "RGTI": -0.3, ...}}) and chain_of_thought."""},
                {"role": "user", "content": f"Analyze this {source} article ({date}):\n\n{article.get('text', '')}"}
            ],
            temperature=0.3,
            max_tokens=1200,
            extra_body={"enable_thinking": False}
        )
        
        elapsed = time.time() - start
        content = response.choices[0].message.content or ""
        
        # Parse signal
        s = content.find('{')
        e = content.rfind('}') + 1
        if s != -1:
            signal = json.loads(content[s:e])
        else:
            return None, elapsed
        
        sv = signal.get('signal_vector', signal)
        if isinstance(sv, list):
            sv = {item['ticker']: item.get('score', 0) for item in sv if isinstance(item, dict) and 'ticker' in item}
        
        cot = signal.get('chain_of_thought', '')
        
        # Clean scores (handle string values)
        sv_clean = {}
        for ticker, val in sv.items():
            if isinstance(val, (int, float)):
                sv_clean[ticker] = float(val)
            elif isinstance(val, dict):
                sv_clean[ticker] = float(val.get('score', 0))
            elif isinstance(val, str):
                try:
                    sv_clean[ticker] = float(val)
                except:
                    sv_clean[ticker] = 0.0
        
        # Record episodes with real outcomes
        for ticker in QUANTUM_TICKERS[:5]:
            score = sv_clean.get(ticker, 0)
            if abs(score) < 0.3:
                continue
            actual_ret = get_forward_return(market, ticker, date, horizon=5)
            if actual_ret is not None:
                predicted_dir = "bullish" if score > 0 else "bearish"
                actual_dir = "bullish" if actual_ret > 0 else "bearish"
                ep = Episode(
                    date=date, ticker=ticker, predicted_score=score,
                    predicted_direction=predicted_dir,
                    actual_return_5d=actual_ret, actual_direction=actual_dir,
                    was_correct=(predicted_dir == actual_dir),
                    article_title=title, source_type=source,
                    reasoning_summary=cot[:150]
                )
                episodic.store_episode(ep)
        
        # Store knowledge
        if cot and len(cot) > 30:
            for ticker in QUANTUM_TICKERS[:5]:
                score = sv_clean.get(ticker, 0)
                if abs(score) >= 0.5:
                    memory.store_knowledge(
                        ticker=ticker, fact_type="signal_context",
                        content=f"[{date}] Score {score:+.1f}: {cot[:100]}",
                        source=source, ttl_days=60
                    )
        
        pred = {
            "status": "success",
            "article_idx": i,
            "date": date,
            "title": article.get('title', ''),
            "source": source,
            "signal": signal,
            "signal_vector_clean": sv_clean,
            "time_seconds": elapsed,
            "memory_facts": memory.get_memory_stats()['knowledge_facts'],
            "rules_active": len(procedural.get_active_rules()),
        }
        return pred, elapsed
        
    except Exception as ex:
        elapsed = time.time() - start
        return {"status": "error", "article_idx": i, "date": date, "error": str(ex)[:100]}, elapsed


# ============================================================
# MAIN LOOP
# ============================================================

# Load data
with open('data/raw/articles_eval.jsonl') as f:
    articles = [json.loads(l) for l in f if l.strip()]
articles = articles[:MAX_ARTICLES]

market = load_market_data()
print(f"Loaded {len(market)} tickers, {len(articles)} articles")

batch_results = []
all_predictions = []
out_f = open(OUTPUT_FILE, 'w')

num_batches = (len(articles) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"\n{'='*70}")
print(f"ITERATIVE MEMORY LOOP: {len(articles)} articles in {num_batches} batches of {BATCH_SIZE}")
print(f"{'='*70}")

for batch_idx in range(num_batches):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, len(articles))
    batch_articles = articles[batch_start:batch_end]
    
    print(f"\n{'='*70}")
    print(f"BATCH {batch_idx + 1}/{num_batches} (articles {batch_start}-{batch_end-1})")
    print(f"Memory: {memory.get_memory_stats()['knowledge_facts']} facts | Rules: {len(procedural.get_active_rules())} | Episodes: {episodic.get_accuracy_by_category()['overall']['total']}")
    print(f"{'='*70}")
    
    batch_preds = []
    batch_errors = 0
    
    for j, article in enumerate(batch_articles):
        global_idx = batch_start + j
        pred, elapsed = process_article(article, global_idx, len(articles))
        
        if pred and pred.get("status") == "success":
            batch_preds.append(pred)
            all_predictions.append(pred)
            out_f.write(json.dumps(pred) + '\n')
            out_f.flush()
        else:
            batch_errors += 1
            if pred:
                out_f.write(json.dumps(pred) + '\n')
                out_f.flush()
        
        # Progress every 10
        if (j + 1) % 10 == 0:
            stats = episodic.get_accuracy_by_category()
            acc = stats['overall']['accuracy'] * 100 if stats['overall']['total'] > 0 else 0
            print(f"  [{global_idx+1}/{len(articles)}] success={len(batch_preds)} err={batch_errors} | acc={acc:.0f}% | mem={memory.get_memory_stats()['knowledge_facts']}")
        
        time.sleep(RATE_LIMIT)
    
    # ---- END OF BATCH: Run feedback loop ----
    print(f"\n  --- Batch {batch_idx+1} complete: {len(batch_preds)} predictions, {batch_errors} errors ---")
    
    # Compute IC for this batch
    batch_ic = compute_batch_ic(batch_preds, market, horizon=5)
    print(f"  Batch IC @5d: {batch_ic}")
    
    # Run feedback loop
    rules = feedback.analyze_and_generate_rules()
    print(f"  Feedback: {len(rules)} rules generated/updated")
    
    # LLM rule generation (every other batch to save tokens)
    if batch_idx % 2 == 1:
        episodes = episodic.get_similar_episodes(limit=20)
        llm_rules = feedback.generate_advanced_rules_with_llm(episodes)
        print(f"  LLM rules: {len(llm_rules)} additional")
    
    # Store batch results
    batch_result = {
        "batch": batch_idx + 1,
        "articles_processed": len(batch_preds),
        "errors": batch_errors,
        "ic_5d": batch_ic,
        "accuracy": episodic.get_accuracy_by_category()['overall']['accuracy'],
        "memory_facts": memory.get_memory_stats()['knowledge_facts'],
        "active_rules": len(procedural.get_active_rules()),
        "total_episodes": episodic.get_accuracy_by_category()['overall']['total'],
    }
    batch_results.append(batch_result)
    
    # Print active rules
    active = procedural.get_active_rules()
    if active:
        print(f"  Active rules ({len(active)}):")
        for r in active[:3]:
            print(f"    [{r.category}] {r.rule_text[:70]}...")

out_f.close()

# ============================================================
# FINAL EVALUATION
# ============================================================

print(f"\n{'='*70}")
print("FINAL RESULTS: IC BY BATCH (measuring improvement over time)")
print(f"{'='*70}\n")

print(f"{'Batch':<8} {'IC @5d':<12} {'p-value':<10} {'N':<6} {'Accuracy':<10} {'Rules':<8} {'Memory':<8}")
print("-" * 70)
for br in batch_results:
    ic = br['ic_5d']
    ic_str = f"{ic['ic']:+.4f}" if ic.get('ic') is not None else "N/A"
    p_str = f"{ic['p_value']:.4f}" if ic.get('p_value') is not None else "N/A"
    print(f"{br['batch']:<8} {ic_str:<12} {p_str:<10} {ic['n']:<6} {br['accuracy']*100:.0f}%{'':<5} {br['active_rules']:<8} {br['memory_facts']:<8}")

# Overall IC
overall_ic = compute_batch_ic(all_predictions, market, horizon=5)
print(f"\nOverall IC @5d: {overall_ic}")

# Check if IC improved across batches
ics = [br['ic_5d']['ic'] for br in batch_results if br['ic_5d'].get('ic') is not None]
if len(ics) >= 2:
    if ics[-1] > ics[0]:
        print(f"\nIMPROVEMENT DETECTED: IC went from {ics[0]:+.4f} to {ics[-1]:+.4f} ({ics[-1]-ics[0]:+.4f})")
    else:
        print(f"\nNO IMPROVEMENT: IC went from {ics[0]:+.4f} to {ics[-1]:+.4f} ({ics[-1]-ics[0]:+.4f})")

# Save results
results = {
    "model": "Memory Agent (qwen-plus + iterative learning)",
    "total_articles": len(articles),
    "total_predictions": len(all_predictions),
    "batch_results": batch_results,
    "overall_ic_5d": overall_ic,
    "final_rules": [{"rule_id": r.rule_id, "rule_text": r.rule_text, "confidence": r.confidence, "category": r.category} for r in procedural.get_active_rules()],
    "final_accuracy": episodic.get_accuracy_by_category(),
}
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: {RESULTS_FILE}")
print(f"Predictions saved to: {OUTPUT_FILE}")
print(f"Memory DB saved to: {DB_PATH}")
