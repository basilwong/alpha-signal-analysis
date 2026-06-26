"""
Real end-to-end test of the memory loop with DashScope API and market data.
Processes 20 articles chronologically, accumulates memory, runs feedback loop.
"""
import json
import os
import sys
import time
import pandas as pd
sys.path.insert(0, '.')

from openai import OpenAI
from agent.memory import MemoryStore
from agent.memory_loop import (
    EpisodicMemory, ProceduralMemory, FeedbackLoop, EnhancedRetriever
)
from agent.seed_data import SEED_FACTS
from agent.config import QUANTUM_TICKERS, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

# DashScope client
client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
MODEL = "qwen-plus"

# Setup fresh memory
DB_PATH = "data/test_real_loop.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

memory = MemoryStore(DB_PATH)
episodic = EpisodicMemory(memory)
procedural = ProceduralMemory(memory)
retriever = EnhancedRetriever(memory)
feedback = FeedbackLoop(memory, llm_client=client, model_name="qwen-plus-2025-07-28")

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

# Load articles
with open('data/raw/articles_eval.jsonl') as f:
    articles = [json.loads(l) for l in f if l.strip()]

market = load_market_data()
print(f"Loaded {len(market)} tickers of market data")
print(f"Loaded {len(articles)} evaluation articles")

# Process 20 articles with full memory loop
N = 20
print(f"\n{'='*60}")
print(f"PROCESSING {N} ARTICLES WITH FULL MEMORY LOOP")
print(f"{'='*60}\n")

predictions = []
for i, article in enumerate(articles[:N]):
    title = article.get('title', 'Untitled')[:50]
    date = article.get('date', '')
    source = article.get('source', 'news')
    
    # Step 1: Build enhanced context (all 3 memory types)
    context = retriever.build_full_context(article.get('text', ''), source_type=source)
    
    # Step 2: Generate signal with memory
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"""You are a quantitative signal generator for quantum computing stocks with persistent memory.

{context}

IMPORTANT: Reference your learned rules and past experiences in your reasoning.
Generate a signal vector for: IONQ, RGTI, QBTS, QUBT, QNT, IBM, GOOGL, MSFT, HON, NVDA.
Score range: -2.0 to +2.0. GOOGL/MSFT/NVDA always 0.0.
Include chain_of_thought that references your memory.
Output ONLY valid JSON with signal_vector and chain_of_thought."""},
                {"role": "user", "content": f"Analyze this {source} article ({date}):\n\n{article.get('text', '')}"}
            ],
            temperature=0.3,
            max_tokens=1200,
            extra_body={"enable_thinking": False}
        )
        
        elapsed = time.time() - start
        content = response.choices[0].message.content or ""
        
        # Parse signal (handle both dict and list formats)
        s = content.find('{')
        e = content.rfind('}') + 1
        if s != -1:
            signal = json.loads(content[s:e])
        else:
            signal = {}
        
        sv = signal.get('signal_vector', signal)
        # Handle list format: [{"ticker": "IONQ", "score": 1.5}, ...]
        if isinstance(sv, list):
            sv_dict = {}
            for item in sv:
                if isinstance(item, dict) and 'ticker' in item:
                    sv_dict[item['ticker']] = item.get('score', 0)
            sv = sv_dict
        cot = signal.get('chain_of_thought', '')
        
        # Step 3: Record outcome (using actual market data)
        for ticker in QUANTUM_TICKERS[:5]:  # Pure-play only
            score = sv.get(ticker, 0)
            if isinstance(score, dict):
                score = score.get('score', 0)
            if not isinstance(score, (int, float)) or abs(score) < 0.3:
                continue
            
            actual_ret = get_forward_return(market, ticker, date, horizon=5)
            if actual_ret is not None:
                from agent.memory_loop import Episode
                predicted_dir = "bullish" if score > 0 else "bearish"
                actual_dir = "bullish" if actual_ret > 0 else "bearish"
                was_correct = (predicted_dir == actual_dir)
                
                ep = Episode(
                    date=date, ticker=ticker, predicted_score=float(score),
                    predicted_direction=predicted_dir,
                    actual_return_5d=actual_ret, actual_direction=actual_dir,
                    was_correct=was_correct, article_title=title,
                    source_type=source, reasoning_summary=cot[:150]
                )
                episodic.store_episode(ep)
        
        # Step 4: Store knowledge from this analysis
        if cot and len(cot) > 30:
            for ticker in QUANTUM_TICKERS[:5]:
                score = sv.get(ticker, 0)
                if isinstance(score, dict):
                    score = score.get('score', 0)
                if isinstance(score, (int, float)) and abs(score) >= 0.5:
                    memory.store_knowledge(
                        ticker=ticker,
                        fact_type="signal_context",
                        content=f"[{date}] Score {score:+.1f}: {cot[:100]}",
                        source=source, ttl_days=60
                    )
        
        predictions.append({"date": date, "title": title, "signal": signal, "source": source})
        
        # Progress
        mem_stats = memory.get_memory_stats()
        ep_stats = episodic.get_accuracy_by_category()
        overall_acc = ep_stats.get('overall', {}).get('accuracy', 0)
        print(f"  [{i+1}/{N}] {title}... | {elapsed:.1f}s | mem={mem_stats['knowledge_facts']} | episodes={ep_stats['overall']['total']} | acc={overall_acc*100:.0f}%")
        
        time.sleep(2)  # Rate limit
        
    except Exception as ex:
        elapsed = time.time() - start
        print(f"  [{i+1}/{N}] ERROR: {str(ex)[:60]} | {elapsed:.1f}s")
        time.sleep(3)

# Step 5: Run the feedback loop to generate rules
print(f"\n{'='*60}")
print("RUNNING FEEDBACK LOOP")
print(f"{'='*60}\n")

rules = feedback.analyze_and_generate_rules()
print(f"Generated {len(rules)} procedural rules:")
for rule in rules:
    print(f"  [{rule.category}] (conf={rule.confidence:.2f}) {rule.rule_text[:80]}")

# Try LLM rule generation
episodes = episodic.get_similar_episodes(limit=20)
llm_rules = feedback.generate_advanced_rules_with_llm(episodes)
print(f"\nLLM generated {len(llm_rules)} additional rules:")
for rule in llm_rules:
    print(f"  [{rule.category}] {rule.rule_text[:80]}")

# Final context with all memory types
print(f"\n{'='*60}")
print("FINAL ENHANCED CONTEXT (after learning)")
print(f"{'='*60}\n")
final_context = retriever.build_full_context("IonQ announces new quantum processor milestone", source_type="news")
print(final_context[:1000])

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
stats = episodic.get_accuracy_by_category()
print(f"Total episodes: {stats['overall']['total']}")
print(f"Overall accuracy: {stats['overall']['accuracy']*100:.0f}%")
print(f"Memory facts: {memory.get_memory_stats()['knowledge_facts']}")
print(f"Active rules: {len(procedural.get_active_rules())}")
print(f"Context size: {len(final_context)} chars")

# Cleanup
os.remove(DB_PATH)
