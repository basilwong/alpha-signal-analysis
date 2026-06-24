"""
Test script: Process 5 evaluation articles through the memory agent.
Validates output format, measures timing, checks for errors/throttling.
Run this before the full batch to confirm safe parameters.
"""
import json
import time
import sys
sys.path.insert(0, '.')

from agent.memory import MemoryStore
from agent.retrieval import MemoryRetriever
from agent.inference import generate_signal
from agent.seed_data import SEED_FACTS
from agent.config import QUANTUM_TICKERS

# Setup fresh memory for the test
MEMORY_DB = "data/test_memory.db"
import os
if os.path.exists(MEMORY_DB):
    os.remove(MEMORY_DB)

memory = MemoryStore(MEMORY_DB)
retriever = MemoryRetriever(memory)

# Seed with baseline knowledge
for fact in SEED_FACTS:
    memory.store_knowledge(fact['ticker'], fact['type'], fact['content'], 'seed')
print(f"Seeded {len(SEED_FACTS)} facts")

# Load eval articles
with open('data/raw/articles_eval.jsonl') as f:
    articles = [json.loads(l) for l in f if l.strip()]

# Process first 5
N_TEST = 5
results = []
errors = []
timings = []

print(f"\n{'='*60}")
print(f"Processing {N_TEST} articles through memory agent")
print(f"{'='*60}\n")

for i, article in enumerate(articles[:N_TEST]):
    print(f"[{i+1}/{N_TEST}] {article.get('title', 'Untitled')[:60]}...")
    
    start = time.time()
    try:
        # Retrieve memory context
        memory_context = retriever.retrieve_context(article.get('text', ''))
        
        # Generate signal
        result = generate_signal(
            article_text=article.get('text', ''),
            source_type=article.get('source', 'news'),
            memory_context=memory_context,
            enable_thinking=False
        )
        
        elapsed = time.time() - start
        timings.append(elapsed)
        
        # Parse the response
        content = result.get('content', '')
        thinking = result.get('thinking', '')
        
        # Try to extract JSON
        try:
            s = content.find('{')
            e = content.rfind('}') + 1
            if s != -1:
                signal = json.loads(content[s:e])
            else:
                signal = json.loads(content)
        except json.JSONDecodeError as je:
            errors.append({"idx": i, "error": f"JSON parse: {je}", "raw": content[:200]})
            print(f"  ERROR: JSON parse failed: {je}")
            continue
        
        # Validate signal vector
        sv = signal.get('signal_vector', signal)
        if isinstance(sv, dict):
            tickers_found = [t for t in QUANTUM_TICKERS if t in sv]
            scores = {t: (sv[t] if isinstance(sv[t], (int, float)) else sv[t].get('score', 0)) for t in tickers_found}
        else:
            tickers_found = []
            scores = {}
        
        # Store in memory (accumulate)
        if scores:
            memory.store_signal(
                article_date=article.get('date', ''),
                article_title=article.get('title', '')[:100],
                article_source=article.get('source', 'news'),
                signal_vector=sv,
                reasoning=signal.get('chain_of_thought', '')[:500],
                model_used='qwen-plus-memory'
            )
        
        # Build result in the format the evaluation expects
        pred = {
            "status": "success",
            "article_idx": i,
            "date": article.get('date', ''),
            "title": article.get('title', ''),
            "source": article.get('source', ''),
            "signal": signal,
            "time_seconds": elapsed,
            "memory_facts_used": len(memory_context.split('\n')) if memory_context else 0,
        }
        results.append(pred)
        
        print(f"  OK ({elapsed:.1f}s) | Tickers: {len(tickers_found)} | Top: {max(scores.values(), default=0):.2f} | Memory: {memory.get_memory_stats()['knowledge_facts']} facts")
        
        # Rate limit (be safe)
        time.sleep(2)
        
    except Exception as ex:
        elapsed = time.time() - start
        timings.append(elapsed)
        errors.append({"idx": i, "error": str(ex), "elapsed": elapsed})
        print(f"  ERROR ({elapsed:.1f}s): {ex}")
        time.sleep(5)  # Back off on error

# Summary
print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Processed: {N_TEST}")
print(f"Successes: {len(results)}")
print(f"Errors: {len(errors)}")
print(f"Avg time: {sum(timings)/len(timings):.1f}s")
print(f"Min time: {min(timings):.1f}s")
print(f"Max time: {max(timings):.1f}s")
print(f"Memory growth: {memory.get_memory_stats()}")
print(f"\nEstimated time for 421 articles: {sum(timings)/len(timings) * 421 / 60:.0f} minutes")
print(f"Estimated tokens (rough): {421 * 2500:.0f} (~{421 * 2500 / 1_000_000:.1f}M)")

if errors:
    print(f"\nErrors:")
    for e in errors:
        print(f"  [{e['idx']}] {e['error'][:100]}")

# Save test results
with open('data/test_memory_predictions.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
print(f"\nTest predictions saved to data/test_memory_predictions.jsonl")

# Cleanup
os.remove(MEMORY_DB)
