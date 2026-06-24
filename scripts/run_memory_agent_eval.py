"""
Full batch evaluation: Process 400 articles through the memory agent.
Memory accumulates chronologically. Saves predictions for IC evaluation.

Usage:
    python scripts/run_memory_agent_eval.py [--limit 400] [--resume]
"""
import json
import time
import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, '.')

from agent.memory import MemoryStore
from agent.retrieval import MemoryRetriever
from agent.inference import generate_signal
from agent.seed_data import SEED_FACTS
from agent.config import QUANTUM_TICKERS

# Paths
EVAL_ARTICLES = "data/raw/articles_eval.jsonl"
OUTPUT_FILE = "data/eval/predictions_memory_agent.jsonl"
MEMORY_DB = "data/memory_eval.db"  # Separate DB for eval (doesn't pollute the demo DB)

def setup_memory():
    """Create a fresh memory store seeded with baseline knowledge."""
    if os.path.exists(MEMORY_DB):
        os.remove(MEMORY_DB)
    memory = MemoryStore(MEMORY_DB)
    for fact in SEED_FACTS:
        memory.store_knowledge(fact['ticker'], fact['type'], fact['content'], 'seed')
    return memory

def extract_knowledge_from_signal(signal: dict, article: dict, memory: MemoryStore):
    """Extract new knowledge from the model's reasoning and store in memory."""
    sv = signal.get('signal_vector', signal)
    cot = signal.get('chain_of_thought', '')
    
    if not isinstance(sv, dict):
        return
    
    # Store facts for tickers with strong signals
    for ticker, data in sv.items():
        score = data if isinstance(data, (int, float)) else (data.get('score', 0) if isinstance(data, dict) else 0)
        reasoning = data.get('reasoning', '') if isinstance(data, dict) else ''
        
        if abs(score) >= 0.5 and reasoning and len(reasoning) > 20:
            memory.store_knowledge(
                ticker=ticker,
                fact_type="signal_context",
                content=f"[{article.get('date', '')}] {reasoning[:200]}",
                source=article.get('source', 'news'),
                ttl_days=60
            )

def load_existing_results():
    """Load existing results for resume support."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return [json.loads(l) for l in f if l.strip()]
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--rate-limit', type=float, default=2.0, help='Seconds between API calls')
    args = parser.parse_args()

    # Load articles
    with open(EVAL_ARTICLES) as f:
        articles = [json.loads(l) for l in f if l.strip()]
    articles = articles[:args.limit]
    print(f"Loaded {len(articles)} evaluation articles")

    # Resume support
    existing = []
    start_idx = 0
    if args.resume and os.path.exists(OUTPUT_FILE):
        existing = load_existing_results()
        start_idx = len(existing)
        print(f"Resuming from article {start_idx} ({len(existing)} already done)")
        # Rebuild memory from existing results
        memory = MemoryStore(MEMORY_DB)
        if memory.get_memory_stats()['knowledge_facts'] == 0:
            # Re-seed if DB was deleted
            for fact in SEED_FACTS:
                memory.store_knowledge(fact['ticker'], fact['type'], fact['content'], 'seed')
    else:
        memory = setup_memory()
        print(f"Fresh memory seeded with {len(SEED_FACTS)} facts")

    retriever = MemoryRetriever(memory)

    # Open output file (append mode for resume)
    mode = 'a' if args.resume and existing else 'w'
    out_f = open(OUTPUT_FILE, mode)

    successes = len(existing)
    errors = 0
    total_time = 0

    print(f"\n{'='*60}")
    print(f"Processing articles {start_idx} to {len(articles)-1}")
    print(f"Model: qwen-plus | Rate limit: {args.rate_limit}s")
    print(f"{'='*60}\n")

    for i in range(start_idx, len(articles)):
        article = articles[i]
        title = article.get('title', 'Untitled')[:50]
        
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
            total_time += elapsed
            
            # Parse response
            content = result.get('content', '')
            try:
                s = content.find('{')
                e = content.rfind('}') + 1
                signal = json.loads(content[s:e]) if s != -1 else json.loads(content)
            except json.JSONDecodeError:
                errors += 1
                pred = {"status": "error", "article_idx": i, "date": article.get('date', ''), "title": title, "error": "JSON parse failed", "time_seconds": elapsed}
                out_f.write(json.dumps(pred) + '\n')
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(articles)}] ERROR (JSON) | {title}... | {elapsed:.1f}s")
                time.sleep(args.rate_limit)
                continue
            
            # Accumulate memory from this signal
            extract_knowledge_from_signal(signal, article, memory)
            
            # Store signal in memory history
            sv = signal.get('signal_vector', signal)
            memory.store_signal(
                article_date=article.get('date', ''),
                article_title=title,
                article_source=article.get('source', 'news'),
                signal_vector=sv,
                reasoning=signal.get('chain_of_thought', '')[:300],
                model_used='qwen-plus-memory'
            )
            
            # Save prediction
            pred = {
                "status": "success",
                "article_idx": i,
                "date": article.get('date', ''),
                "title": article.get('title', ''),
                "source": article.get('source', ''),
                "signal": signal,
                "time_seconds": elapsed,
                "memory_facts_at_time": memory.get_memory_stats()['knowledge_facts'],
            }
            out_f.write(json.dumps(pred) + '\n')
            out_f.flush()
            successes += 1
            
            # Progress log every 10 articles
            if (i + 1) % 10 == 0:
                avg_time = total_time / (i - start_idx + 1)
                remaining = (len(articles) - i - 1) * avg_time / 60
                mem_stats = memory.get_memory_stats()
                print(f"  [{i+1}/{len(articles)}] success={successes} errors={errors} | avg={avg_time:.1f}s | ETA={remaining:.0f}min | mem={mem_stats['knowledge_facts']} facts, {mem_stats['signals_stored']} signals")
            
        except Exception as ex:
            elapsed = time.time() - start
            total_time += elapsed
            errors += 1
            pred = {"status": "error", "article_idx": i, "date": article.get('date', ''), "title": title, "error": str(ex)[:200], "time_seconds": elapsed}
            out_f.write(json.dumps(pred) + '\n')
            out_f.flush()
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(articles)}] ERROR: {str(ex)[:50]} | {elapsed:.1f}s")
            time.sleep(5)  # Back off on error
        
        # Rate limit
        time.sleep(args.rate_limit)

    out_f.close()

    # Final summary
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {successes + errors}")
    print(f"Successes: {successes}")
    print(f"Errors: {errors}")
    print(f"Success rate: {successes/(successes+errors)*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Avg per article: {total_time/(successes+errors):.1f}s")
    print(f"Final memory: {memory.get_memory_stats()}")
    print(f"\nPredictions saved to: {OUTPUT_FILE}")
    print(f"Memory DB saved to: {MEMORY_DB} (can be used for demo)")

if __name__ == "__main__":
    main()
