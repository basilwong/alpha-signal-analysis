"""
6-Way Model Comparison Evaluation.
Runs predictions for all configurations that are missing.

Configs:
1. qwen3-8b baseline (no memory)
2. qwen3-8b + memory (iterative)
3. qwen3-14b baseline (no memory)
4. qwen3-14b + memory (iterative)
5. qwen3-14b-ft baseline (no memory)
6. qwen3-14b-ft + memory (iterative)

Usage:
    python scripts/run_6way_eval.py --config 1   # Run specific config
    python scripts/run_6way_eval.py --config all  # Run all missing configs
"""
import json
import os
import sys
import time
import argparse

sys.path.insert(0, '.')

from openai import OpenAI
from agent.config import QUANTUM_TICKERS

# API setup - use the international endpoint for the fine-tuned model
API_KEY = 'sk-ws-H.IIMPYP.OVEd.MEYCIQCgnJiyfu3TI7aOMuMio4dSrWTf5zbFNrCpKP-NTyUGagIhAJQ6AGEG4uC8C9LmDEqJCLQGSUnilOLV6lQ1QR7QvVBi'
BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

CONFIGS = {
    "1_8b_base": {"model": "qwen3-8b", "memory": False, "output": "data/eval/predictions_8b_base.jsonl"},
    "2_8b_memory": {"model": "qwen3-8b", "memory": True, "output": "data/eval/predictions_8b_memory.jsonl"},
    "3_14b_base": {"model": "qwen3-14b", "memory": False, "output": "data/eval/predictions_14b_base.jsonl"},
    "4_14b_memory": {"model": "qwen3-14b", "memory": True, "output": "data/eval/predictions_14b_memory.jsonl"},
    "5_14b_ft_base": {"model": "qwen3-14b-248ab2996693", "memory": False, "output": "data/eval/predictions_14b_ft_base.jsonl"},
    "6_14b_ft_memory": {"model": "qwen3-14b-248ab2996693", "memory": True, "output": "data/eval/predictions_14b_ft_memory.jsonl"},
}

EVAL_ARTICLES = "data/raw/articles_eval.jsonl"
SYSTEM_PROMPT = """You are a quantitative signal generator for quantum computing stocks.
Generate a signal vector for: IONQ, RGTI, QBTS, QUBT, QNT, IBM, GOOGL, MSFT, HON, NVDA.
Score range: -2.0 to +2.0. GOOGL/MSFT/NVDA always 0.0.
Output ONLY valid JSON with signal_vector (dict: {"IONQ": 1.5, "RGTI": -0.3, ...}) and chain_of_thought."""

MAX_ARTICLES = 200  # Process 200 for each config
RATE_LIMIT = 2


def run_config(config_name, config, articles):
    """Run a single configuration."""
    model = config["model"]
    use_memory = config["memory"]
    output_file = config["output"]
    
    # Check for existing results (resume support)
    existing = 0
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = sum(1 for _ in f)
        if existing >= MAX_ARTICLES:
            print(f"  Config {config_name}: Already complete ({existing} predictions)")
            return existing
        print(f"  Config {config_name}: Resuming from {existing}")
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Memory setup (if needed)
    memory_context = ""
    if use_memory:
        from agent.memory import MemoryStore
        from agent.memory_loop import EpisodicMemory, ProceduralMemory, EnhancedRetriever, FeedbackLoop, Episode
        from agent.seed_data import SEED_FACTS
        import pandas as pd
        
        db_path = f"data/memory_{config_name}.db"
        if existing == 0 and os.path.exists(db_path):
            os.remove(db_path)
        memory = MemoryStore(db_path)
        if existing == 0:
            for fact in SEED_FACTS:
                memory.store_knowledge(fact['ticker'], fact['type'], fact['content'], 'seed')
        retriever = EnhancedRetriever(memory)
        episodic = EpisodicMemory(memory)
        feedback = FeedbackLoop(memory, llm_client=client, model_name=model)
        
        # Load market data for outcome tracking
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
    
    out_f = open(output_file, 'a' if existing > 0 else 'w')
    successes = existing
    errors = 0
    
    for i in range(existing, min(len(articles), MAX_ARTICLES)):
        article = articles[i]
        date = article.get('date', '')
        source = article.get('source', 'news')
        
        # Build prompt
        if use_memory:
            memory_context = retriever.build_full_context(article.get('text', ''), source_type=source)
            system = f"{SYSTEM_PROMPT}\n\n{memory_context}"
        else:
            system = SYSTEM_PROMPT
        
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Analyze ({date}, {source}):\n\n{article.get('text', '')}"}
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
                raise ValueError("No JSON found in response")
            
            sv = signal.get('signal_vector', signal)
            if isinstance(sv, list):
                sv = {item['ticker']: item.get('score', 0) for item in sv if isinstance(item, dict) and 'ticker' in item}
            
            # Clean scores
            sv_clean = {}
            for ticker, val in sv.items():
                if isinstance(val, (int, float)):
                    sv_clean[ticker] = float(val)
                elif isinstance(val, dict):
                    sv_clean[ticker] = float(val.get('score', 0))
                else:
                    try:
                        sv_clean[ticker] = float(val)
                    except:
                        sv_clean[ticker] = 0.0
            
            # Memory: record episode and store knowledge
            if use_memory and date:
                cot = signal.get('chain_of_thought', '')
                for ticker in QUANTUM_TICKERS[:5]:
                    score = sv_clean.get(ticker, 0)
                    if abs(score) < 0.3:
                        continue
                    # Get forward return
                    if ticker in market:
                        dates_list = market[ticker]["dates"]
                        values_list = market[ticker]["values"]
                        try:
                            start_idx = next(j for j, d in enumerate(dates_list) if d >= date)
                            end_idx = min(start_idx + 5, len(values_list) - 1)
                            if end_idx > start_idx and values_list[start_idx] != 0:
                                actual_ret = (values_list[end_idx] - values_list[start_idx]) / values_list[start_idx]
                                predicted_dir = "bullish" if score > 0 else "bearish"
                                actual_dir = "bullish" if actual_ret > 0 else "bearish"
                                ep = Episode(
                                    date=date, ticker=ticker, predicted_score=score,
                                    predicted_direction=predicted_dir,
                                    actual_return_5d=actual_ret, actual_direction=actual_dir,
                                    was_correct=(predicted_dir == actual_dir),
                                    article_title=article.get('title', '')[:50],
                                    source_type=source, reasoning_summary=cot[:150]
                                )
                                episodic.store_episode(ep)
                        except StopIteration:
                            pass
                
                # Store knowledge
                if cot and len(cot) > 30:
                    for ticker in QUANTUM_TICKERS[:5]:
                        score = sv_clean.get(ticker, 0)
                        if abs(score) >= 0.5:
                            memory.store_knowledge(ticker, "signal_context",
                                f"[{date}] Score {score:+.1f}: {cot[:100]}", source, ttl_days=60)
                
                # Run feedback every 50 articles
                if (i + 1) % 50 == 0:
                    feedback.analyze_and_generate_rules()
            
            pred = {
                "status": "success", "article_idx": i, "date": date,
                "title": article.get('title', ''), "source": source,
                "signal": signal, "signal_vector_clean": sv_clean,
                "time_seconds": elapsed
            }
            out_f.write(json.dumps(pred) + '\n')
            out_f.flush()
            successes += 1
            
        except Exception as ex:
            elapsed = time.time() - start
            errors += 1
            pred = {"status": "error", "article_idx": i, "date": date, "error": str(ex)[:100], "time_seconds": elapsed}
            out_f.write(json.dumps(pred) + '\n')
            out_f.flush()
        
        # Progress
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{MAX_ARTICLES}] success={successes} err={errors} | {elapsed:.1f}s")
        
        time.sleep(RATE_LIMIT)
    
    out_f.close()
    print(f"  Config {config_name}: Complete. {successes} successes, {errors} errors.")
    return successes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="all", help="Config to run (1-6 or 'all')")
    args = parser.parse_args()
    
    # Load articles
    with open(EVAL_ARTICLES) as f:
        articles = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(articles)} evaluation articles")
    
    # Determine which configs to run
    if args.config == "all":
        configs_to_run = list(CONFIGS.items())
    else:
        key = [k for k in CONFIGS if k.startswith(f"{args.config}_")]
        if key:
            configs_to_run = [(key[0], CONFIGS[key[0]])]
        else:
            print(f"Unknown config: {args.config}")
            sys.exit(1)
    
    print(f"\nRunning {len(configs_to_run)} configurations:")
    for name, cfg in configs_to_run:
        print(f"  {name}: model={cfg['model']}, memory={cfg['memory']}")
    
    # Run each config
    for name, cfg in configs_to_run:
        print(f"\n{'='*60}")
        print(f"CONFIG: {name}")
        print(f"{'='*60}")
        run_config(name, cfg, articles)


if __name__ == "__main__":
    main()
