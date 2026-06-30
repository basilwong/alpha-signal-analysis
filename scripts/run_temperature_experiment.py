"""
Temperature Experiment: Re-run 14B configs at different temperatures to diagnose model collapse.
Runs configs 3-6 at temperatures 0.5, 0.7, 1.0 (we already have 0.3 results).
"""
import json
import os
import sys
import time
import argparse

sys.path.insert(0, '.')
from openai import OpenAI
from agent.config import QUANTUM_TICKERS

API_KEY = 'sk-ws-H.IIMPYP.OVEd.MEYCIQCgnJiyfu3TI7aOMuMio4dSrWTf5zbFNrCpKP-NTyUGagIhAJQ6AGEG4uC8C9LmDEqJCLQGSUnilOLV6lQ1QR7QvVBi'
BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

MODELS = {
    "14b_base": "qwen3-14b",
    "14b_ft": "qwen3-14b-248ab2996693",
}

TEMPERATURES = [0.5, 0.7, 1.0]
MAX_ARTICLES = 100  # Smaller sample for speed (100 articles per config)
RATE_LIMIT = 2
EVAL_ARTICLES = "data/raw/articles_eval.jsonl"

SYSTEM_PROMPT = """You are a quantitative signal generator for quantum computing stocks.
Generate a signal vector for: IONQ, RGTI, QBTS, QUBT, QNT, IBM, GOOGL, MSFT, HON, NVDA.
Score range: -2.0 to +2.0. GOOGL/MSFT/NVDA always 0.0.
Output ONLY valid JSON with signal_vector (dict: {"IONQ": 1.5, "RGTI": -0.3, ...}) and chain_of_thought."""


def run_experiment(model_key, model_name, temperature, articles):
    output_file = f"data/eval/temp_exp/{model_key}_t{temperature:.1f}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Resume support
    existing = 0
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = sum(1 for _ in f)
        if existing >= MAX_ARTICLES:
            print(f"  Already complete ({existing})")
            return output_file
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    out_f = open(output_file, 'a' if existing > 0 else 'w')
    successes = 0
    errors = 0
    
    for i in range(existing, min(len(articles), MAX_ARTICLES)):
        article = articles[i]
        date = article.get('date', '')
        source = article.get('source', 'news')
        
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze ({date}, {source}):\n\n{article.get('text', '')}"}
                ],
                temperature=temperature,
                max_tokens=1200,
                extra_body={"enable_thinking": False}
            )
            elapsed = time.time() - start
            content = response.choices[0].message.content or ""
            
            s = content.find('{')
            e = content.rfind('}') + 1
            if s != -1:
                signal = json.loads(content[s:e])
            else:
                raise ValueError("No JSON found")
            
            sv = signal.get('signal_vector', signal)
            if isinstance(sv, list):
                sv = {item['ticker']: item.get('score', 0) for item in sv if isinstance(item, dict) and 'ticker' in item}
            
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
            
            pred = {"status": "success", "article_idx": i, "date": date,
                    "title": article.get('title', ''), "source": source,
                    "signal_vector_clean": sv_clean, "time_seconds": elapsed}
            out_f.write(json.dumps(pred) + '\n')
            out_f.flush()
            successes += 1
        except Exception as ex:
            elapsed = time.time() - start
            errors += 1
            pred = {"status": "error", "article_idx": i, "date": date, "error": str(ex)[:100]}
            out_f.write(json.dumps(pred) + '\n')
            out_f.flush()
        
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{MAX_ARTICLES}] success={successes} err={errors}")
        
        time.sleep(RATE_LIMIT)
    
    out_f.close()
    print(f"  Done: {successes} successes, {errors} errors")
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", help="Model to test (14b_base, 14b_ft, or all)")
    parser.add_argument("--temp", type=float, default=None, help="Specific temperature (0.5, 0.7, 1.0)")
    args = parser.parse_args()
    
    with open(EVAL_ARTICLES) as f:
        articles = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(articles)} articles (using first {MAX_ARTICLES})")
    
    models_to_run = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    temps_to_run = [args.temp] if args.temp else TEMPERATURES
    
    for model_key, model_name in models_to_run.items():
        for temp in temps_to_run:
            print(f"\n{'='*60}")
            print(f"Model: {model_key} ({model_name}) | Temperature: {temp}")
            print(f"{'='*60}")
            run_experiment(model_key, model_name, temp, articles)


if __name__ == "__main__":
    main()
