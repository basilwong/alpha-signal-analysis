import json

with open('data/eval/results_multi_model.json') as f:
    results = json.load(f)

print("Existing Model Results (IC @ +5 days):")
print("=" * 70)
print(f"{'Model':<45} {'IC@5d':>8} {'p-value':>10} {'DirAcc':>8}")
print("-" * 70)
for model in results:
    ic5 = None
    pval = None
    for h in results[model].get('decay_curve', []):
        if h['horizon'] == 5:
            ic5 = h['ic']
            pval = h['p_value']
    da = results[model].get('direction_accuracy_5d', {}).get('accuracy')
    ic_str = f"{ic5:.4f}" if ic5 is not None else "N/A"
    p_str = f"{pval:.4f}" if pval is not None else "N/A"
    da_str = f"{da:.1%}" if da is not None else "N/A"
    print(f"  {model:<43} {ic_str:>8} {p_str:>10} {da_str:>8}")

print("\n\nNote: OpenReasoning-Nemotron-7B predictions are ready but not yet evaluated.")
print("Run: python eval/run_multi_model_eval.py to include it.")
