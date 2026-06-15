import json

INPUT = "data/training/quantum_alpha_train_v4.jsonl"
errors = []
total = 0

with open(INPUT) as f:
    for i, line in enumerate(f):
        total += 1
        r = json.loads(line)
        msgs = r.get("messages", [])
        
        # Check structure
        if len(msgs) != 3:
            errors.append(f"Line {i}: Expected 3 messages, got {len(msgs)}")
            continue
        if msgs[0]["role"] != "system" or msgs[1]["role"] != "user" or msgs[2]["role"] != "assistant":
            errors.append(f"Line {i}: Wrong role order")
            continue
        
        # Check assistant content is valid JSON
        try:
            signal = json.loads(msgs[2]["content"])
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: Invalid JSON in assistant: {e}")
            continue
        
        # Check signal vector has all tickers
        sv = signal.get("signal_vector", {})
        expected_tickers = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA", "QNT"]
        for ticker in expected_tickers:
            if ticker not in sv:
                errors.append(f"Line {i}: Missing ticker {ticker}")
        
        # Check MSFT/GOOGL/NVDA are 0.0
        for ticker in ["MSFT", "GOOGL", "NVDA"]:
            if ticker in sv and sv[ticker].get("score", 0) != 0:
                errors.append(f"Line {i}: {ticker} score should be 0.0, got {sv[ticker].get('score')}")
        
        # Check chain_of_thought is substantive
        cot = signal.get("chain_of_thought", "")
        if len(cot) < 50 or "REDACTED" in cot or "Not disclosed" in cot:
            errors.append(f"Line {i}: chain_of_thought too short or redacted")

print(f"Total: {total}")
print(f"Errors: {len(errors)}")
for e in errors[:20]:
    print(f"  {e}")
if len(errors) == 0:
    print("✓ All validation checks passed!")
