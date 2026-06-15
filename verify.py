"""Verify the Space is running correctly."""
import requests

url = "https://build-small-hackathon-alpha-signal-analysis.hf.space"

print("=" * 60)
print("VERIFICATION: Alpha Signal Analysis Space")
print("=" * 60)

# Test 1: API models endpoint
print("\n[1] Testing GET /api/models...")
try:
    resp = requests.get(f"{url}/api/models", timeout=30)
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"    Models: {[m['name'] for m in data.get('models', [])]}")
        print("    ✓ PASS")
    else:
        print(f"    ✗ FAIL: {resp.text[:200]}")
except Exception as e:
    print(f"    ✗ ERROR: {e}")

# Test 2: Frontend
print("\n[2] Testing GET / (frontend)...")
try:
    resp = requests.get(url, timeout=30)
    print(f"    Status: {resp.status_code}")
    has_quantum = "Alpha Signal Analysis" in resp.text
    print(f"    Contains 'Alpha Signal Analysis': {has_quantum}")
    if has_quantum:
        print("    ✓ PASS")
    else:
        print(f"    ✗ FAIL: First 500 chars: {resp.text[:500]}")
except Exception as e:
    print(f"    ✗ ERROR: {e}")

# Test 3: Events endpoint
print("\n[3] Testing GET /api/events...")
try:
    resp = requests.get(f"{url}/api/events?model=Qwen3-8B Fine-tuned V1 (qwen3-max)", timeout=30)
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"    Events count: {len(data.get('events', []))}")
        print("    ✓ PASS")
    else:
        print(f"    ✗ FAIL: {resp.text[:200]}")
except Exception as e:
    print(f"    ✗ ERROR: {e}")

# Test 4: Eval metrics
print("\n[4] Testing GET /api/eval_metrics...")
try:
    resp = requests.get(f"{url}/api/eval_metrics", timeout=30)
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"    Models in eval: {list(data.keys())[:5]}")
        print("    ✓ PASS")
    else:
        print(f"    ✗ FAIL: {resp.text[:200]}")
except Exception as e:
    print(f"    ✗ ERROR: {e}")

# Test 5: Sector data
print("\n[5] Testing GET /api/sector_data...")
try:
    resp = requests.get(f"{url}/api/sector_data", timeout=30)
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"    Tickers: {list(data.get('tickers', {}).keys())}")
        print("    ✓ PASS")
    else:
        print(f"    ✗ FAIL: {resp.text[:200]}")
except Exception as e:
    print(f"    ✗ ERROR: {e}")

# Test 6: Prediction comparison
print("\n[6] Testing GET /api/prediction_comparison?article_idx=0...")
try:
    resp = requests.get(f"{url}/api/prediction_comparison?article_idx=0", timeout=30)
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"    Models with predictions: {list(data.get('models', {}).keys())}")
        print("    ✓ PASS")
    else:
        print(f"    ✗ FAIL: {resp.text[:200]}")
except Exception as e:
    print(f"    ✗ ERROR: {e}")

# Test 7: Static files (CSS/JS)
print("\n[7] Testing GET /static/styles.css...")
try:
    resp = requests.get(f"{url}/static/styles.css", timeout=30)
    print(f"    Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"    CSS length: {len(resp.text)} chars")
        print("    ✓ PASS")
    else:
        print(f"    ✗ FAIL: {resp.text[:200]}")
except Exception as e:
    print(f"    ✗ ERROR: {e}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
