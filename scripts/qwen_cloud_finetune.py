"""
Fine-tune qwen3-vl-8b-instruct on Alibaba Cloud Model Studio (DashScope).
Uploads training data, creates the SFT job, and monitors until completion.

Usage:
    python scripts/qwen_cloud_finetune.py --upload    # Upload data + start job
    python scripts/qwen_cloud_finetune.py --status    # Check job status
    python scripts/qwen_cloud_finetune.py --full      # Upload + start + monitor until done
"""
import argparse
import json
import os
import sys
import time
import requests

sys.path.insert(0, '.')
from agent.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

# Use the workspace-specific base URL for fine-tuning
# The fine-tuning API uses the same base as inference
API_BASE = DASHSCOPE_BASE_URL.rstrip('/')
TRAINING_FILE = "data/training/quantum_alpha_train_v4.jsonl"
STATE_FILE = "data/finetune_state.json"

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def upload_training_file():
    """Upload the training JSONL file to Model Studio."""
    print(f"Uploading {TRAINING_FILE}...")
    
    if not os.path.exists(TRAINING_FILE):
        print(f"ERROR: Training file not found: {TRAINING_FILE}")
        sys.exit(1)
    
    # Count lines for sanity check
    with open(TRAINING_FILE) as f:
        lines = sum(1 for _ in f)
    print(f"  Training examples: {lines}")
    
    url = f"{API_BASE}/files"
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    
    with open(TRAINING_FILE, 'rb') as f:
        resp = requests.post(
            url,
            headers=headers,
            files={"file": (os.path.basename(TRAINING_FILE), f, "application/jsonl")},
            data={"purpose": "fine-tune"}
        )
    
    if resp.status_code == 200:
        data = resp.json()
        file_id = data.get("id")
        print(f"  Upload successful! File ID: {file_id}")
        print(f"  Filename: {data.get('filename')}")
        print(f"  Bytes: {data.get('bytes')}")
        
        state = load_state()
        state["file_id"] = file_id
        state["upload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_state(state)
        return file_id
    else:
        print(f"  Upload FAILED: {resp.status_code}")
        print(f"  Response: {resp.text[:500]}")
        return None

def create_finetune_job(file_id):
    """Create a fine-tuning job on Model Studio."""
    print(f"\nCreating fine-tuning job...")
    print(f"  Model: qwen3-vl-8b-instruct")
    print(f"  Training file: {file_id}")
    print(f"  Epochs: 4")
    
    url = f"{API_BASE}/fine_tuning/jobs"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen3-vl-8b-instruct",
        "training_file": file_id,
        "hyperparameters": {
            "n_epochs": 4
        }
    }
    
    resp = requests.post(url, headers=headers, json=payload)
    
    if resp.status_code == 200:
        data = resp.json()
        job_id = data.get("id")
        status = data.get("status")
        print(f"  Job created! ID: {job_id}")
        print(f"  Status: {status}")
        
        state = load_state()
        state["job_id"] = job_id
        state["job_status"] = status
        state["job_created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_state(state)
        return job_id
    else:
        print(f"  Job creation FAILED: {resp.status_code}")
        print(f"  Response: {resp.text[:500]}")
        return None

def check_job_status(job_id=None):
    """Check the status of a fine-tuning job."""
    if not job_id:
        state = load_state()
        job_id = state.get("job_id")
    
    if not job_id:
        print("ERROR: No job ID found. Run --upload first.")
        return None
    
    url = f"{API_BASE}/fine_tuning/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    
    resp = requests.get(url, headers=headers)
    
    if resp.status_code == 200:
        data = resp.json()
        status = data.get("status")
        model = data.get("fine_tuned_model", "")
        trained_tokens = data.get("trained_tokens", 0)
        
        print(f"  Job ID: {job_id}")
        print(f"  Status: {status}")
        if model:
            print(f"  Fine-tuned model: {model}")
        if trained_tokens:
            print(f"  Trained tokens: {trained_tokens:,}")
        
        # Save state
        state = load_state()
        state["job_status"] = status
        if model:
            state["fine_tuned_model"] = model
        state["last_checked"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_state(state)
        
        return data
    else:
        print(f"  Status check FAILED: {resp.status_code}")
        print(f"  Response: {resp.text[:500]}")
        return None

def monitor_until_complete(job_id, check_interval=60):
    """Monitor the job until it completes or fails."""
    print(f"\nMonitoring job {job_id}...")
    print(f"  Checking every {check_interval} seconds")
    print(f"  Expected duration: 1-3 hours\n")
    
    start_time = time.time()
    checks = 0
    
    while True:
        checks += 1
        elapsed = (time.time() - start_time) / 60
        
        data = check_job_status(job_id)
        if not data:
            print(f"  [{checks}] Failed to get status. Retrying in {check_interval}s...")
            time.sleep(check_interval)
            continue
        
        status = data.get("status", "unknown")
        print(f"  [{checks}] Status: {status} | Elapsed: {elapsed:.1f} min")
        
        if status == "succeeded":
            model_name = data.get("fine_tuned_model", "")
            print(f"\n  FINE-TUNING COMPLETE!")
            print(f"  Model: {model_name}")
            print(f"  Total time: {elapsed:.1f} minutes")
            return model_name
        
        elif status in ("failed", "cancelled"):
            error = data.get("error", {})
            print(f"\n  FINE-TUNING FAILED!")
            print(f"  Error: {error}")
            return None
        
        elif status in ("queued", "running", "validating_files"):
            time.sleep(check_interval)
        
        else:
            print(f"  Unknown status: {status}. Continuing to monitor...")
            time.sleep(check_interval)

def main():
    parser = argparse.ArgumentParser(description="Qwen Cloud Fine-Tuning")
    parser.add_argument("--upload", action="store_true", help="Upload data and start job")
    parser.add_argument("--status", action="store_true", help="Check job status")
    parser.add_argument("--full", action="store_true", help="Full pipeline: upload + start + monitor")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    args = parser.parse_args()
    
    if args.status:
        check_job_status()
    elif args.upload or args.full:
        # Step 1: Upload
        file_id = upload_training_file()
        if not file_id:
            sys.exit(1)
        
        # Step 2: Create job
        job_id = create_finetune_job(file_id)
        if not job_id:
            sys.exit(1)
        
        # Step 3: Monitor (if --full)
        if args.full:
            model_name = monitor_until_complete(job_id, check_interval=args.interval)
            if model_name:
                print(f"\nReady to use! Set model='{model_name}' in your inference calls.")
            else:
                sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
