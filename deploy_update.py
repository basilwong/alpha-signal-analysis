"""Upload the fixed app.py and frontend to HF Space."""
import time
from huggingface_hub import HfApi

TOKEN = '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'
SPACE_ID = 'build-small-hackathon/alpha-signal-analysis'

api = HfApi(token=TOKEN)

# Upload the Qwen3-30B Thinking predictions file (not yet on Space)
print("Uploading predictions_qwen3_30b_thinking.jsonl...")
api.upload_file(
    path_or_fileobj='data/eval/predictions_qwen3_30b_thinking.jsonl',
    path_in_repo='data/eval/predictions_qwen3_30b_thinking.jsonl',
    repo_id=SPACE_ID,
    repo_type='space'
)

# Upload the fixed app.py
print("Uploading fixed app.py...")
api.upload_file(
    path_or_fileobj='app_server_fixed.py',
    path_in_repo='app.py',
    repo_id=SPACE_ID,
    repo_type='space'
)

# Upload the updated frontend app.js
print("Uploading updated frontend/app.js...")
api.upload_file(
    path_or_fileobj='frontend_v2/app.js',
    path_in_repo='frontend_v2/app.js',
    repo_id=SPACE_ID,
    repo_type='space'
)

print("All files uploaded. Restarting Space...")
api.restart_space(SPACE_ID)

print("Waiting 90 seconds for Space to rebuild...")
time.sleep(90)

# Check status
for attempt in range(5):
    info = api.space_info(SPACE_ID)
    runtime = info.runtime
    stage = runtime.stage
    print(f"\nAttempt {attempt + 1}: Stage = {stage}")
    
    if stage == 'RUNNING':
        print("SUCCESS! Space is running.")
        break
    elif stage == 'RUNTIME_ERROR':
        error_msg = runtime.raw.get('errorMessage', 'Unknown error')
        print(f"FAILED: {error_msg}")
        break
    elif stage in ('BUILDING', 'STARTING', 'APP_STARTING'):
        print("Still building/starting, waiting 60 more seconds...")
        time.sleep(60)
    else:
        print(f"Unknown stage: {stage}, waiting 30 seconds...")
        time.sleep(30)
