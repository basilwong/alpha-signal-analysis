"""Deploy the fixed app to HF Spaces and monitor status."""
import time
from huggingface_hub import HfApi

TOKEN = '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'
SPACE_ID = 'build-small-hackathon/alpha-signal-analysis'

api = HfApi(token=TOKEN)

# Step 1: Upload the fixed app.py
print("Uploading fixed app.py to Space...")
api.upload_file(
    path_or_fileobj='app_server_fixed.py',
    path_in_repo='app.py',
    repo_id=SPACE_ID,
    repo_type='space'
)
print("Upload complete.")

# Step 2: Update requirements.txt to ensure gradio version supports Server mode
requirements = """# Core (HF Spaces deployment)
gradio>=5.29.0
spaces
transformers>=4.45.0
torch>=2.0.0
accelerate
peft
bitsandbytes
sentencepiece
protobuf

# Data processing
pandas
numpy
pyarrow

# Utilities
tqdm
requests
"""

print("Uploading updated requirements.txt...")
api.upload_file(
    path_or_fileobj=requirements.encode(),
    path_in_repo='requirements.txt',
    repo_id=SPACE_ID,
    repo_type='space'
)
print("Requirements uploaded.")

# Step 3: Restart the Space
print("Restarting Space...")
api.restart_space(SPACE_ID)
print("Space restart triggered.")

# Step 4: Wait and check status
print("Waiting 90 seconds for Space to rebuild...")
time.sleep(90)

# Step 5: Check status
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
    elif stage in ('BUILDING', 'STARTING'):
        print("Still building/starting, waiting 60 more seconds...")
        time.sleep(60)
    else:
        print(f"Unknown stage: {stage}, waiting 30 seconds...")
        time.sleep(30)
