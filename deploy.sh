#!/bin/bash
# Deploy the memory agent to Alibaba Cloud ECS

set -e

# 1. Apply Terraform (user must have set up credentials)
cd infra
terraform init
terraform apply -auto-approve
ECS_IP=$(terraform output -raw agent_public_ip)
echo "ECS IP: $ECS_IP"

# 2. Wait for instance to be ready
echo "Waiting for instance to boot..."
sleep 60

# 3. SSH and deploy the application
ssh -o StrictHostKeyChecking=no root@$ECS_IP << 'EOF'
cd /opt/alpha-signal-analysis
git pull origin main
pip3 install -r requirements.txt
# Seed the memory
python3 -c "from agent.seed_data import SEED_FACTS; from agent.memory import MemoryStore; m = MemoryStore('/opt/alpha-signal-analysis/data/memory.db'); [m.store_knowledge(f['ticker'], f['type'], f['content'], 'seed') for f in SEED_FACTS]"
# Start the server
pkill -f "uvicorn agent.server" || true
nohup python3 -m uvicorn agent.server:app --host 0.0.0.0 --port 8000 > /var/log/agent.log 2>&1 &
echo "Agent running on port 8000"
EOF

echo "Memory Agent deployed at http://$ECS_IP:8000"
echo "Health check: curl http://$ECS_IP:8000/api/health"
