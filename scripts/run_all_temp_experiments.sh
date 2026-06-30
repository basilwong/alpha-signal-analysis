#!/bin/bash
cd /home/ubuntu/quantum-alpha-intelligence

python3 scripts/run_temperature_experiment.py --model 14b_base --temp 0.5 &
python3 scripts/run_temperature_experiment.py --model 14b_base --temp 0.7 &
python3 scripts/run_temperature_experiment.py --model 14b_base --temp 1.0 &
python3 scripts/run_temperature_experiment.py --model 14b_ft --temp 0.5 &
python3 scripts/run_temperature_experiment.py --model 14b_ft --temp 0.7 &
python3 scripts/run_temperature_experiment.py --model 14b_ft --temp 1.0 &

echo "All 6 experiments started. Waiting..."
wait
echo "All experiments complete."
