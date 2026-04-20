#!/bin/bash
# Run one time series experiment and extract key metrics.
# Usage: bash script.sh
python ts_model.py > run.log 2>&1
echo "--- run complete ---"
grep "^val_mse:\|^peak_vram_mb:\|^model:\|^num_params_M:" run.log
