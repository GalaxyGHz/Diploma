#!/bin/bash
#BATCH --job-name=lazar_no_fine_tuning_sarcasm_8B
#SBATCH --partition=frida
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=sbatch_out/8B/%x_%j.out

srun \
  --container-image nvcr.io#nvidia/pytorch:24.07-py3 \
  --container-mounts ${PWD}:${PWD} \
  --container-workdir ${PWD} \
  bash -c 'pip install pandas accelerate python-dotenv transformers;MODEL_ID=0 python3 run_no_fine_tuning.py'
