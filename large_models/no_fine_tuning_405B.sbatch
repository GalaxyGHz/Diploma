#!/bin/bash
#BATCH --job-name=lazar_no_fine_tuning_sarcasm_405B
#SBATCH --partition=frida
#SBATCH --time=1-24:00:00
#SBATCH --gres=gpu:H100:8
#SBATCH --tasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --output=sbatch_out/405B/%x_%j.out

srun \
  --container-image nvcr.io#nvidia/pytorch:24.07-py3 \
  --container-mounts ${PWD}:${PWD} \
  --container-workdir ${PWD} \
  bash -c 'pip install pandas accelerate python-dotenv transformers peft bitsandbytes;MODEL_ID=2 python3 run_model.py'
