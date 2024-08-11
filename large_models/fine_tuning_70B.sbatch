#!/bin/bash
#BATCH --job-name=lazar_fine_tuning_sarcasm_70B
#SBATCH --partition=frida
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:H100:6
#SBATCH --tasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --output=sbatch_out/70B/%x_%j.out

srun \
  --container-image nvcr.io#nvidia/pytorch:24.07-py3 \
  --container-mounts ${PWD}:${PWD} \
  --container-workdir ${PWD} \
  bash -c 'pip install pandas accelerate python-dotenv transformers datasets optuna evaluate trl peft bitsandbytes;MODEL_ID=1 accelerate launch trainer_70B.py'
