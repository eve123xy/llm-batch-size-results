#!/bin/bash
#SBATCH --job-name=kv_delay_probe
#SBATCH --account=torch_pr_1087_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xl5874/kv_delay_probe_20260909/job_%j.out
#SBATCH --error=/scratch/xl5874/kv_delay_probe_20260909/job_%j.err
set -euo pipefail
export TMPDIR=/scratch/xl5874/kv_delay_probe_20260909/tmp
export HF_HOME=/scratch/xl5874/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PATH=/scratch/xl5874/conda/envs/vllm_serve/bin:$PATH
cd /home/xl5874/llm-batch-size-results
python -u -B run_kv_delay_probe.py --results /scratch/xl5874/kv_delay_probe_20260909/results_$SLURM_JOB_ID
