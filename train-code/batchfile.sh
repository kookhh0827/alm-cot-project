#!/bin/bash

#SBATCH -D /jet/home/hkook/workspace/alm_cot_project/alm-cot-project/train-code
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

#SBATCH -A cis220031p
#SBATCH -p GPU-shared
#SBATCH --time=12:00:00
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
##SBATCH -N 1

module load anaconda3
conda activate /ocean/projects/cis220031p/hkook/envs/almcot
cd /jet/home/hkook/workspace/alm_cot_project/alm-cot-project/train-code

# Launch with torchrun for DDP
# --nproc_per_node should match the number of GPUs requested
torchrun --nproc_per_node=1 finetune_qwen_audio.py