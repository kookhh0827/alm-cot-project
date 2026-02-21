#!/bin/bash
#SBATCH -J extract_feats
#SBATCH -A cis220031p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o extract_feats.out
#SBATCH -e extract_feats.err

set -e

cd /ocean/projects/cis220031p/hkook/hailey/new_13_feats
python process_dataset.py
