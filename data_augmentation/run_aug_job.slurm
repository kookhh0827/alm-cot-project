#!/bin/bash
#SBATCH -J ears_test
#SBATCH -A cis220031p
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o ears_aug.out
#SBATCH -e ears_aug.err

set -e
cd /ocean/projects/cis220031p/hkook/hailey
python3 ears_aug.py
