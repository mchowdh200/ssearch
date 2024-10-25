#!/bin/env bash
#SBATCH --job-name=metagenomic-index
#SBATCH --partition=nvidia-a100
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=500GB
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=metagenomic-index.out
#SBATCH --error=metagenomic-index.err

python ssearch/experiments/metagenomics_index.py

# OUTPUT_DIR: "/cache/much8161-results"
mv /cache/much8161-results /scratch/Shares/layer/projects/sequence_similarity_search/experiments/metagenomic/index
