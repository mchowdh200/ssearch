#!/bin/env bash
#SBATCH --job-name=plot-metagenomic-results
#SBATCH --partition=short
#SBATCH --time=00:60:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=logs/plot-metagenomic-results.out
#SBATCH --error=logs/plot-metagenomic-results.err

set -e

python ssearch/experiments/metagenomics_index.py plot \
    --output-dir /scratch/Shares/layer/projects/sequence_similarity_search/metagenomics-experiment/plots
