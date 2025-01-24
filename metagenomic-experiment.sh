#!/bin/env bash
#SBATCH --job-name=metagenomic-index
#SBATCH --partition=nvidia-a100
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=500GB
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=logs/metagenomic-index.out
#SBATCH --error=logs/metagenomic-index.err

set -e

# TODO move inference data to /cache/much8161-results
# TODO after inference, move the results back to /scratch/Shares/layer/projects/sequence_similarity_search/metagenomics-experiment/


mkdir -p /cache/much8161-results

python ssearch/experiments/metagenomics_index.py build-index

# if [ ! -f "/cache/much8161-results/index.faiss" ]; then
#     cp /scratch/Shares/layer/projects/sequence_similarity_search/metagenomics-experiment/index.faiss \
#        /cache/much8161-results/
# fi

# python ssearch/experiments/metagenomics_index.py search-index
# cp /cache/much8161-results/query_results_I.npy /scratch/Shares/layer/projects/sequence_similarity_search/metagenomics-experiment/
# cp /cache/much8161-results/query_results_D.npy /scratch/Shares/layer/projects/sequence_similarity_search/metagenomics-experiment/
# cp /cache/much8161-results/query_dataset_0.metadata /scratch/Shares/layer/projects/sequence_similarity_search/metagenomics-experiment/
# cp /cache/much8161-results/index.faiss /scratch/Shares/layer/projects/sequence_similarity_search/metagenomics-experiment/
