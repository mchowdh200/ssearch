#!/bin/env bash
#SBATCH --job-name=knn-reference
#SBATCH --partition=nvidia-a100
#SBATCH --nodelist=fijigpu-05
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=500GB
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=logs/knn-reference.out
#SBATCH --error=logs/knn-reference.err

set -e

# source ${HOME}/miniforge3/bin/activate ssearch

python ssearch/experiments/knn_reference_small.py build-index

mkdir -p /scratch/Shares/layer/projects/sequence_similarity_search/knn-reference-small
cp /cache/much8161-results/index.faiss \
   /scratch/Shares/layer/projects/sequence_similarity_search/knn-reference-small/index.faiss

