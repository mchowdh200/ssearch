#!/bin/env bash
#SBATCH --job-name=finetune
#SBATCH --partition=nvidia-a100
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=500GB
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=logs/fine-tuning.out
#SBATCH --error=logs/fine-tuning.err

mkdir -p CHECKPOINTS
ssearch train
# python finetune.py \
#     --checkpoint-dir fine-tuning-normalized \
#     --train-data /scratch/Shares/layer/projects/sequence_similarity_search/training_data/grch38-gg-silva.txt \
#     --val-data /scratch/Shares/layer/projects/sequence_similarity_search/training_data/mpox-subsampled.txt \
#     --devices 4 \
#     --epochs 100 \
#     --batch-size 768 \
#     --num-workers 32 \
#     --learning-rate 1e-4 \
#     --weight-decay 1e-2 \
#     --name finetune-normalized
