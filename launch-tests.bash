#!/bin/env bash
#SBATCH --job-name=launch-tests
#SBATCH --partition=nvidia-a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=test-model.out
#SBATCH --error=test-model.err

if [[ ! -f 'fine-tuning-normalized.faiss' ]]; then
    python test-model.py make-index \
        --checkpoint-path './fine-tuning-normalized/epoch=5-val_loss=0.0084.ckpt' \
        --fastq /localscratch/BetaCoV_bat_Yunnan_RmYN02_2019_1.fq.gz \
        --index-path fine-tuning-normalized.faiss \
        --batch-size 4096 \
        --devices 4 \
        --embeddings-dim 256
fi

python test-model.py search \
    --index-path fine-tuning-normalized.faiss \
    --checkpoint-path './fine-tuning-normalized/epoch=5-val_loss=0.0084.ckpt' \
    --batch-size 4096 \
    --fastas /scratch/Shares/layer/projects/sequence_similarity_search/viruses/*.fasta \
    --window-size 150 \
    --stride 10 \
    --topk 5 \
    --devices 4 \
    --query-output scored-queries.bed \
    --results-output aggregated-results.png
