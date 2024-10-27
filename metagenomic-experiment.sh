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

set -e

if [ ! -f "/cache/much8161-results/BetaCoV_bat_Yunnan_RmYN02_2019_1.fq.gz" ]; then
    cp /scratch/Shares/layer/projects/sequence_similarity_search/SRR12432009/BetaCoV_bat_Yunnan_RmYN02_2019_1.fq.gz \
       /cache/much8161-results/
    cp /scratch/Shares/layer/projects/sequence_similarity_search/SRR12432009/BetaCoV_bat_Yunnan_RmYN02_2019_1.fq.gz.fxi \
       /cache/much8161-results/
    cp /scratch/Shares/layer/projects/sequence_similarity_search/SRR12432009/BetaCoV_bat_Yunnan_RmYN02_2019_2.fq.gz \
       /cache/much8161-results/
    cp /scratch/Shares/layer/projects/sequence_similarity_search/SRR12432009/BetaCoV_bat_Yunnan_RmYN02_2019_2.fq.gz.fxi \
       /cache/much8161-results/
fi

python ssearch/experiments/metagenomics_index.py build-index

# OUTPUT_DIR: "/cache/much8161-results"
rm /cache/much8161-results/BetaCoV_bat_Yunnan_RmYN02_2019_1.fq.gz
rm /cache/much8161-results/BetaCoV_bat_Yunnan_RmYN02_2019_2.fq.gz
mv /cache/much8161-results /scratch/Shares/layer/projects/sequence_similarity_search/experiments/metagenomic/index

