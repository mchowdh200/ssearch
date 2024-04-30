#!/bin/env bash
#SBATCH --partition=nvidia-a100
#SBATCH --job-name=make_index
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=make_index.out
#SBATCH --error=make_index.err

python make_index.py \
    --fastq /localscratch/BetaCoV_bat_Yunnan_RmYN02_2019_1.fq.gz \
    --batch_size 4096

