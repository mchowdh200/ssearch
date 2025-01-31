#!/usr/bin/env bash
#SBATCH --job-name=run_knn_reference
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=logs/main.out
#SBATCH --error=logs/main.err

mkdir -p logs
snakemake --executor slurm \
    --latency-wait 10 \
    --snakefile run_knn_reference.smk \
    --jobs 32 \
    --cores 32
