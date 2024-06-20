#!/bin/env bash
#SBATCH --job-name=test-knn-reference
#SBATCH --partition=nvidia-a100
#SBATCH --nodelist=fijigpu-04
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=test-knn-reference.out
#SBATCH --error=test-knn-reference.err
#SBATCH --signal=SIGUSR1@90

set -e

# cleanup() {
#     echo "Saving work and cleaning up"
#     rsync -a $workdir/ $savedir/
#     rm -r $workdir
# }
# trap cleanup EXIT SIGINT SIGTERM SIGUSR1


## ----------------------------------------------------------------------------
## Setup
## TODO try something other than hg002 (eg something from 1kg)
## ----------------------------------------------------------------------------
workdir="/cache/murad"
mkdir -p $workdir

savedir="/scratch/Shares/layer/projects/sequence_similarity_search/knn-reference-test"
if [ -d $savedir ]; then
    echo "Restoring from previous run"
    cp -r $savedir/. $workdir/
fi

ref="$workdir/GRCh38_full_analysis_set_plus_decoy_hla.fa"
bed="$workdir/GRCh38_full_analysis_set_plus_decoy_hla.fa.bed"
faiss_index=$workdir/GRCh38_full_analysis_set_plus_decoy_hla.fa.faiss

if [ ! -f $workdir/$ref ]; then
    cp /scratch/Shares/layer/ref/hg38/GRCh38_full_analysis_set_plus_decoy_hla.fa "$workdir/"
    cp /scratch/Shares/layer/ref/hg38/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai "$workdir/"
fi

fastq1_url='ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/HG002_HiSeq300x_fastq/140528_D00360_0018_AH8VC6ADXX/Project_RM8391_RM8392/Sample_2A1/2A1_CGATGT_L001_R1_001.fastq.gz'
fastq2_url='ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/HG002_HiSeq300x_fastq/140528_D00360_0018_AH8VC6ADXX/Project_RM8391_RM8392/Sample_2A1/2A1_CGATGT_L001_R2_001.fastq.gz'

fq1=$(basename $fastq1_url)
fq2=$(basename $fastq2_url)


## ----------------------------------------------------------------------------
## BUILD INDEX
## ----------------------------------------------------------------------------
if [ ! -f $faiss_index ]; then
    echo "Indexing reference genome"
    python3 knn-reference.py build \
        --fasta $ref \
        --window-size 150 \
        --stride 25 \
        --checkpoint-path "fine-tuning-normalized/epoch=6-val_loss=0.0079.ckpt" \
        --batch-size 4096 \
        --embeddings-dim 256 \
        --num-workers 16 \
        --devices 4
        # --chromosome-names chr22 # add this line for quick testing
fi

## ----------------------------------------------------------------------------
## QUERY WITH HG002
## ----------------------------------------------------------------------------
if [ ! -f "$workdir/$fq1" ]; then
    wget $fastq1_url -O $workdir/$fq1
fi

if [ ! -f "$workdir/$fq2" ]; then
    wget $fastq2_url -O $workdir/$fq2
fi

echo "Running KNN search"
python knn-reference.py query \
    --faiss-index $faiss_index \
    --windowed-bed $bed \
    --query-fastqs "$workdir/$fq1" "$workdir/$fq2" \
    --checkpoint-path "fine-tuning-normalized/epoch=6-val_loss=0.0079.ckpt" \
    --batch-size 512 \
    --devices 1 \
    --num-workers 16 \
    --topk 10 \
    --output $workdir/knn-reference-queries.txt
