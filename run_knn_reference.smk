import os
import shutil
import re
from os.path import basename, dirname
from ssearch.config import KNNReferenceConfig

# from ssearch.experiments import knn_reference
from ssearch.data_utils.genomic_utils import filter_contigs
from pathlib import Path

Config = KNNReferenceConfig()
outdir = Path(Config.OUTPUT_DIR)
logdir = Path("logs")


def stem(path: str) -> tuple[str, str]:
    # Two capture groups: first is path stopping at first dot, second is the rest
    return re.match(r"(.+?)(\.[^.]+)+$", basename(path)).groups()[0]


rule All:
    input:
        contig_indices=[
            f"{outdir}/knn_indices/{contig}" for contig in Config.REFERENCE_CONTIGS
        ],
        bam=f"{outdir}/NA12878.bam",


rule FilterContigs:
    """
    For now, we just get want the main contigs of the reference genome
    """
    input:
        fasta=Config.REFERENCE_FASTA,
    output:
        fasta=f"{outdir}/{stem(Config.REFERENCE_FASTA)}_main_contigs.fa",
    resources:
        slurm_partion="short",
        runtime=60,  # minutes
        mem="16GB",
        cpus_per_task=1,
        nodes=1,
        slurm_extra=f"--output={logdir}/filter_contigs.out --error={logdir}/filter_contigs.err",
    run:
        knn_reference.filter_contigs(
            reference_fasta=input.fasta,
            output_fasta=output.fasta,
            keep=Config.REFERENCE_CONTIGS,
        )


rule BWAIndex:
    """
    Index the reference genome with BWA
    """
    input:
        fasta=rules.FilterContigs.output.fasta,
    output:
        bwt=f"{outdir}/{stem(Config.REFERENCE_FASTA)}_main_contigs.fa.bwt",
        sa=f"{outdir}/{stem(Config.REFERENCE_FASTA)}_main_contigs.fa.sa",
        pac=f"{outdir}/{stem(Config.REFERENCE_FASTA)}_main_contigs.fa.pac",
        amb=f"{outdir}/{stem(Config.REFERENCE_FASTA)}_main_contigs.fa.amb",
        ann=f"{outdir}/{stem(Config.REFERENCE_FASTA)}_main_contigs.fa.ann",
        fai=f"{outdir}/{stem(Config.REFERENCE_FASTA)}_main_contigs.fa.fai",
    resources:
        slurm_partition="short",
        runtime=240,  # minutes
        mem="64GB",
        cpus_per_task=1,
        nodes=1,
        slurm_extra=f"--output={logdir}/bwa_index.out --error={logdir}/bwa_index.err",
    shell:
        """
        samtools faidx {input.fasta}
        bwa index {input.fasta}
        """


rule BWAMem:
    """
    Align reads to the reference genome with BWA
    """
    input:
        fasta=rules.FilterContigs.output.fasta,
        bwt=rules.BWAIndex.output.bwt,
        sa=rules.BWAIndex.output.sa,
        pac=rules.BWAIndex.output.pac,
        amb=rules.BWAIndex.output.amb,
        ann=rules.BWAIndex.output.ann,
        fastq_r1=Config.FASTQ_R1,
        fastq_r2=Config.FASTQ_R2,
    output:
        bam=f"{outdir}/NA12878.bam",
    threads: workflow.cores
    resources:
        slurm_partition="short",
        runtime=720,  # minutes
        mem="64GB",
        cpus_per_task=workflow.cores,
        nodes=1,
        slurm_extra=f"--output={logdir}/bwa_mem.out --error={logdir}/bwa_mem.err",
    shell:
        """
        bash snakemake_scripts/bwa_align.sh {input.fasta} {input.fastq} {output.sam}
        """


rule BuildKNNIndices:
    input:
        fasta=rules.FilterContigs.output.fasta,
    # TODO modify this later to list the relevant files within the directories
    output:
        contig_indices=[
            directory(f"{outdir}/knn_indices/{contig}")
            for contig in Config.REFERENCE_CONTIGS
        ],
    threads: workflow.cores
    resources:
        slurm_partition="nvidia-a100",
        runtime=1440,  # 24 hrs
        mem="512GB",
        cpus_per_task=workflow.cores,
        nodes=1,
        slurm_extra=f"--gres=gpu:4 --output={logdir}/build_knn_indices.out --error={logdir}/build_knn_indices.err",
    params:
        use_amp="--use-amp" if Config.USE_AMP else "",
    shell:
        f"""
        mkdir -p {Config.CACHE_DIR}
        singularity exec --nv --bind {outdir}:{outdir},{Config.CACHE_DIR}:{Config.CACHE_DIR} \
        ssearch_gpu.sif python build_knn_index.py \
        --reference-fasta {{input.fasta}} \
        --window-size {Config.WINDOW_SIZE} \
        --stride {Config.STRIDE} \
        --output-dir {outdir} \
        --cache-dir {Config.CACHE_DIR} \
        --base-model {Config.BASE_MODEL} \
        --adapter-checkpoint {Config.ADAPTER_CHECKPOINT} \
        --embedding-dim {Config.EMBEDDING_DIM} \
        --batch-size {Config.BATCH_SIZE} \
        --num-workers-per-gpu {Config.NUM_WORKERS_PER_GPU} \
        --num-gpus {Config.NUM_GPUS} \
        {{params.use_amp}}
        """
