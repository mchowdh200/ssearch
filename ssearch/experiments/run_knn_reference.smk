import os
import re
from os.path import basename, dirname
from ssearch.config import KNNReferenceConfig
from ssearch.experiments import knn_reference
from pathlib import Path

outdir = Path(KNNReferenceConfig.OUTPUT_DIR)


def stem(path: str) -> Tuple[str, str]:
    # Two capture groups: first is path stopping at first dot, second is the rest
    return re.match(r"(.+?)(\.[^.]+)+$", basename(path)).groups()[0]


rule All:
    input:
        contig_indices=[
            f"{outdir}/knn_indices/{contig}"
            for contig in KNNReferenceConfig.REFERENCE_CONTIGS
        ],


rule SetupDirectories:
    """
    Create the output and log directories
    """
    output:
        outdir=directory(outdir),
        logdir=directory(f"{outdir}/logs"),
    resources:
        slurm_partition="short",
        runtime=1,  # minutes
        mem="1GB",
        cpus_per_task=1,
        nodes=1,
        output="/dev/null",
        error="logs/setup.err",
    run:
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(f"{outdir}/logs", exist_ok=True)


rule FilterContigs:
    """
    For now, we just get want the main contigs of the reference genome
    """
    input:
        fasta=KNNReferenceConfig.REFERENCE_FASTA,
        outdir=rules.SetupDirectories.output.outdir,
        logdir=rules.SetupDirectories.output.logdir,
    output:
        fasta=f"{outdir}/{stem(KNNReferenceConfig.REFERENCE_FASTA)}_main_contigs.fa",
    resources:
        slurm_partion="short",
        runtime=60,  # minutes
        mem="16GB",
        cpus_per_task=1,
        nodes=1,
        output="logs/filter_contigs.log",
        error="logs/filter_contigs.err",
    run:
        knn_reference.filter_contigs(
            reference_fasta=input.fasta,
            output_fasta=output.fasta,
            keep=KNNReferenceConfig.REFERENCE_CONTIGS,
        )


rule BWAIndex:
    """
    Index the reference genome with BWA
    """
    input:
        fasta=rules.FilterContigs.output.fasta,
    output:
        bwt=f"{outdir}/{stem(KNNReferenceConfig.REFERENCE_FASTA)}_main_contigs.fa.bwt",
        sa=f"{outdir}/{stem(KNNReferenceConfig.REFERENCE_FASTA)}_main_contigs.fa.sa",
        pac=f"{outdir}/{stem(KNNReferenceConfig.REFERENCE_FASTA)}_main_contigs.fa.pac",
        amb=f"{outdir}/{stem(KNNReferenceConfig.REFERENCE_FASTA)}_main_contigs.fa.amb",
        ann=f"{outdir}/{stem(KNNReferenceConfig.REFERENCE_FASTA)}_main_contigs.fa.ann",
        fai=f"{outdir}/{stem(KNNReferenceConfig.REFERENCE_FASTA)}_main_contigs.fa.fai",
    resources:
        slurm_partition="short",
        runtime=240,  # minutes
        mem="64GB",
        cpus_per_task=1,
        nodes=1,
        output="logs/bwa_index.log",
        error="logs/bwa_index.err",
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
        fastq_r1=KNNReferenceConfig.FASTQ_R1,
        fastq_r2=KNNReferenceConfig.FASTQ_R2,
    output:
        bam=f"{outdir}/NA12878.bam",
    threads: workflow.cores
    resources:
        slurm_partition="short",
        runtime=720,  # minutes
        mem="64GB",
        cpus_per_task=workflow.cores,
        nodes=1,
        output="logs/bwa_mem.log",
        error="logs/bwa_mem.err",
    shell:
        """
        bash snakemake_scripts/bwa_align.sh {input.fasta} {input.fastq} {output.sam}
        """


rule BuildKNNIndices:
    input:
        fasta=rules.FilterContigs.output.fasta,
    output:
        contig_indices=[
            directory(f"{outdir}/knn_indices/{contig}")
            for contig in KNNReferenceConfig.REFERENCE_CONTIGS
        ],
    threads: workflow.cores
    resources:
        slurm_partition="nvidia-a100",
        gres="gpu:4",
        runtime=1440,  # 24 hrs
        mem="512GB",
        cpus_per_task=workflow.cores,
        nodes=1,
        output="logs/build_knn_indices.log",
        error="logs/build_knn_indices.err",
    run:
        knn_reference.build_knn_indices(
            reference_fasta=input.fasta,
            window_size=KNNReferenceConfig.WINDOW_SIZE,
            stride=KNNReferenceConfig.STRIDE,
            output_dir=outdir / "knn_indices",
            base_model=KNNReferenceConfig.BASE_MODEL,
            adapter_checkpoint=KNNReferenceConfig.ADAPTER_CHECKPOINT,
            embedding_dim=KNNReferenceConfig.EMBEDDING_DIM,
            batch_size=KNNReferenceConfig.BATCH_SIZE,
            num_workers_per_gpu=KNNReferenceConfig.NUM_WORKERS_PER_GPU,
            num_gpus=KNNReferenceConfig.NUM_GPUS,
            use_amp=KNNReferenceConfig.USE_AMP,
        )
