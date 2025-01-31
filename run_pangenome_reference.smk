import os
import shutil
from os.path import basename, dirname
from ssearch.config import PangenomeConfigReferenceConfig
from pathlib import Path
import re

Config = PangenomeConfigReferenceConfig()
outdir = Path(Config.OUTPUT_DIR)
logdir = Path(Config.LOG_DIR)


def stem(path: str) -> tuple[str, str]:
    # Two capture groups: first is path stopping at first dot, second is the rest
    return re.match(r"(.+?)(\.[^.]+)+$", basename(path)).groups()[0]


rule All:
    input:
        index=outdir / f"{stem(Config.REFERENCE_FASTA)}_knn.faiss",


rule BuildPangenomeKNNIndex:
    input:
        fasta=Config.REFERENCE_FASTA,
    output:
        index=outdir / f"{stem(Config.REFERENCE_FASTA)}_knn.faiss",
    threads: worklfow.cores
    resources:
        slurm_partition="nvidia-a100",
        runtime=1440,  # 24 hrs
        mem="512GB",
        cpus_per_task=workflow.cores,
        nodes=1,
        slurm_extra=f"--gres=gpu:4 --output={logdir}/build_pangenome_index.out --error={logdir}/build_pangenome_index.err",
    params:
        use_amp="--use-amp" if Config.USE_AMP else "",
    shell:
        f"""
        mkdir -p {Config.CACHE_DIR}
        singularity exec --nv --bind {outdir}:{outdir},{Config.CACHE_DIR}:{Config.CACHE_DIR} {Config.SINGULARITY_IMAGE} \
            build_pangenome_knn_index.py \
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
                --num-gpus {workflow.cores} \
                {{params.use_amp}}
        """
