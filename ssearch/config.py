"""
Global constants used throughout the project
"""

from dataclasses import dataclass, field
from os.path import abspath, dirname
from typing import Optional

from yaml import safe_load


@dataclass(kw_only=True)
class ModelConfig:
    BASE_MODEL: str
    # SEQUENCE_LENGTH: int


@dataclass(kw_only=True)
class TrainingDataConfig:
    TRAIN_DATA: str
    VAL_DATA: str


@dataclass(kw_only=True)
class TrainerConfig:
    DEVICES: int
    EPOCHS: int
    BATCH_SIZE: int
    NUM_WORKERS: int
    LEARNING_RATE: float
    WEIGHT_DECAY: float
    ACCUMULATE_GRAD_BATCHES: int
    SIMILARITY_THRESHOLD: float


@dataclass(kw_only=True)
class LoggingConfig:
    NAME: str
    PROJECT: str
    PROGRESS_BAR: bool
    CHECKPOINT_DIR: str


@dataclass(kw_only=True)
class MetagenomicIndexConfig:
    BASE_MODEL: str
    ADAPTER_CHECKPOINT: str
    OUTPUT_DIR: str
    BATCH_SIZE: int
    NUM_WORKERS_PER_GPU: int
    NUM_GPUS: int
    USE_AMP: bool
    METAGENOMIC_INDEX_DATA: list[str]  # bat metagenomic samples (fastq)
    METAGENOMIC_QUERY_DATA: list[str]  # viral genomes (fasta)
    METADATA_PATH: str
    DISTANCES_PATH: str
    WINDOW_SIZE: int
    STRIDE: int
    K: int  # TODO add this to the metagenomic index args


# TODO add a work directory to the config for low latency access to data like fastqs
@dataclass(kw_only=True)
class KNNReferenceConfig:
    BASE_MODEL: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    ADAPTER_CHECKPOINT: str = (
        "/Users/much8161/Repositories/ssearch/CHECKPOINTS-IA3/epoch=71-val_loss=0.0010/nucleotide-transformer-ia3-ssearch"
    )
    EMBEDDING_DIM: int = 512
    OUTPUT_DIR: str = "/scratch/Shares/layer/projects/ssearch/knn-reference/results"
    CACHE_DIR: str = "/cache/much8161/"  # local to fiji gpu nodes
    BATCH_SIZE: int = 2048
    NUM_WORKERS_PER_GPU: int = 8
    NUM_GPUS: int = 4
    USE_AMP: bool = True
    REFERENCE_FASTA: str = (
        "/scratch/Shares/layer/ref/GRCh38_full_analysis_set_plus_decoy_hla/GRCh38_full_analysis_set_plus_decoy_hla.fa"
    )
    # taken from ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR622/SRR622457/SRR622457_{1|2}.fastq.gz
    FASTQ_R1: str = (
        "/scratch/Shares/layer/projects/ssearch/knn-reference/NA12878/SRR622457_1.fastq.gz"
    )
    FASTQ_R2: str = (
        "/scratch/Shares/layer/projects/ssearch/knn-reference/NA12878/SRR622457_2.fastq.gz"
    )
    WINDOW_SIZE: int = 100  # TODO figure out the read length of samples
    STRIDE: int = 50
    K: int = 10
    REFERENCE_CONTIGS: set[str] = field(
        default_factory=lambda: {str(c) for c in range(1, 23)}
        | {"X", "Y", "MT"}
        | {f"chr{c}" for c in range(1, 23)}
        | {"chrX", "chrY", "chrM"}
    )


@dataclass(kw_only=True)
class PanGenomeReferenceConfig:
    BASE_MODEL: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    ADAPTER_CHECKPOINT: str = (
        "/Users/much8161/Repositories/ssearch/CHECKPOINTS-IA3/epoch=71-val_loss=0.0010/nucleotide-transformer-ia3-ssearch"
    )
    EMBEDDING_DIM: int = 512
    OUTPUT_DIR: str = "/scratch/Shares/layer/projects/ssearch/knn-reference/results"
    CACHE_DIR: str = "/cache/much8161/"  # local to fiji gpu nodes
    BATCH_SIZE: int = 2048
    NUM_WORKERS_PER_GPU: int = 8
    NUM_GPUS: int = 4
    USE_AMP: bool = True
    REFERENCE_FASTA: str = (
        "/scratch/Shares/layer/projects/ssearch/pangenome/bacterialgenomes/M.gnavus.fa"
    )
    # there's other stuff in this directory
    # so we only need the *_R1.fastq.gz and *_R2.fastq.gz files
    FASTQ_DIR: str = "/scratch/Shares/layer/workspace/microbiome"
    WINDOW_SIZE: int = 100  # TODO figure out the read length of samples
    STRIDE: int = 25
    K: int = 100


# TODO do away with this
@dataclass(kw_only=True)
class Config:
    Model: ModelConfig
    TrainingData: TrainingDataConfig
    Trainer: TrainerConfig
    Logging: LoggingConfig
    MetagenomicIndex: (
        MetagenomicIndexConfig  # TODO change this to split into task specific configs
    )
    # KNNReference: KNNReferenceConfig

    @staticmethod
    def from_yaml(config_path: str):
        with open(config_path, "r") as file:
            config = safe_load(file)
        return Config(
            Model=ModelConfig(**config["ModelConfig"]),
            TrainingData=TrainingDataConfig(**config["TrainingDataConfig"]),
            Trainer=TrainerConfig(**config["TrainerConfig"]),
            Logging=LoggingConfig(**config["LoggingConfig"]),
            MetagenomicIndex=MetagenomicIndexConfig(**config["MetagenomicIndexConfig"]),
            # KNNReference=KNNReferenceConfig(**config["KNNReferenceConfig"]),
        )

    def to_dict(self):
        return {
            "ModelConfig": self.Model.__dict__,
            "TrainingDataConfig": self.TrainingData.__dict__,
            "TrainerConfig": self.Trainer.__dict__,
            "LoggingConfig": self.Logging.__dict__,
            "MetagenomicIndexConfig": self.MetagenomicIndex.__dict__,
            # "KNNReferenceConfig": self.KNNReference.__dict__,
        }


DefaultConfig = Config.from_yaml(dirname(abspath(__file__)) + "/config.yaml")
