"""
Global constants used throughout the project
"""

from dataclasses import dataclass
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
    K: int  # TODO add this to the metagenomic index args

    @staticmethod
    def from_yaml(config_path: str):
        with open(config_path, "r") as file:
            config = safe_load(file)
        return MetagenomicIndexConfig(**config)


@dataclass(kw_only=True)
class KNNReferenceConfig:
    BASE_MODEL: str
    ADAPTER_CHECKPOINT: str
    OUTPUT_DIR: str
    BATCH_SIZE: int
    NUM_WORKERS_PER_GPU: int
    NUM_GPUS: int
    USE_AMP: bool
    REFERENCE_FASTA: str
    QUERY_FASTAS: list[str]
    WINDOW_SIZE: int
    STRIDE: int
    K: int


@dataclass(kw_only=True)
class Config:
    Model: ModelConfig
    TrainingData: TrainingDataConfig
    Trainer: TrainerConfig
    Logging: LoggingConfig
    MetagenomicIndex: (
        MetagenomicIndexConfig  # TODO change this to split into task specific configs
    )
    KNNReference: KNNReferenceConfig

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
            KNNReference=KNNReferenceConfig(**config["KNNReferenceConfig"]),
        )

    def to_dict(self):
        return {
            "ModelConfig": self.Model.__dict__,
            "TrainingDataConfig": self.TrainingData.__dict__,
            "TrainerConfig": self.Trainer.__dict__,
            "LoggingConfig": self.Logging.__dict__,
            "MetagenomicIndexConfig": self.MetagenomicIndex.__dict__,
            "KNNReferenceConfig": self.KNNReference.__dict__,
        }


DefaultConfig = Config.from_yaml(dirname(abspath(__file__)) + "/config.yaml")
