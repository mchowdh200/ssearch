import argparse
import os
import re
import subprocess
import sys
from functools import partial
from multiprocessing import Pool
from os.path import basename, dirname, exists
from pprint import pprint

from ssearch.config import DefaultConfig


###############################################################################
## Utility functions
###############################################################################
def list2str(l: list[str], delim: str = " ") -> str:
    return delim.join(l)


def job_done(output: list[str]) -> bool:
    for o in output:
        if not exists(o):
            return False
    print(f"Job already done: {output}", file=sys.stderr)
    return True


def delete_incomplete(output: list[str]):
    for o in output:
        if exists(o):
            os.remove(o)


def run_cmds(cmds: list[str], outputs: list[str], processes: int = 1):
    """
    Given a list of shell commands, run them each in their own process.
    If any command fails, delete any outputs and raise an exception.
    """
    pool = Pool(processes)
    results = pool.map(partial(subprocess.run, shell=True), cmds)
    if any([r.returncode != 0 for r in results]):
        print(f"Failed to run command:\n {cmds}", file=sys.stderr)
        print(f"Deleting incomplete outputs", file=sys.stderr)
        delete_incomplete(outputs)
        raise Exception("Failed to run command")


###############################################################################
## Pipeline steps
###############################################################################
def Fastq2Fasta(inputs: list[str], outputs: list[str], processes: int = 1):
    """
    Convert list of fastqs to fasta.
    """
    if job_done(outputs):
        print(f"Fastq2Fasta step already done. Skipping.", file=sys.stderr)
        return

    for o in outputs:
        os.makedirs(dirname(o), exist_ok=True)

    commands = [
        f"""
        bash snakemake_scripts/fastq2fasta.sh {input} {output} 
        """
        for input, output in zip(inputs, outputs)
    ]
    run_cmds(commands, outputs, processes)
    if not job_done(outputs):
        raise Exception("Job complteted but outputs not found")


def ConcatFastas(inputs: list[str], outputs: list[str], processes: int = 1):
    """
    Concatenate paired fastas with (_1 and _2) suffixes.
    """

    if job_done(outputs):
        print(f"ConcatFastas step already done. Skipping.", file=sys.stderr)
        return
    for o in outputs:
        os.makedirs(dirname(o), exist_ok=True)

    fasta1 = sorted([f for f in inputs if f.endswith("_1.fa.gz")])
    fasta2 = sorted([f for f in inputs if f.endswith("_2.fa.gz")])
    assert (
        len(fasta1) == len(fasta2) == len(outputs)
    ), "Number of inputs and outputs must match"

    commands = [
        f"bash snakemake_scripts/concat_fastas.sh {r1} {r2} {o}"
        for r1, r2, o in zip(fasta1, fasta2, sorted(outputs))
    ]
    run_cmds(commands, outputs, processes)
    if not job_done(outputs):
        raise Exception("Job complteted but outputs not found")


def BwaIndex(inputs: list[str], outputs: list[str], processes: int = 1):

    if job_done(outputs):
        print(f"BwaIndex step already done. Skipping.", file=sys.stderr)
        return
    for o in outputs:
        os.makedirs(dirname(o), exist_ok=True)

    commands = [f"bwa index {input}" for input in inputs]
    run_cmds(commands, outputs, processes)
    if not job_done(outputs):
        raise Exception("Job complteted but outputs not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processes",
        type=int,
        help="Number of processes to run in parallel",
        default=1,
    )
    args = parser.parse_args()
    config = DefaultConfig.MetagenomicIndex

    #######################################################################
    inputs = config.METAGENOMIC_INDEX_DATA
    fq_pattern = r"(\.fq\.gz)"
    outputs = [
        f"{config.OUTPUT_DIR}/index_fastas/{basename(dirname(f))}/{re.split(fq_pattern, basename(f))[0]}.fa.gz"
        for f in config.METAGENOMIC_INDEX_DATA
    ]
    Fastq2Fasta(
        inputs=inputs,
        outputs=outputs,
        processes=args.processes,
    )

    #######################################################################
    inputs = outputs[:]
    outputs = [
        re.sub(r"_1\.fa\.gz", r"_combined.fa.gz", f)
        for f in inputs
        if f.endswith("_1.fa.gz")
    ]
    ConcatFastas(
        inputs=inputs,
        outputs=outputs,
        processes=args.processes,
    )

    #######################################################################
    inputs = outputs[:]
    outputs = (
        [f"{f}.amb" for f in inputs]
        + [f"{f}.ann" for f in inputs]
        + [f"{f}.bwt" for f in inputs]
        + [f"{f}.pac" for f in inputs]
        + [f"{f}.sa" for f in inputs]
    )
    BwaIndex(inputs=inputs, outputs=outputs, processes=args.processes)


if __name__ == "__main__":
    main()
