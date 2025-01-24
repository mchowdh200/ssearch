import argparse
import itertools
import os
import re
import subprocess
import sys
from functools import partial
from multiprocessing import Pool
from os.path import basename, dirname, exists, splitext
from pprint import pprint
from typing import Callable

from ssearch.config import DefaultConfig
from ssearch.experiments.snakemake_scripts.make_windowed_fasta import \
    make_windowed_fasta


###############################################################################
## Utility functions
###############################################################################
def list2str(l: list[str], delim: str = " ") -> str:
    return delim.join(l)


def bname_no_ext(f: str) -> str:
    return basename(f).split(".")[0]


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


def run_shell_cmds(cmds: list[str], outputs: list[str], processes: int = 1):
    """
    Given a list of shell commands, run them each in their own process.
    If any command fails, delete any outputs and raise an exception.
    """
    pool = Pool(processes)
    results = pool.map(partial(subprocess.run, shell=True), cmds)
    # if any([r.returncode != 0 for r in results]):
    #     print(f"Failed to run command:\n {cmds}", file=sys.stderr)
    #     print(f"Deleting incomplete outputs", file=sys.stderr)
    #     delete_incomplete(outputs)
    #     raise Exception("Failed to run command")


def run_func(func: Callable[..., None], args: dict):
    func(**args)


def run_parallel_func(
    func: Callable[..., None], args: list[dict], outputs: list[str], processes: int = 1
):
    """
    Given a python function and a list of arguments, run the function with each
    set of args in parallel.
    """
    pool = Pool(processes)
    try:
        _ = pool.map(partial(run_func, func), args)
    except Exception as e:
        print(f"Failed to run python function:\n {func}", file=sys.stderr)
        print(f"Deleting incomplete outputs", file=sys.stderr)
        delete_incomplete(outputs)
        raise e


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
    run_shell_cmds(commands, outputs, processes)
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
    run_shell_cmds(commands, outputs, processes)
    if not job_done(outputs):
        raise Exception("Job complteted but outputs not found")


def BwaIndex(inputs: list[str], outputs: list[str], processes: int = 1):

    if job_done(outputs):
        print(f"BwaIndex step already done. Skipping.", file=sys.stderr)
        return
    for o in outputs:
        os.makedirs(dirname(o), exist_ok=True)

    commands = [f"bwa index {input}" for input in inputs]
    run_shell_cmds(commands, outputs, processes)
    if not job_done(outputs):
        raise Exception("Job complteted but outputs not found")


def MakeWindowedFastas(
    inputs: list[str],
    outputs: list[str],
    window_size: int,
    stride: int,
    processes: int = 1,
):
    if job_done(outputs):
        print(f"MakeWindowedFastas step already done. Skipping.", file=sys.stderr)
        return
    for o in outputs:
        os.makedirs(dirname(o), exist_ok=True)

    samples = [splitext(basename(f))[0] for f in inputs]

    args = [
        {
            "fasta": f,
            "sample": s,
            "window_size": window_size,
            "stride": stride,
            "output": o,
        }
        for f, s, o in zip(inputs, samples, outputs)
    ]
    run_parallel_func(make_windowed_fasta, args, outputs, processes)
    if not job_done(outputs):
        raise Exception("Job complteted but outputs not found")


def BwaAlign(
    input_queries: list[str],
    input_refs: list[str],
    outputs: list[str],
    processes: int = 1,
):
    if job_done(outputs):
        print(f"BwaAlign step already done. Skipping.", file=sys.stderr)
        return
    for o in outputs:
        os.makedirs(dirname(o), exist_ok=True)

    query_ref_pairs = list(itertools.product(input_queries, input_refs))
    assert len(outputs) == len(query_ref_pairs), "Number of outputs must match inputs"

    bwa_processes = 8
    bwa_threads = max(1, processes // bwa_processes)

    commands = [
        f"bash snakemake_scripts/bwa_align.sh -q {query} -r {ref} -o {output} -t {bwa_threads}"
        for (query, ref), output in zip(
            query_ref_pairs,
            outputs,
        )
    ]
    run_shell_cmds(commands, outputs, bwa_processes)
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
    fq2fa_inputs = config.METAGENOMIC_INDEX_DATA
    fq_pattern = r"(\.fq\.gz)"
    fq2fa_outputs = [
        f"{config.OUTPUT_DIR}/index_fastas/{basename(dirname(f))}/{re.split(fq_pattern, basename(f))[0]}.fa.gz"
        for f in config.METAGENOMIC_INDEX_DATA
    ]
    Fastq2Fasta(
        inputs=fq2fa_inputs,
        outputs=fq2fa_outputs,
        processes=args.processes,
    )

    #######################################################################
    concat_inputs = fq2fa_outputs[:]
    concat_outputs = [
        re.sub(r"_1\.fa\.gz", r"_combined.fa.gz", f)
        for f in concat_inputs
        if f.endswith("_1.fa.gz")
    ]
    ConcatFastas(
        inputs=concat_inputs,
        outputs=concat_outputs,
        processes=args.processes,
    )

    #######################################################################
    bwa_index_inputs = concat_outputs[:]
    bwa_index_outputs = (
        [f"{f}.amb" for f in bwa_index_inputs]
        + [f"{f}.ann" for f in bwa_index_inputs]
        + [f"{f}.bwt" for f in bwa_index_inputs]
        + [f"{f}.pac" for f in bwa_index_inputs]
        + [f"{f}.sa" for f in bwa_index_inputs]
    )
    BwaIndex(
        inputs=bwa_index_inputs,
        outputs=bwa_index_outputs,
        processes=args.processes,
    )

    #######################################################################
    win_fa_inputs = config.METAGENOMIC_QUERY_DATA
    win_fa_outputs = [
        f"{config.OUTPUT_DIR}/windowed_query_fasta/{basename(f)}.windowed.fa"
        for f in win_fa_inputs
    ]
    MakeWindowedFastas(
        inputs=win_fa_inputs,
        outputs=win_fa_outputs,
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE,
        processes=args.processes,
    )

    #######################################################################
    input_queries = win_fa_outputs[:]
    input_refs = bwa_index_inputs[:]
    bwa_align_outputs = [
        f"{config.OUTPUT_DIR}/bwa_alignments/{bname_no_ext(q)}_{bname_no_ext(r)}.bam"
        for q, r in itertools.product(input_queries, input_refs)
    ]
    BwaAlign(
        input_queries=input_queries,
        input_refs=input_refs,
        outputs=bwa_align_outputs,
        processes=args.processes,
    )


if __name__ == "__main__":
    main()
