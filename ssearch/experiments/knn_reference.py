import argparse
from pathlib import Path

def write_genomic_positions(batch: dict, output_path: str | Path):
    """
    Write genomic positions for each batch element to metadata file.
    """
    chroms = batch["name"].split()[0]
    pos = batch["pos"]
    with open(output_path, "a") as f:
        for chrom, (start, end) in zip(chroms, pos):
            f.write(f"{chrom}\t{start}\t{end}\n")
