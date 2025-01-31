import pyfastx
from pathlib import Path


def filter_contigs(
    reference_fasta: Path | str, keep: set[str], output_fasta: Path | str
):
    """
    Print the fasta entries for the contigs in to file
    """
    fasta = pyfastx.Fasta(
        reference_fasta, build_index=True, key_func=lambda x: x.split()[0]
    )
    with open(output_fasta, "w") as f:
        for seq in fasta:
            if seq.name in keep:
                f.write(seq.raw.strip() + "\n")
