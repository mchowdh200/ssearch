import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a windowed fasta file from a fasta with one sequence"
    )
    parser.add_argument("--fasta", type=str, help="Input fasta file")
    parser.add_argument("--sample", type=str, help="Sample name")
    parser.add_argument("--window-size", type=int, help="Window size")
    parser.add_argument("--step-size", type=int, help="Step size")
    parser.add_argument("--output", type=str, help="Output fasta file")
    return parser.parse_args()


def make_windowed_fasta(fasta, sample, window_size, step_size, output):
    with open(fasta) as f:
        # first is line header, the rest is sequence
        lines = f.readlines()
        seq = "".join([line.strip() for line in lines[1:]])
        seq_len = len(seq)
        with open(output, "w") as out:
            for i in range(0, seq_len - window_size + 1, step_size):
                window = seq[i : i + window_size]
                out.write(f">{sample}:{i}-{i + window_size}\n")
                out.write(window + "\n")


if __name__ == "__main__":
    args = parse_args()
    make_windowed_fasta(
        fasta=args.fasta,
        sample=args.sample,
        window_size=args.window_size,
        step_size=args.step_size,
        output=args.output,
    )
