import argparse

import intervaltree
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_bed", help="BED file with scores")
    parser.add_argument("--output", help="Output plot")
    return parser.parse_args()


def smooth_data(x, window_size):
    """
    Smooth the data using a moving average filter
    """
    return np.convolve(
        x,
        np.ones((window_size,)) / window_size,
        mode="same",
    )


def get_sample_names(scores_bed: str) -> set[str]:
    """
    Use "chromosome" names as sample names
    """
    samples = set()
    for line in open(scores_bed):
        chrom, *_ = line.strip().split()
        samples.add(chrom)
    return samples


def load_scores(scores_bed: str) -> dict[str, intervaltree.IntervalTree]:
    regions = {}
    with open(scores_bed) as f:
        for line in f:
            chrom, start, end, score = line.strip().split()
            start, end, score = int(start), int(end), float(score)
            if chrom not in regions:
                regions[chrom] = intervaltree.IntervalTree()
            regions[chrom].addi(start, end, score)
    return regions


def make_bins(start, end, step) -> intervaltree.IntervalTree:
    interval_bins = intervaltree.IntervalTree()
    for i in range(start, end, step):
        interval_bins.addi(i, i + step, [])
    return interval_bins


def make_plot(scores_bed: str, output: str):
    all_scores = load_scores(scores_bed)
    labels = []

    average_scores = {}
    for sample in get_sample_names(scores_bed):
        interval_bins = make_bins(-250, 29_001, 250)
        scores = all_scores[sample]
        average_scores[sample] = np.mean([1-np.sqrt(i.data) / 2 for i in scores])

        for i in scores:
            ovlps = interval_bins.overlap(i)
            for o in ovlps:
                o.data.append(i.data)

        bins = sorted(
            [(i.begin, i.end, np.mean(i.data)) for i in interval_bins if i.data]
        )

        labels.append(f"{sample} (genome-wide average = {average_scores[sample]:.2f})")
        plt.plot(
            [x[0] for x in bins],
            # smooth_data([x[2] for x in bins], window_size=50),
            smooth_data([1 - np.sqrt(x[2]) / 2 for x in bins], window_size=10),
            # [1 - np.sqrt(x[2]) / 2 for x in bins],
            label=f"{sample} (genome-wide average = {average_scores[sample]:.2f})",
            linewidth=0.75,
        )

    plt.ylim(.8, 1.0) # todo make into args
    plt.legend(labels)
    plt.xlabel("Genomic position")
    plt.ylabel("Similarity")
    plt.savefig(output)


if __name__ == "__main__":
    args = parse_args()
    make_plot(args.scores_bed, args.output)
