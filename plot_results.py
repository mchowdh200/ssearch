import argparse

import intervaltree
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_bed", help="BED file with scores")
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


def make_plot(scores_bed: str):
    all_scores = load_scores(scores_bed)
    labels = []

    for sample in get_sample_names(scores_bed):
        interval_bins = make_bins(0, 30_000, 50)
        scores = all_scores[sample]

        for i in scores:
            ovlps = interval_bins.overlap(i)
            for o in ovlps:
                o.data.append(i.data)

        bins = sorted(
            [(i.begin, i.end, np.mean(i.data)) for i in interval_bins if i.data]
        )

        labels.append(sample)
        plt.plot(
            [x[0] for x in bins],
            smooth_data([1 - np.sqrt(x[2]) / 2 for x in bins], window_size=20),
            label=sample,
        )

    plt.ylim(.7, .85)
    plt.legend(labels)
    plt.xlabel("Genomic position")
    plt.ylabel("Mean distance")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    make_plot(args.scores_bed)
