#!/usr/bin/env python
import argparse

from ssearch.training import finetune
from ssearch.scripts import test


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=finetune.main)


    return parser.parse_args()

def main():
    args = parse_args()
    func = args.func

    match func:
        case finetune.main:
            func()
        case test.main:
            func(args.checkpoint)

if __name__ == "__main__":
    main()
