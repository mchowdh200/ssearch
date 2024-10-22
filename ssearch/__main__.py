#!/usr/bin/env python
import argparse

from ssearch.training import finetune
from ssearch.scripts import test


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=finetune.main)

    test_parser = subparsers.add_parser("test")
    test_parser.set_defaults(func=test.main)
    test_parser.add_argument("--checkpoint", type=str, required=True)

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
