#!/usr/bin/env python
import argparse

from ssearch.training import finetune


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=finetune.main)

    return parser.parse_args()

def main():
    args = parse_args()
    args.func()  # TODO for now, just call the function without any arguments

if __name__ == "__main__":
    main()
