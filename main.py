#! /usr/bin/env python3

from milvus import __version__
from testscript.test import Test
from testscript.report import Report
import logging
import argparse


def main(args):
    print(f"Run on pymilvus v{__version__}")
    t = Test(nvec=args.nvec)
    report = Report(t.run(suite={
        "nq": [1,10,100,1000],
        "topk": [1,10,100,1000],
        "nprobe": [10],
        }))
    print(report.dump())
    if args.output:
        print(f"Output report to {args.output}")
        report.file(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nvec", type=int, help="The number of vectors in database")
    parser.add_argument("-o", "--output", help="Save report to specify file")
    parser.add_argument("-d", "--debug", help="Debugging mode", action="store_true")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    main(args)
