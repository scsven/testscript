#!/usr/bin/env python3

from milvus import __version__

from testscript.test import Test
import logging

logging.basicConfig(level=logging.DEBUG)


def main():
    print(f"Run on pymilvus v{__version__}")
    test = Test(100 * 10000)
    print(test.run())


if __name__ == "__main__":
    main()
