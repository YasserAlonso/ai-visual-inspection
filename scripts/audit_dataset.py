"""Convenience wrapper; install the package first with pip install -e ."""

import sys

from visual_inspection.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["audit", *sys.argv[1:]]))
