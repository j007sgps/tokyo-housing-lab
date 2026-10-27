"""命令列入口。子命令（fetch / load / query）會在之後的步驟加上。"""

import argparse

from tokyo_housing_lab import DISCLAIMER


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tokyo-housing-lab", epilog=DISCLAIMER)
    parser.parse_args(argv)
    parser.print_help()
