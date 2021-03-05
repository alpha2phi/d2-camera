import sys

import fire


def cli():
    if any("--help" in arg for arg in sys.argv):
        print("Type `d2 --help` for usage info.")
        sys.exit()


def main():
    """
    Main function.
    """
    fire.Fire(start)
