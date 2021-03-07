import sys
import fire

from .d2 import video_capture


def cli():
    """
    CLI entry point.

    """
    if any("--help" in arg for arg in sys.argv):
        print("Type `d2 --help` for usage info.")
        sys.exit()

    video_capture()


def main():
    """
    Main function.
    """
    fire.Fire(cli)
