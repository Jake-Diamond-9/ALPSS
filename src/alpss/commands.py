import argparse

from alpss.alpss_watcher import Watcher
from alpss.alpss_main import alpss_main

"""
Credit to Michael Cho
https://michaelcho.me/article/using-pythons-watchdog-to-monitor-changes-to-a-directory
"""


def start_watcher():
    w = Watcher()
    w.run()


def run_alpss():
    parser = argparse.ArgumentParser(description="Run ALPSS on a file")
    parser.add_argument(
        "filename",
        type=str,
        help="The name of the file to run ALPSS on",
    )
    args = parser.parse_args()
    alpss_main(filename=args.filename)
