#!/usr/bin/env python3
"""
PyWeka – Machine Learning Explorer
A Python desktop application inspired by Weka, with the same core ML features.

Usage:
    python app.py                    # Launch GUI
    python app.py dataset.csv        # Launch GUI and load dataset
"""

import sys
import os

# Ensure the app directory is on the path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from ui.main_window import MainWindow


def main():
    app = MainWindow()

    # If a file path was passed as argument, load it
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            try:
                app.dm.load(path)
            except Exception as e:
                print(f"Warning: Could not load '{path}': {e}")

    app.run()


if __name__ == "__main__":
    main()
