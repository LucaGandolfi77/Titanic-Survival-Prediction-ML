import sys
import logging
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

import tkinter as tk
from gui.app import Application

if __name__ == "__main__":
    root = tk.Tk()
    Application(root)
    root.mainloop()
