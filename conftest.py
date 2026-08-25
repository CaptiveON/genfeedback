# Makes the repository root importable when running `pytest` directly.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
