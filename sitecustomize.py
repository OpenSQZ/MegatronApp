"""Ensure the src/ directory is on sys.path when running from the repo root."""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

