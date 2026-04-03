"""Pytest configuration: ensures project root is on sys.path."""

import sys
from pathlib import Path

# Add project root to path so `from src.xxx import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
