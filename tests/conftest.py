"""Pytest configuration: ensure plugin root and AstrBot root are on sys.path."""

import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
ASTRBOT_ROOT = PLUGIN_ROOT.parents[2]

for path in (str(PLUGIN_ROOT), str(ASTRBOT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
