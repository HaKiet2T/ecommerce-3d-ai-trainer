"""Backward compatibility wrapper for src.data.blender_render"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.blender_render import *

if __name__ == "__main__":
    from src.data.blender_render import main
    main()
