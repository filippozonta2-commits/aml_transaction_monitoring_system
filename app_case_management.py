"""Operational Layer V1 launcher.

The portfolio-grade investigation console lives in app.py. Keeping this filename
preserves the V1 launch command while avoiding a second, simplified dashboard.
"""
from pathlib import Path
import runpy

APP = Path(__file__).with_name("app.py")
if not APP.exists():
    raise FileNotFoundError("app.py investigation console not found")
runpy.run_path(str(APP), run_name="__main__")
