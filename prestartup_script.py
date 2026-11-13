"""Prepare the private legacy backend before ComfyUI imports custom nodes."""
import importlib.util
from pathlib import Path
import sys


installer_path = Path(__file__).resolve().with_name("install.py")
try:
    spec = importlib.util.spec_from_file_location("_licon_h3_installer", installer_path)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    # ComfyUI's command-line arguments must never reach the installer.
    installer.main([])
except Exception as error:
    print(f"[Licon H3 INT8 Backend] Automatic setup failed: {error}", file=sys.stderr)
    print(
        "[Licon H3 INT8 Backend] ComfyUI will continue. Native remains available; "
        "Legacy requires a verified backend. Fix the error and restart, or run "
        f"{installer_path} with ComfyUI's Python.",
        file=sys.stderr,
    )
