"""Filesystem locations only; no fair-v2 imports or experimental settings."""
import os
from pathlib import Path


def get_project_dir():
    """Original pred data root, independent of the invocation directory."""
    default = Path(__file__).resolve().parents[2]
    return str(Path(os.environ.get("FINDER_PROJECT_DIR", default)).expanduser().resolve()) + os.sep


def get_data_dir():
    """Resolve processed FINDER inputs relative to the code repository root."""
    project_root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("FINDER_DATA_DIR")
    if configured:
        data_dir = Path(configured).expanduser()
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
    else:
        data_dir = project_root.parent / "data" / "inputs" / "finder"
    return data_dir.resolve()


def get_output_dir():
    """Separate output location; never default to archived experiment outputs."""
    default = Path(__file__).resolve().parents[2] / "runs" / "baseline_comparison"
    return str(Path(os.environ.get("FINDER_OUTPUT_DIR", default)).expanduser().resolve()) + os.sep
