from importlib.resources import files as _importlib_resources_files
from pathlib import Path as _Path

miniproject_assets_dir = _Path(
    str(_importlib_resources_files("miniproject") / "assets")
)
