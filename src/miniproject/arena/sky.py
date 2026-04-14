from pathlib import Path
from miniproject.utils import miniproject_assets_dir


class SkyMixin:
    def _add_skybox(self):
        self.mjcf_root.asset.add(
            "texture",
            name="skybox",
            type="skybox",
            file=(miniproject_assets_dir / "sky.png").absolute().as_posix(),
            gridsize=(3, 4),
            gridlayout=".U..LFRB.D..",
        )
