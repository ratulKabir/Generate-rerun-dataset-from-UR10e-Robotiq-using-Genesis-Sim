from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import copy

@dataclass
class Config:
    outdir: str = "outputs/dataset"
    session: Optional[str] = None

    dt: float = 0.01
    duration: Optional[float] = None
    headless: bool = False

    robot_xml: str = "./assets/xml/universal_robots_ur10e/ur10e_robotiq.xml"
    home_qpose: List[float] = field(default_factory=lambda: [1.57,-1.57,1.57,-1.57,-1.57,0.0])

    hover_in: float = 0.8
    hover_out: float = 0.8
    margin_z: float = 0.28
    dwell_steps: int = 50
    steps_per_segment: int = 150

    add_camera: bool = True
    cam_fps: float = 30.0
    cam_res: List[int] = field(default_factory=lambda: [640,480])
    cam_tilt_deg: float = 60.0
    cam_offset_xyz: List[float] = field(default_factory=lambda: [0.0,0.0,0.2])

    cube_size: List[float] = field(default_factory=lambda: [0.1,0.04,0.05])
    cube_pos: List[float] = field(default_factory=lambda: [-0.65,-0.5,0.0])

    place_xyz: List[float] = field(default_factory=lambda: [-0.30,0.40,0.035])

    streams: List[str] = field(default_factory=lambda: [
        "robot/state/q","robot/state/dq","commands/q_target",
        "gripper/state/width","world/ee","cam/rgb"
    ])

def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: Optional[str], cli_overrides: dict) -> Config:
    # start with dataclass defaults
    base = Config().__dict__
    if path:
        with open(path, "r") as f:
            yml = yaml.safe_load(f) or {}
        base = _deep_update(base, yml)

    # apply CLI overrides (only if not None)
    clean_cli = {k: v for k, v in cli_overrides.items() if v is not None}
    merged = _deep_update(base, clean_cli)

    # materialize dataclass
    return Config(**merged)
