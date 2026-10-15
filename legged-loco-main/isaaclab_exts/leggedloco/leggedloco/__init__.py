"""legged-loco low-level locomotion for Isaac Lab 2.x (port of omni.isaac.leggedloco, Isaac Lab 1.1).

Importing this package registers the gym tasks go2_base, go2_base_play, go2_vision and go2_vision_play.
Only the Go2 configs (the robot NaVILA runs on) are ported; go1/h1/g1 remain in the original extension.
"""

from . import config  # noqa: F401
