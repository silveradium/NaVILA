"""MDP terms for legged-loco on Isaac Lab 2.x (Isaac Lab's own terms + the legged-loco specific ones)."""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
