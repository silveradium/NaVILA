"""Mid-level bridge: NaVILA language actions -> body velocity commands for a locomotion policy.

NaVILA emits mid-level actions in natural language ("The next action is move forward 75 cm.", "... turn left
30 degrees.", "... stop."). Following the NaVILA paper, each one is executed as a fixed-speed velocity command
(0.5 m/s forward, pi/6 rad/s turning). The legged locomotion policy (legged-loco) consumes (vx, vy, wz)
and outputs joint targets.

Unlike a purely time-based execution, :class:`MidLevelController` closes the loop on odometry, so the robot
travels the requested distance/angle even when the policy tracks the commanded velocity imperfectly.

This module is pure Python and shared by the Isaac Sim runner and the real Go2 (go2_sdk) client.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

FORWARD, TURN_LEFT, TURN_RIGHT, STOP = "forward", "turn_left", "turn_right", "stop"

# Discretisation NaVILA was trained with (VLN-CE / Habitat): 25 cm forward steps, 15 degree turns.
FORWARD_UNIT_M = 0.25
TURN_UNIT_DEG = 15.0

_PATTERNS = {
    STOP: re.compile(r"\bstop\b", re.IGNORECASE),
    FORWARD: re.compile(r"\bmove forward\b", re.IGNORECASE),
    TURN_LEFT: re.compile(r"\bturn left\b", re.IGNORECASE),
    TURN_RIGHT: re.compile(r"\bturn right\b", re.IGNORECASE),
}


@dataclass
class MidLevelAction:
    kind: str  # one of FORWARD, TURN_LEFT, TURN_RIGHT, STOP
    value: float = 0.0  # meters for FORWARD, degrees for turns
    text: str = ""

    def __str__(self) -> str:
        if self.kind == FORWARD:
            return f"forward {self.value * 100:.0f} cm"
        if self.kind in (TURN_LEFT, TURN_RIGHT):
            return f"{self.kind.replace('_', ' ')} {self.value:.0f} deg"
        return "stop"


def parse_navila_output(text: str, snap: bool = False) -> MidLevelAction:
    """Parse NaVILA's text output.

    Mirrors the regexes in evaluation/vlnce_baselines/navila_trainer.py (same defaults on failure: 25 cm / 15 deg).
    With ``snap=True`` values are snapped to the Habitat grid ({25,50,75} cm, {15,30,45} deg) as in the benchmark;
    by default the exact value is kept since a legged robot can execute continuous distances.
    """
    kind = next((k for k, p in _PATTERNS.items() if p.search(text)), None)
    if kind is None:
        # navila_trainer.py falls back to "move forward" when nothing matches
        return MidLevelAction(FORWARD, FORWARD_UNIT_M, text)
    if kind == STOP:
        return MidLevelAction(STOP, 0.0, text)
    if kind == FORWARD:
        m = re.search(r"move forward (\d+(?:\.\d+)?)\s*(cm|centimet|m\b|meter)?", text, re.IGNORECASE)
        dist = FORWARD_UNIT_M
        if m:
            dist = float(m.group(1)) / (1.0 if (m.group(2) or "cm").lower().startswith("m") else 100.0)
        if snap:
            dist = min([0.25, 0.5, 0.75], key=lambda x: abs(x - dist))
        return MidLevelAction(FORWARD, dist, text)
    m = re.search(r"turn (?:left|right) (\d+(?:\.\d+)?)\s*degree", text, re.IGNORECASE)
    deg = float(m.group(1)) if m else TURN_UNIT_DEG
    if snap:
        deg = min([15.0, 30.0, 45.0], key=lambda x: abs(x - deg))
    return MidLevelAction(kind, deg, text)


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class MidLevelConfig:
    lin_speed: float = 0.5  # m/s   (NaVILA paper)
    ang_speed: float = math.pi / 6  # rad/s (NaVILA paper)
    closed_loop: bool = True  # False -> paper-style open loop: run for value/speed seconds
    min_lin_speed: float = 0.15  # slow-down floor near the target (closed loop)
    min_ang_speed: float = 0.15
    lin_gain: float = 2.0  # proportional slow-down: v = clip(gain * remaining, min, max)
    ang_gain: float = 2.0
    heading_hold_gain: float = 1.0  # keeps the robot straight while moving forward
    dist_tol: float = 0.03
    ang_tol: float = math.radians(2.0)
    timeout_factor: float = 3.0  # abort after factor * nominal duration + 2 s (e.g. blocked by a wall)


class MidLevelController:
    """Turns one :class:`MidLevelAction` into a stream of (vx, vy, wz) commands until it is completed.

    Call :meth:`start` with the action and the current planar pose (x, y, yaw), then :meth:`step` every control tick.
    ``units_done`` counts completed 25 cm / 15 deg units so callers can record observation frames at the same
    granularity NaVILA's training data had.
    """

    def __init__(self, cfg: MidLevelConfig | None = None):
        self.cfg = cfg or MidLevelConfig()
        self.action: MidLevelAction | None = None
        self.active = False
        self.timed_out = False
        self.units_done = 0
        self.heading_ref: float | None = None

    def reset_heading(self, yaw: float | None = None):
        """Start of an episode: the heading reference restarts from the robot's current yaw."""
        self.heading_ref = yaw

    def start(self, action: MidLevelAction, pose: tuple[float, float, float]):
        self.action = action
        self.active = action.kind != STOP
        self.timed_out = False
        self.units_done = 0
        self._start_pose = pose
        self._elapsed = 0.0
        self._last_yaw = pose[2]
        self._turned = 0.0  # accumulated signed yaw (handles wrap-around)
        self._forward_progress = 0.0
        # Like Habitat's discrete agent, headings are exact multiples of the commanded turns: turns move the
        # reference by exactly +-angle and forward motion holds it, so small tracking errors do not accumulate.
        if self.heading_ref is None:
            self.heading_ref = pose[2]
        nominal = 0.0
        if action.kind == FORWARD:
            nominal = action.value / self.cfg.lin_speed
        elif action.kind in (TURN_LEFT, TURN_RIGHT):
            sign = 1.0 if action.kind == TURN_LEFT else -1.0
            residual = wrap_to_pi(self.heading_ref - pose[2])  # heading error left over from earlier actions
            # amount to turn in the commanded direction, tracked by integrating yaw (unambiguous even near 180 deg)
            self._turn_target = math.radians(action.value) + sign * residual
            self.heading_ref = wrap_to_pi(self.heading_ref + sign * math.radians(action.value))
            nominal = max(self._turn_target, 0.0) / self.cfg.ang_speed
        self._nominal = nominal
        self._timeout = self.cfg.timeout_factor * nominal + 2.0

    @property
    def progress(self) -> float:
        """Distance traveled (m) for forward, absolute angle turned (deg) for turns."""
        if self.action is None:
            return 0.0
        if self.action.kind == FORWARD:
            return self._forward_progress
        return abs(math.degrees(self._turned))

    def step(self, pose: tuple[float, float, float], dt: float) -> tuple[float, float, float]:
        if not self.active:
            return 0.0, 0.0, 0.0
        c, a = self.cfg, self.action
        self._elapsed += dt
        self._turned += wrap_to_pi(pose[2] - self._last_yaw)
        self._last_yaw = pose[2]
        x0, y0, _ = self._start_pose
        dx, dy = pose[0] - x0, pose[1] - y0
        ref = self.heading_ref
        self._forward_progress = dx * math.cos(ref) + dy * math.sin(ref)

        if a.kind == FORWARD:
            remaining = a.value - self._forward_progress
            self.units_done = int(max(self._forward_progress, 0.0) / FORWARD_UNIT_M + 1e-6)
            if c.closed_loop:
                done = remaining <= c.dist_tol
                vx = min(c.lin_speed, max(c.min_lin_speed, c.lin_gain * remaining))
                wz = c.heading_hold_gain * wrap_to_pi(ref - pose[2])
            else:
                done = self._elapsed >= self._nominal
                vx, wz = c.lin_speed, 0.0
            cmd = (vx, 0.0, max(-c.ang_speed, min(c.ang_speed, wz)))
        else:
            sign = 1.0 if a.kind == TURN_LEFT else -1.0
            turned = sign * self._turned
            self.units_done = int(max(math.degrees(turned), 0.0) / TURN_UNIT_DEG + 1e-6)
            if c.closed_loop:
                remaining = self._turn_target - turned
                done = remaining <= c.ang_tol
                wz = min(c.ang_speed, max(c.min_ang_speed, c.ang_gain * remaining))
            else:
                done = self._elapsed >= self._nominal
                wz = c.ang_speed
            cmd = (0.0, 0.0, sign * wz)

        if self._elapsed > self._timeout:
            self.timed_out = True
            done = True
        if done:
            self.active = False
            return 0.0, 0.0, 0.0
        return cmd
