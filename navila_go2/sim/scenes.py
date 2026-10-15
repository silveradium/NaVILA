"""Indoor scenes for NaVILA vision-language navigation on the simulated Go2.

Each scene lists a default spawn pose and a few language-navigation episodes. An episode's goal region is only used
to score the run (navigation error / success); NaVILA never sees it.
Coordinates are in the scene's world frame (meters, yaw in degrees, counter-clockwise from +X).
"""

from __future__ import annotations

from dataclasses import dataclass, field

NUCLEUS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"


@dataclass
class Episode:
    instruction: str
    goal_xy: tuple[float, float]
    success_radius: float = 1.0
    spawn: tuple[float, float, float, float] | None = None  # (x, y, z, yaw_deg); None -> scene default


@dataclass
class SceneSpec:
    name: str
    usd_path: str
    spawn: tuple[float, float, float, float]  # (x, y, z, yaw_deg)
    floor_z: float | None = 0.0
    episodes: list[Episode] = field(default_factory=list)


SCENES: dict[str, SceneSpec] = {
    # Landmarks (from navila_go2/sim/scout_scene.py): green waiting chairs x[-9,-4.3] y[0,1.2]; U-shaped reception
    # desk x[-0.7,4.5] y[1.8,4.2]; black vending machine (10,2.2) next to the restrooms; staircase x[7.8,10.2]
    # y[10.8,15.2] (bottom at the south end); patient-room corridor y[9,12] with a hospital bed at (-13,11.5).
    "hospital": SceneSpec(
        name="hospital",
        usd_path=f"{NUCLEUS}/Environments/Hospital/hospital.usd",
        spawn=(0.0, 0.0, 0.0, 0.0),
        episodes=[
            Episode(  # 0: short approach
                "Walk forward toward the wooden chairs with green seats and stop right in front of them.",
                goal_xy=(-3.6, 0.5),
                success_radius=1.5,
                spawn=(0.0, 0.0, 0.0, 180.0),
            ),
            Episode(  # 1: straight hallway with a landmark to pass
                "Walk straight down the hall, passing the reception desk on your left. "
                "Stop when you reach the black vending machine.",
                goal_xy=(9.0, 2.0),
                success_radius=1.5,
                spawn=(0.0, 0.0, 0.0, 0.0),
            ),
            Episode(  # 2: turn around first
                "Turn around and walk to the waiting area with the green chairs, then stop.",
                goal_xy=(-3.6, 0.5),
                success_radius=1.5,
                spawn=(0.0, 0.0, 0.0, 0.0),
            ),
            Episode(  # 3: hallway + left turn + stop at stairs
                "Walk past the reception desk, then turn left into the large open hall. "
                "Go straight toward the staircase and stop at the bottom of the stairs.",
                goal_xy=(9.0, 10.0),
                success_radius=1.5,
                spawn=(0.0, 0.0, 0.0, 0.0),
            ),
            Episode(  # 4: long horizon, two landmarks
                "Walk forward toward the stairs. Before you reach the stairs, turn left into the long hallway with "
                "doors on both sides. Walk all the way down the hallway and stop next to the hospital bed at the end.",
                goal_xy=(-11.5, 10.5),
                success_radius=2.0,
                spawn=(6.5, 0.0, 0.0, 90.0),
            ),
        ],
    ),
    # Open-plan office x[-24,6] y[-27,58]. From (-3,6): city-view windows to +X, glass double doors to +Y opening
    # onto a long hallway with a dark carpet runner (x~-4.4, y[12,52]), curved dark reception desk to -Y.
    "office": SceneSpec(
        name="office",
        usd_path=f"{NUCLEUS}/Environments/Office/office.usd",
        spawn=(-3.0, 6.0, 0.0, 90.0),
        episodes=[
            Episode(  # 0: through the doors
                "Walk through the glass doors in front of you and stop in the hallway behind them.",
                goal_xy=(-4.4, 15.0),
                success_radius=2.5,
                spawn=(-3.0, 6.0, 0.0, 90.0),
            ),
            Episode(  # 1: turn first, then the doors
                "Turn left, walk through the glass doors and stop in the hallway.",
                goal_xy=(-4.4, 15.0),
                success_radius=2.5,
                spawn=(-3.0, 6.0, 0.0, 0.0),
            ),
        ],
    ),
    # The scenes below are not scouted yet: spawn poses are unverified and there are no scripted episodes.
    # Use navila_go2/sim/scout_scene.py to pick a spawn, then --spawn x,y,yaw with --instruction or the GUI panel.
    "warehouse": SceneSpec(
        name="warehouse",
        usd_path=f"{NUCLEUS}/Environments/Simple_Warehouse/full_warehouse.usd",
        spawn=(0.0, 0.0, 0.0, 0.0),
    ),
    "warehouse_shelves": SceneSpec(
        name="warehouse_shelves",
        usd_path=f"{NUCLEUS}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
        spawn=(0.0, 0.0, 0.0, 0.0),
    ),
    "simple_room": SceneSpec(
        name="simple_room",
        usd_path=f"{NUCLEUS}/Environments/Simple_Room/simple_room.usd",
        spawn=(0.0, 0.0, 0.0, 0.0),
    ),
}
