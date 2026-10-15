"""legged-loco `go2_vision` (LiDAR height-map low-level policy), ported from Isaac Lab 1.1 to Isaac Lab 2.x.

Same changes as go2_low_base_cfg.py, plus attach_yaw_only -> ray_alignment for the ray casters.
The Unitree L1 LiDAR on the head (Head_lower, mounted upside down and tilted) feeds a 2.5D height map of the
area around the front legs (mdp.height_map_lidar) so the policy can step over / around obstacles.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ActionsCfg, CommandsCfg, RewardsCfg

import leggedloco.mdp as mdp

from .go2_low_base_cfg import HISTORY_LENGTH, UNITREE_GO2_CFG, Go2BaseRoughEnvCfg, Go2RoughPPORunnerCfg


@configclass
class Go2VisionRoughPPORunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_vision"


##
# Terrains: obstacles to step around / over plus a little roughness
##
Go2_Vision_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.01, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
        "init_pos": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.6,
            num_obstacles=10,
            obstacle_height_mode="fixed",
            obstacle_height_range=(0.3, 2.0),
            obstacle_width_range=(0.4, 1.0),
            platform_width=0.0,
        ),
    },
)
# reset_root_state_from_terrain spawns robots on flat patches stored under the key "init_pos" (every sub-terrain
# must provide it; the original used the sub-terrain name as key, which only worked for the "init_pos" terrain)
for _sub_terrain_cfg in Go2_Vision_TERRAINS_CFG.sub_terrains.values():
    _sub_terrain_cfg.flat_patch_sampling = {
        "init_pos": FlatPatchSamplingCfg(
            num_patches=2, patch_radius=[0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 1.0, 1.2, 1.5], max_height_diff=0.3
        )
    }


##
# Scene
##
@configclass
class Go2VisionSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=Go2_Vision_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.0, 2.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # Unitree L1 on the head, upside down and tilted forward (quaternion from the original config)
    lidar_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Head_lower",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.0), rot=(0.0, -0.991, 0.0, -0.131)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32, vertical_fov_range=(0.0, 90.0), horizontal_fov_range=(-180, 180.0), horizontal_res=4.0
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Rewards
##
@configclass
class CustomGo2RewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.32},
    )
    stand_still_penalty = RewTerm(
        func=mdp.stand_still_penalty,
        weight=-1.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.02,
    )
    joint_power = RewTerm(
        func=mdp.power_penalty,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
        },
    )
    collision = RewTerm(
        func=mdp.collision_penalty,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head.*", ".*_hip", ".*_thigh", ".*_calf"]),
            "threshold": 0.1,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0},
    )
    hip_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip"]), "threshold": 1.0},
    )
    head_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head.*"]), "threshold": 1.0},
    )
    leg_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_calf"]), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0},
    )


##
# Observations
##
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_rpy = ObsTerm(func=mdp.base_rpy, noise=Unoise(n_min=-0.1, n_max=0.1))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_map_lidar,
            params={"sensor_cfg": SceneEntityCfg("lidar_sensor"), "offset": 0.0},
            clip=(-10.0, 10.0),
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_rpy = ObsTerm(func=mdp.base_rpy, noise=Unoise(n_min=-0.1, n_max=0.1))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = HISTORY_LENGTH
            self.flatten_history_dim = True

    @configclass
    class CriticObsCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    critic: CriticObsCfg = CriticObsCfg()


@configclass
class Go2VisionRoughEnvCfg(Go2BaseRoughEnvCfg):
    """Configuration for the Go2 locomotion velocity-tracking environment with LiDAR."""

    scene: Go2VisionSceneCfg = Go2VisionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = CustomGo2RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.04)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.lidar_sensor.update_period = self.sim.dt * self.decimation
        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_from_terrain,
            mode="reset",
            params={
                "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        )
        self.rewards.feet_air_time.weight = 0.02
        self.rewards.action_rate_l2.weight = -0.04
        self.rewards.track_ang_vel_z_exp.weight = 1.5


@configclass
class Go2VisionRoughEnvCfg_PLAY(Go2VisionRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.observations.proprio.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
