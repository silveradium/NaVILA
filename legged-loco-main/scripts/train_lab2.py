"""Train legged-loco low-level policies on Isaac Lab 2.x (port of scripts/train.py, which targets Isaac Lab 1.1).

  python legged-loco-main/scripts/train_lab2.py --task go2_base --headless --max_iterations 2000
  python legged-loco-main/scripts/train_lab2.py --task go2_vision --headless --num_envs 2048

Checkpoints go to legged-loco-main/logs/rsl_rl/<experiment>/<timestamp>/model_*.pt, which
navila_go2/sim/run_navila_go2.py loads with --task go2_base_play --checkpoint <path>.
"""

import argparse
import os
import os.path as osp
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a legged-loco policy with RSL-RL (Isaac Lab 2.x).")
parser.add_argument("--task", default="go2_base")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="")
parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

LEGGED_LOCO = osp.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, osp.join(LEGGED_LOCO, "isaaclab_exts", "leggedloco"))
import leggedloco  # noqa: E402, F401  (registers go2_base / go2_vision)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = env_cfg.seed = args.seed
    agent_cfg.device = env_cfg.sim.device

    log_dir = osp.join(LEGGED_LOCO, "logs", "rsl_rl", agent_cfg.experiment_name,
                       datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.run_name}" if args.run_name else ""))
    env_cfg.log_dir = log_dir
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args.resume:
        runner.load(args.resume)
    dump_yaml(osp.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(osp.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"[train] {args.task}: {env.num_envs} envs, {agent_cfg.max_iterations} iterations -> {log_dir}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
