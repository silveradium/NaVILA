"""Replay one RxR episode through NaVILA and render predicted vs. ground-truth actions.

For every step of the episode the model sees the same 8 frames it saw in training
(7 sampled history frames + current observation) and predicts the next action.
Steps follow the ground-truth trajectory (teacher forcing), so a wrong prediction
does not change what the model sees next.

Outputs (in --out-dir/<episode>/):
  step_XXX.png   one panel per step: current view with GT/pred arrows, instruction, model inputs
  episode.mp4    all panels as a video
  summary.txt    per-step predictions and match stats

Examples (run from the repo root with the `navila` env active):
  python scripts/inference/visualize_rxr.py                    # random episode
  python scripts/inference/visualize_rxr.py --episode 23289    # specific episode
"""

import argparse
import json
import math
import os
import os.path as osp
import random
import re
import textwrap
import time
from collections import defaultdict

import cv2
import numpy as np

# Same directory, so this also applies quick_infer's transformers compatibility shim
from quick_infer import REPO_ROOT, action_type, load_model, normalize_instruction, predict
from llava.mm_utils import vlnce_frame_sampling

GT_COLOR = (60, 200, 60)  # BGR green
PRED_COLOR = (255, 60, 255)  # BGR magenta
TYPE_COLOR = (0, 200, 255)  # BGR orange: same action type, different amount
BAD_COLOR = (60, 60, 255)  # BGR red
TEXT_COLOR = (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX

VIEW = 512  # current observation size
THUMB = 150  # model input thumbnails
WIDTH = 1280
HEIGHT = VIEW + THUMB + 70


def parse_action(text):
    """Return (kind, value) with kind in {'left', 'right', 'forward', 'stop', 'unknown'}."""
    text = text.lower()
    if "stop" in text:
        return "stop", 0
    match = re.search(r"(turn left|turn right|move forward)\s+(\d+)", text)
    if not match:
        return "unknown", 0
    kind = {"turn left": "left", "turn right": "right", "move forward": "forward"}[match.group(1)]
    return kind, int(match.group(2))


def short_label(text):
    kind, value = parse_action(text)
    return {"left": f"L {value}deg", "right": f"R {value}deg", "forward": f"F {value}cm"}.get(kind, kind.upper())


def verdict_color(pred, gt):
    if pred.strip() == gt.strip():
        return GT_COLOR
    return TYPE_COLOR if action_type(pred) == action_type(gt) else BAD_COLOR


def draw_action(img, text, color, thickness, stop_offset):
    """Draw an action arrow from the bottom-centre of the view (up = forward)."""
    kind, value = parse_action(text)
    if kind == "stop":
        center = (VIEW // 2 + stop_offset, VIEW // 2)
        cv2.circle(img, center, 48, color, thickness, cv2.LINE_AA)
        cv2.putText(img, "STOP", (center[0] - 34, center[1] + 8), FONT, 0.75, color, 2, cv2.LINE_AA)
        return
    if kind == "unknown":
        return
    if kind == "forward":
        angle, length = 0, 60 + value * 2  # 25/50/75 cm -> 110/160/210 px
    else:
        angle, length = (-value if kind == "left" else value), 150
    origin = (VIEW // 2, VIEW - 30)
    rad = math.radians(angle)
    tip = (int(origin[0] + length * math.sin(rad)), int(origin[1] - length * math.cos(rad)))
    cv2.arrowedLine(img, origin, tip, color, thickness, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(img, short_label(text), (tip[0] + 8, tip[1]), FONT, 0.6, color, 2, cv2.LINE_AA)


def render_step(episode, step_idx, steps, results, inputs):
    pred, gt = results[step_idx]
    canvas = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)

    # Current observation with GT (thick, underneath) and prediction (thin, on top)
    view = cv2.resize(cv2.cvtColor(np.array(inputs[-1]), cv2.COLOR_RGB2BGR), (VIEW, VIEW))
    draw_action(view, gt, GT_COLOR, 6, stop_offset=-60)
    draw_action(view, pred, PRED_COLOR, 3, stop_offset=60)
    canvas[:VIEW, :VIEW] = view

    # Text panel
    x, color = VIEW + 20, verdict_color(pred, gt)
    verdict = {GT_COLOR: "EXACT MATCH", TYPE_COLOR: "SAME ACTION TYPE", BAD_COLOR: "MISMATCH"}[color]
    cv2.putText(canvas, f"RxR episode {episode}   step {step_idx + 1}/{len(steps)}", (x, 32), FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"GT:   {gt}", (x, 72), FONT, 0.6, GT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Pred: {pred}", (x, 102), FONT, 0.6, PRED_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, verdict, (x, 136), FONT, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Instruction:", (x, 176), FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    lines = textwrap.wrap(normalize_instruction(steps[step_idx]["q"]), width=92)
    if len(lines) > 15:
        lines = lines[:14] + [lines[14][:88] + " ..."]
    for k, line in enumerate(lines):
        cv2.putText(canvas, line, (x, 200 + 20 * k), FONT, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    # Model inputs: 7 sampled history frames + current observation
    y0 = VIEW + 10
    for k, img in enumerate(inputs):
        thumb = cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (THUMB, THUMB))
        x0 = 10 + k * (THUMB + 8)
        canvas[y0 : y0 + THUMB, x0 : x0 + THUMB] = thumb
        label = "current" if k == len(inputs) - 1 else f"hist {k + 1}"
        cv2.putText(canvas, label, (x0 + 4, y0 + 18), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Timeline: one cell per step, current step outlined
    ty, cell = HEIGHT - 40, (WIDTH - 20) / len(steps)
    for k, (p, g) in enumerate(results):
        cv2.rectangle(canvas, (int(10 + k * cell), ty), (int(10 + (k + 1) * cell) - 1, ty + 22), verdict_color(p, g), -1)
    cx = int(10 + step_idx * cell)
    cv2.rectangle(canvas, (cx - 2, ty - 4), (int(cx + cell) + 1, ty + 26), (255, 255, 255), 2)
    return canvas


def load_episode(annotations, episode, seed):
    with open(annotations) as f:
        samples = json.load(f)
    episodes = defaultdict(dict)
    for s in samples:
        ep_id, step = s["video_id"].rsplit("-", 1)
        episodes[ep_id].setdefault(int(step), s)  # duplicated stop steps are dropped
    if episode is None:
        random.seed(seed)
        episode = random.choice(sorted(episodes))
    if episode not in episodes:
        raise SystemExit(f"Episode {episode} not found in {annotations}")
    steps = episodes[episode]
    return episode, [steps[k] for k in sorted(steps)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", help="RxR episode id, e.g. 23289 (folder name in train/). Random if omitted.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for picking a random episode")
    parser.add_argument("--max-steps", type=int, default=None, help="Only run the first N steps")
    parser.add_argument("--model-path", default=osp.join(REPO_ROOT, "checkpoints/navila-llama3-8b-8f"))
    parser.add_argument("--precision", choices=["4bit", "8bit", "fp16"], default="4bit")
    parser.add_argument("--annotations", default=osp.join(REPO_ROOT, "NaVILA-Dataset/RxR/annotations.json"))
    parser.add_argument("--image-folder", default=osp.join(REPO_ROOT, "NaVILA-Dataset/RxR/train"))
    parser.add_argument("--out-dir", default=osp.join(REPO_ROOT, "output/rxr_vis"))
    parser.add_argument("--fps", type=float, default=1.5)
    args = parser.parse_args()

    print(f"Reading {args.annotations} ...")
    episode, steps = load_episode(args.annotations, args.episode, args.seed)
    steps = steps[: args.max_steps]
    print(f"Episode {episode}: {len(steps)} steps")
    print(f"Instruction: {normalize_instruction(steps[0]['q'])}\n")

    tokenizer, model, image_processor, _ = load_model(args.model_path, args.precision)
    num_frames = model.config.num_video_frames

    results, inputs, t0 = [], [], time.time()
    for i, sample in enumerate(steps):
        images = vlnce_frame_sampling([osp.join(args.image_folder, f) for f in sample["frames"]], num_frames)
        pred = predict(model, tokenizer, image_processor, images, normalize_instruction(sample["q"]))
        results.append((pred, sample["a"]))
        inputs.append(images)
        mark = {GT_COLOR: "==", TYPE_COLOR: "~=", BAD_COLOR: "!="}[verdict_color(pred, sample["a"])]
        print(f"[{i + 1:3d}/{len(steps)}] pred {short_label(pred):>10} {mark} gt {short_label(sample['a'])}")
    print(f"\nInference: {time.time() - t0:.1f}s for {len(steps)} steps")

    out_dir = osp.join(args.out_dir, episode)
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):  # drop panels left over from a longer previous run
        if re.fullmatch(r"step_\d+\.png", name):
            os.remove(osp.join(out_dir, name))
    writer = cv2.VideoWriter(osp.join(out_dir, "episode.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (WIDTH, HEIGHT))
    for i in range(len(steps)):
        panel = render_step(episode, i, steps, results, inputs[i])
        cv2.imwrite(osp.join(out_dir, f"step_{i:03d}.png"), panel)
        writer.write(panel)
    writer.release()

    n = len(results)
    exact = sum(p.strip() == g.strip() for p, g in results)
    same_type = sum(action_type(p) == action_type(g) for p, g in results)
    with open(osp.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"episode {episode}\ninstruction: {normalize_instruction(steps[0]['q'])}\n\n")
        for i, (p, g) in enumerate(results):
            f.write(f"step {i:3d} | pred: {p} | gt: {g}\n")
        f.write(f"\nexact match: {exact}/{n}\naction-type match: {same_type}/{n}\n")
    print(f"Exact match: {exact}/{n} | Action-type match: {same_type}/{n}")
    print(f"Saved {n} panels + episode.mp4 + summary.txt to {out_dir}")


if __name__ == "__main__":
    main()
