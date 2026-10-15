"""Quick NaVILA inference test on a single GPU.

Two modes:
  1. Dataset mode (default): sample N steps from NaVILA-Dataset/RxR and compare
     the predicted action against the ground-truth action in annotations.json.
  2. Custom mode: pass --video-file (mp4 or a folder of frames) and --query.

Examples (run from the repo root with the `navila` env active):
  python scripts/inference/quick_infer.py --num-samples 10
  python scripts/inference/quick_infer.py --video-file my_walk.mp4 --query "Walk to the door and stop."
"""

import argparse
import json
import os
import os.path as osp
import random
import re
import time

import cv2
import torch
import transformers.integrations as hf_integrations
from transformers import BitsAndBytesConfig

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_frame_from_vcap_vlnce,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    vlnce_frame_sampling,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))

# The patched modeling_utils.py (llava/train/transformers_replace) comes from an older transformers
# and passes `fp16_statistics`, which 4.37 renamed to `quantized_stats`. Translate it so
# bitsandbytes 4-bit/8-bit loading works without touching the installed library.
# modeling_utils imports it lazily from transformers.integrations, so patch it there.
_set_quantized_tensor = hf_integrations.set_module_quantized_tensor_to_device


def _set_quantized_tensor_compat(*args, fp16_statistics=None, **kwargs):
    if fp16_statistics is not None:
        kwargs["quantized_stats"] = {"SCB": fp16_statistics}
    return _set_quantized_tensor(*args, **kwargs)


hf_integrations.set_module_quantized_tensor_to_device = _set_quantized_tensor_compat


def load_model(model_path, precision):
    # Passing quantization_config directly: the builder's load_4bit/load_8bit flags
    # conflict with transformers 4.37. Only the LLM is quantized; vision tower stays fp16.
    kwargs = {}
    if precision == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif precision == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "fp16":
        # fp16 does not fit in 16 GB VRAM; spill the remaining LLM layers to CPU RAM (slow).
        # The cap covers LLM weights only; leave room for the vision tower, offloaded layers
        # streamed in per forward pass (lm_head alone is ~1 GB), and activations.
        kwargs["max_memory"] = {0: "11GiB", "cpu": "48GiB"}
    return load_pretrained_model(model_path, get_model_name_from_path(model_path), **kwargs)


def normalize_instruction(text):
    # Same cleanup as LazyVLNCEDataset in llava/data/dataset.py
    text = text.replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r"(?<=\.\s)([a-z])", lambda x: x.group().upper(), text.capitalize())
    return re.sub(r"\s+\.", ".", text)


def build_prompt(instruction, num_frames):
    image_tokens = "<image>\n" * (num_frames - 1)
    question = (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f'of historical observations {image_tokens}, and current observation <image>\n. Your assigned task is: "{instruction}" '
        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
        f"degree, moving forward a certain distance, or stop if the task is completed."
    )
    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv


def predict(model, tokenizer, image_processor, images, instruction):
    conv = build_prompt(instruction, len(images))
    prompt = conv.get_prompt()
    images_tensor = process_images(images, image_processor, model.config).to("cuda", dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            max_new_tokens=32,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text.removesuffix(stop_str).strip()


def action_type(text):
    text = text.lower()
    for name in ("turn left", "turn right", "move forward", "stop"):
        if name in text:
            return name
    return "unknown"


def load_custom_frames(video_file, num_frames):
    if osp.isdir(video_file):
        files = sorted(
            (f for f in os.listdir(video_file) if f.lower().endswith((".jpg", ".jpeg", ".png"))),
            key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)],
        )
        return vlnce_frame_sampling([osp.join(video_file, f) for f in files], num_frames)
    images, _ = get_frame_from_vcap_vlnce(cv2.VideoCapture(video_file), num_frames)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=osp.join(REPO_ROOT, "checkpoints/navila-llama3-8b-8f"))
    parser.add_argument("--precision", choices=["4bit", "8bit", "fp16"], default="4bit")
    parser.add_argument("--annotations", default=osp.join(REPO_ROOT, "NaVILA-Dataset/RxR/annotations.json"))
    parser.add_argument("--image-folder", default=osp.join(REPO_ROOT, "NaVILA-Dataset/RxR/train"))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-file", help="Custom mode: mp4 file or folder of frames")
    parser.add_argument("--query", help="Custom mode: navigation instruction")
    args = parser.parse_args()

    if args.video_file and not args.query:
        parser.error("--video-file requires --query")

    disable_torch_init()
    t0 = time.time()
    tokenizer, model, image_processor, _ = load_model(args.model_path, args.precision)
    num_frames = model.config.num_video_frames
    print(f"Loaded {args.precision} model in {time.time() - t0:.1f}s ({num_frames} frames per step)")

    if args.video_file:
        images = load_custom_frames(args.video_file, num_frames)
        t0 = time.time()
        output = predict(model, tokenizer, image_processor, images, normalize_instruction(args.query))
        print(f"\nAction: {output}  ({time.time() - t0:.2f}s)")
    else:
        print(f"Reading {args.annotations} ...")
        with open(args.annotations) as f:
            samples = json.load(f)
        random.seed(args.seed)
        picked = random.sample(samples, args.num_samples)
        del samples

        exact, same_type, latencies = 0, 0, []
        for i, sample in enumerate(picked, 1):
            paths = [osp.join(args.image_folder, frame) for frame in sample["frames"]]
            images = vlnce_frame_sampling(paths, num_frames)
            t0 = time.time()
            output = predict(model, tokenizer, image_processor, images, normalize_instruction(sample["q"]))
            latencies.append(time.time() - t0)
            exact += output.strip() == sample["a"].strip()
            same_type += action_type(output) == action_type(sample["a"])
            print(f"\n[{i}/{len(picked)}] {sample['video_id']}  ({len(sample['frames'])} history frames)")
            print(f"  instruction: {sample['q'][:150]}{'...' if len(sample['q']) > 150 else ''}")
            print(f"  predicted:   {output}")
            print(f"  gt:          {sample['a']}")

        n = len(picked)
        print(f"\nExact match: {exact}/{n} | Action-type match: {same_type}/{n}")
        print(f"Mean latency: {sum(latencies) / n:.2f}s (first call includes warm-up)")

    print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
