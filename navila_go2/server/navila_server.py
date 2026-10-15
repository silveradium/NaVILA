"""NaVILA high-level policy server.

Runs in the `navila` conda env and serves the VLA over HTTP so that robot clients (Isaac Sim runner, real Go2)
can live in their own environments / machines.

  python navila_go2/server/navila_server.py --precision 4bit --port 8000

API
  GET  /health -> {"ok": true, "num_frames": 8, ...}
  POST /act    {"instruction": str, "frames": [base64 jpeg, ...]}  (oldest first, last = current view)
            -> {"text": "The next action is turn left 30 degrees.", "action": {...}, "latency_s": 1.2}
"""

import argparse
import os
import os.path as osp
import sys
import threading
import time

# Less fragmentation -> lower reserved VRAM next to Isaac Sim. Must be set before CUDA initializes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.join(REPO_ROOT, "scripts", "inference"))

# quick_infer patches transformers' quantized-tensor loader at import time (needed for 4/8-bit).
from quick_infer import load_model, normalize_instruction, predict  # noqa: E402

from llava.utils import disable_torch_init  # noqa: E402
from navila_go2.common.client import decode_jpeg, sample_frames  # noqa: E402
from navila_go2.common.mid_level import parse_navila_output  # noqa: E402


class ActRequest(BaseModel):
    instruction: str
    frames: list[str]


def last_token_logits_only(model):
    """During prefill, transformers 4.37 computes lm_head logits for every prompt position (~1.7k positions x 128k
    vocab, ~1.3 GB with the fp32 upcast) although greedy decoding only reads the last one. Slice before lm_head.
    Generation-only: never use this when a loss over all positions is needed."""

    def hook(_module, inputs):
        hidden = inputs[0]
        if hidden.dim() == 3 and hidden.shape[1] > 1:
            return (hidden[:, -1:, :],)
        return None

    model.get_lm_head().register_forward_pre_hook(hook)


def chunked_image_encoding(model, chunk: int = 2):
    """Run the vision tower on `chunk` frames at a time. SigLIP's eager attention materializes
    frames x 16 heads x 729^2 fp32 weights per layer (~270 MB for 8 frames); frames are encoded independently,
    so chunking gives the same features with a ~4x smaller activation peak."""
    encode = model.encode_images

    def encode_images(images):
        if images.shape[0] <= chunk:
            return encode(images)
        return torch.cat([encode(images[i : i + chunk]) for i in range(0, images.shape[0], chunk)], dim=0)

    model.encode_images = encode_images


def build_app(model_path: str, precision: str) -> FastAPI:
    disable_torch_init()
    t0 = time.time()
    tokenizer, model, image_processor, _ = load_model(model_path, precision)
    last_token_logits_only(model)
    chunked_image_encoding(model)
    num_frames = model.config.num_video_frames
    print(f"[navila-server] loaded {precision} model in {time.time() - t0:.1f}s ({num_frames} frames per step)")
    lock = threading.Lock()
    app = FastAPI(title="NaVILA server")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model": osp.basename(model_path.rstrip("/")),
            "precision": precision,
            "num_frames": num_frames,
            "vram_gib": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        }

    @app.post("/act")
    def act(req: ActRequest):
        if not req.frames:
            raise HTTPException(400, "frames must not be empty")
        frames = sample_frames([decode_jpeg(f) for f in req.frames], num_frames)
        images = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
        instruction = normalize_instruction(req.instruction)
        with lock:
            t0 = time.time()
            try:
                text = predict(model, tokenizer, image_processor, images, instruction)
            except torch.cuda.OutOfMemoryError:
                # another process on the GPU (Isaac Sim) spiked: free our cache and try once more
                torch.cuda.empty_cache()
                time.sleep(0.5)
                try:
                    text = predict(model, tokenizer, image_processor, images, instruction)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    raise HTTPException(503, "CUDA out of memory on the NaVILA server")
            latency = time.time() - t0
            # Isaac Sim shares this GPU: hand cached activation blocks back instead of hoarding them.
            torch.cuda.empty_cache()
        action = parse_navila_output(text)
        print(f"[navila-server] {latency:.2f}s  {text!r} -> {action}")
        return {
            "text": text,
            "action": {"kind": action.kind, "value": action.value},
            "latency_s": latency,
        }

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=osp.join(REPO_ROOT, "checkpoints/navila-llama3-8b-8f"))
    parser.add_argument("--precision", choices=["4bit", "8bit", "fp16"], default="4bit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(build_app(args.model_path, args.precision), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
