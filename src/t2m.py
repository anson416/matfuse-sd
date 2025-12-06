import datetime as dt
import json
import os
from dataclasses import dataclass, fields
from os.path import join
from typing import Optional

import matplotlib

# Set the backend to non-interactive "Agg" before importing other modules
matplotlib.use("Agg")

import numpy as np
from PIL import Image

from utils.inference_helpers import run_generation


@dataclass
class MatFuseMaps(object):
    diffuse: np.ndarray
    normal: np.ndarray
    roughness: np.ndarray
    specular: np.ndarray


@dataclass
class MatFuseOutput(object):
    basic: MatFuseMaps
    ema: MatFuseMaps
    cfg: MatFuseMaps


def run_matfuse(
    prompt: str,
    resolution: int = 512,
    steps: Optional[int] = None,
    guidance_scale: float = 5.0,
    seed: Optional[int] = None,
) -> tuple[MatFuseOutput, int, int]:
    steps = steps if steps is not None else 50
    if seed is None:
        seed = -1
    results, seed = run_generation(
        render_emb=None,
        palette_source=None,
        sketch=None,
        prompt=prompt,
        num_samples=1,
        image_resolution=resolution,
        ddim_steps=steps,
        seed=seed,
        ddim_eta=0.0,
        ucg_scale=guidance_scale,
    )
    return (
        MatFuseOutput(
            basic=MatFuseMaps(
                diffuse=results[2][:resolution, :resolution, :],
                normal=results[2][resolution:, :resolution, :],
                roughness=results[2][:resolution, resolution:, :],
                specular=results[2][resolution:, resolution:, :],
            ),
            ema=MatFuseMaps(
                diffuse=results[3][:resolution, :resolution, :],
                normal=results[3][resolution:, :resolution, :],
                roughness=results[3][:resolution, resolution:, :],
                specular=results[3][resolution:, resolution:, :],
            ),
            cfg=MatFuseMaps(
                diffuse=results[4][:resolution, :resolution, :],
                normal=results[4][resolution:, :resolution, :],
                roughness=results[4][:resolution, resolution:, :],
                specular=results[4][resolution:, resolution:, :],
            ),
        ),
        steps,
        seed,
    )


def save_map(array: np.ndarray, path: str) -> None:
    img = Image.fromarray(array)
    img.save(path)


if __name__ == "__main__":
    now = dt.datetime.now(dt.timezone.utc).strftime(r"%Y%m%d-%H%M%S-%f")
    prompt = input("Enter prompt: ")
    out_dir = input("Enter output directory (skip means ./outputs): ")
    steps = input("Enter steps (skip means default): ")
    seed = input("Enter seed (skip means random): ")

    if out_dir.strip() == "":
        out_dir = "./outputs"
    out_dir = join(out_dir, now)
    steps = None if steps.strip() == "" else int(steps)
    seed = None if seed.strip() == "" else int(seed)

    output, steps, seed = run_matfuse(prompt, steps=steps, seed=seed)
    for i, mode in enumerate([f.name for f in fields(MatFuseOutput)]):
        _out_dir = join(out_dir, f"{i + 1}_{mode}")
        os.makedirs(_out_dir, exist_ok=True)
        for cat in [f.name for f in fields(MatFuseMaps)]:
            save_map(
                getattr(getattr(output, mode), cat),
                join(_out_dir, f"{cat}.png"),
            )
    with open(join(out_dir, "info.json"), "w") as f:
        json.dump(
            {"prompt": prompt, "steps": steps, "seed": seed}, f, indent=2
        )

    print(f"![OUT_DIR] {out_dir}")
