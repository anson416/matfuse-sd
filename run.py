# #!/bin/zsh

# if [ $# -eq 0 ]; then
#     echo "Error: No argument provided."
#     echo "Usage: ./run.sh <prompt>"
#     exit 1
# fi

# PROMPT=$1

# python src/utils/inference_helpers.py --ckpt models/matfuse-full.ckpt --config src/configs/diffusion/matfuse-ldm-vq_f8.yaml

import argparse
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str)
parser.add_argument("--steps", type=str, default="")
parser.add_argument("--seed", type=str, default="")
args = parser.parse_args()

command = [
    "python",
    "src/t2m.py",
    "--ckpt",
    "models/matfuse-full.ckpt",
    "--config",
    "src/configs/diffusion/matfuse-ldm-vq_f8.yaml",
]
proc = subprocess.Popen(
    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf8"
)
output, err = proc.communicate(f"{args.prompt}\n\n{args.steps}\n{args.seed}\n")

re_match = re.search(r"^!\[OUT_DIR\] (.+)$", output, flags=re.MULTILINE)
if re_match is not None:
    output_directory = re_match.group(1)
    print(f"Output directory: {output_directory}")
else:
    print("Output directory not found in the output.")
