"""Build the ``selected_fewshots/<dataset>-by-seed/seed_<seed>/<class>/`` tree.

The generation pipeline expects one class sub-folder per seed. This script turns
the flat ``image_paths/<dataset>-train-16shots-seed=<seed>.txt`` produced by
``save_fewshot_samples.py`` into that structure by copying the selected images.
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_paths_file", type=str, required=True)
    parser.add_argument("--datasets_root", type=str, default="selected_fewshots")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.image_paths_file) as f:
        image_paths = [line.strip() for line in f if line.strip()]

    out_root = (
        Path(args.datasets_root) / f"{args.dataset}-by-seed" / f"seed_{args.seed}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image_path = Path(image_path)
        class_name = image_path.parent.name
        class_dir = out_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        dst = class_dir / image_path.name
        if not dst.exists():
            shutil.copy2(image_path, dst)

    print(f"Copied {len(image_paths)} images into {out_root}")


if __name__ == "__main__":
    main()
