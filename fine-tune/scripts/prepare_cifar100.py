"""Download CIFAR100 and lay it out as a DISEF-style dataset.

This produces:

- ``data/CIFAR100/<folder_name>/<image>.png`` (one class sub-folder per class)
- ``artifacts/cifar100/metadata.csv``  (``idx, folder_name, class_name``)
- ``artifacts/cifar100/split_coop.csv`` (``split, filename``)

The class ``folder_name`` uses the torchvision CIFAR100 names (underscored, e.g.
``aquarium_fish``) while ``class_name`` is a human-readable version used for the
prompts. The ``val`` split mirrors ``test``, matching the other DISEF datasets
(``data_val == data_test``).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import CIFAR100
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/")
    parser.add_argument("--download_root", type=str, default="data/_cifar100_raw/")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.data_dir) / "CIFAR100"
    artifact_path = Path(args.artifact_dir) / "cifar100"
    dataset_path.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)

    # Download (if needed) and load CIFAR100 through torchvision.
    trainset = CIFAR100(root=args.download_root, train=True, download=True)
    testset = CIFAR100(root=args.download_root, train=False, download=True)

    folder_names = list(trainset.classes)  # 100 underscored names
    class_names = [c.replace("_", " ") for c in folder_names]

    metadata_df = pd.DataFrame(
        {
            "idx": list(range(len(folder_names))),
            "folder_name": folder_names,
            "class_name": class_names,
        }
    )
    metadata_df.to_csv(artifact_path / "metadata.csv", index=False)

    rows = []
    for split, dataset in [("train", trainset), ("test", testset)]:
        data = np.asarray(dataset.data)  # (N, 32, 32, 3) uint8
        targets = np.asarray(dataset.targets)  # (N,) int
        for i in tqdm(range(len(data)), desc=f"extracting {split}"):
            folder_name = folder_names[int(targets[i])]
            class_dir = dataset_path / folder_name
            class_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{split}_{i:05d}.png"
            filepath = class_dir / filename
            if not filepath.exists():
                Image.fromarray(data[i]).save(filepath)

            rows.append((split, f"{folder_name}/{filename}"))

    split_df = pd.DataFrame(rows, columns=["split", "filename"])

    # The dataloaders expose val == test; add a val split that mirrors test.
    val_df = split_df[split_df["split"] == "test"].copy()
    val_df["split"] = "val"
    split_df = pd.concat([split_df, val_df], ignore_index=True)

    split_df.to_csv(artifact_path / "split_coop.csv", index=False)

    print(f"Wrote {metadata_df.shape[0]} classes to {artifact_path / 'metadata.csv'}")
    print(f"Wrote {split_df.shape[0]} rows to {artifact_path / 'split_coop.csv'}")
    print(f"Images under {dataset_path}")


if __name__ == "__main__":
    main()
