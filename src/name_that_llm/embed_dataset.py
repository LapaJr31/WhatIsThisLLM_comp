import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = (Path(__file__).resolve().parent / "../../dataset").resolve()


class EmbedDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, name: str, embedder: str, data_dir: Path | None = None):
        d = (data_dir or DATA_DIR) / "cache"
        emb_path = d / f"{name}-{embedder}-emb.npy"
        labels_path = d / f"{name}-{embedder}-labels.npy"
        names_path = d / f"{name}-{embedder}-labels.json"

        if not (emb_path.is_file() and labels_path.is_file()):
            raise FileNotFoundError(
                f"missing embedding cache: {emb_path} / {labels_path} — run the extraction step first"
            )

        self.emb = torch.from_numpy(np.load(emb_path))
        label_ids = np.load(labels_path).astype(np.int64)
        self.label_ids = torch.from_numpy(label_ids)

        if names_path.is_file():
            self.label_names = json.loads(names_path.read_text())["names"]
        else:
            self.label_names = sorted(set(label_ids.tolist()))

        assert len(self.emb) == len(self.label_ids)

    def __len__(self) -> int:
        return len(self.emb)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.emb[index], int(self.label_ids[index])
