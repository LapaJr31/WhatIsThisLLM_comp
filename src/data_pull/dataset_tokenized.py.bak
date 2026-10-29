import numpy as np
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from typing import Any, cast

DATA_DIR = (Path(__file__).resolve().parent / "../../dataset").resolve()

class LLMDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, data: str):

        path = DATA_DIR / data
        cache = DATA_DIR / "cache"
        cache.mkdir(parents=True, exist_ok=True)

        ids_path = cache / f"{path.stem}-modernbert-base-input_ids.npy"
        mask_path = cache / f"{path.stem}-modernbert-base-attention_mask.npy"

        if not path.is_file():
            raise FileNotFoundError(path)

        try:
            pq.read_metadata(path)
        except (pa.ArrowInvalid, OSError) as exc:
            raise ValueError(f"{path} is not a valid parquet file") from exc

        dataset = pd.read_parquet(path, engine="pyarrow")

        tokenizer = cast(Any, AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base"))

        self.text = dataset["text"].astype(str).tolist()
        self.labels: list[str] = dataset["model"].astype(str).tolist()

        self.label_names = sorted(set(self.labels))
        self.label_to_id = {name: i for i, name in enumerate(self.label_names)}
        self.label_ids = [self.label_to_id[label] for label in self.labels]

        if ids_path.is_file() and mask_path.is_file():
            ids_np = np.load(ids_path)
            mask_np = np.load(mask_path)
        else:
            encoded_list = tokenizer(
                self.text,
                truncation=False,
                padding=False,
            )
            max_len = max(len(seq) for seq in encoded_list["input_ids"])
            pad_id = tokenizer.pad_token_id

            ids_np = np.full((len(self.text), max_len), pad_id, dtype=np.int64)
            mask_np = np.zeros((len(self.text), max_len), dtype=np.int64)
            for i, seq in enumerate(encoded_list["input_ids"]):
                ids_np[i, : len(seq)] = seq
                mask_np[i, : len(seq)] = 1

            np.save(ids_path, ids_np)
            np.save(mask_path, mask_np)

        self.input_ids = torch.from_numpy(ids_np)
        self.attention_mask = torch.from_numpy(mask_np)
        self.lengths = mask_np.sum(axis=1)

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.input_ids[index], self.attention_mask[index], self.label_ids[index]
