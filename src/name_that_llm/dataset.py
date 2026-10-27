import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from typing import Any, cast

DATA_DIR = (Path(__file__).resolve().parent / "../datasets").resolve()

BATCH_SIZE = 512

class LLMDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, data: str):

        path = DATA_DIR / data
        cache = DATA_DIR / "cache"
        cache.mkdir(parents=True, exist_ok=True)

        encoded_path = cache / f"{path.stem}-modernbert-base.pt"

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

        if encoded_path.is_file():
            encoded = torch.load(encoded_path, map_location="cpu", weights_only=True)
        else:
            encoded = {
                "input_ids": [],
                "attention_mask": [],
            }
            for start in range(0, len(self.text), BATCH_SIZE):
                batch = tokenizer(
                    self.text[start : start + BATCH_SIZE],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded["input_ids"].append(batch["input_ids"])
                encoded["attention_mask"].append(batch["attention_mask"])
            torch.save(encoded, encoded_path)

        self.input_ids = torch.cat(encoded["input_ids"])
        self.attention_mask = torch.cat(encoded["attention_mask"])

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.input_ids[index], self.attention_mask[index], self.label_ids[index]
