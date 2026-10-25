import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from typing import Any, cast

DATA_DIR = (Path(__file__).resolve().parent / "../datasets").resolve()

class LLMDataset(Dataset[tuple[torch.Tensor, str]]):
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = cast(Any, AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base"))

        self.text = dataset["text"].astype(str).tolist()
        self.labels: list[str] = dataset["model"].astype(str).tolist()


        if encoded_path.is_file():
            encoded = torch.load(encoded_path, map_location="cpu", weights_only=True)
        else:
            encoded = tokenizer(
                self.text,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            torch.save(dict(encoded), encoded_path)


        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        return self.input_ids[index], self.labels[index]
