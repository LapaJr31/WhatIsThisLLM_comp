import torch
import numpy as np

from pathlib import Path
from transformers import AutoTokenizer, AutoModel

class EmbeddingExtractor:
    EMBEDDERS = {
       "embedgemma300m":  "google/embeddinggemma-300m",
       "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
       "modernbert-large": "answerdotai/ModernBERT-large",
    }
    DATASET_PATH = (Path(__file__).resolve().parent / "../../dataset").resolve()
    def __init__(
        self,
        embedder: str,
        data_dir: Path = DATASET_PATH,
        batch_size: int = 64,
        max_length: int = 512,
        device: str = "auto",
        fp16: bool = True,
    )
        pass

    def load_data(self, path: Path = DATASET_PATH / "train_plus.parquet") -> tuple[list[str], np.ndarray, list[str]]:
        pass

    def tokenize(self, texts: list[str]) -> dict[str, Tensor]:
        pass

    @torch.no_grad()
    def _embed_batch(self, input_ids,  attn_mask) -> np.ndarray:
        pass

    def extract_all(self, texts: list[str]) -> np.ndarray:
        pass

    def save(self, name: str, emb: np.ndarray, label_ids: np.ndarray, label_names: list[str]) -> None
        pass

    def run(self, name: str = "train_plus") -> None:
        pass

    @classmethod
    def run_all(cls, name: str= "train_plus", embedders: list[str] | None = None) -> Pass:
        pass
