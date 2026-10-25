import torch
import numpy as np

from .model import BertClassifierLLM
from .dataset import LLMDataset

from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model: BertClassifierLLM, dataset: LLMDataset, seed=99):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset

        self.batch_size = self.max_batch(self.probe_batch) if self.device.type == "cuda" else 1

    def max_batch(self, probe, limit=4096):
        for b in range(1, limit + 1):
            try:
                probe(b)
                torch.cuda.synchronize(self.device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return b - 1
        return limit

    def probe_batch(self, batch_size):
        pass # will do later
