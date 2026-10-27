from .dataset import LLMDataset
from .model import BertClassifierLLM
from .train import Trainer

if __name__ == "__main__":
    trainer = Trainer(
        model = BertClassifierLLM(),
        dataset = LLMDataset("train.parquet")
    )
    trainer.train(
        epochs=5,
    )
