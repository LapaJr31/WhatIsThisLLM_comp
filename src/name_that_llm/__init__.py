from .dataset import LLMDataset
from .model import BertClassifierLLM
from .train import Trainer

def main() -> None:
    trainer = Trainer(
        model=BertClassifierLLM(),
        dataset=LLMDataset("train.parquet"),
    )
    trainer.train(epochs=5)

if __name__ == "__main__":
    main()
