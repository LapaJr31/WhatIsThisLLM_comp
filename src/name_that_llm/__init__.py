from .embed_dataset import EmbedDataset
from .embed_model import EmbedModel
from .head_train import HeadTrainer

DEFAULT_EMBEDDER = "embeddinggemma-300m"


def main() -> None:
    dataset = EmbedDataset("train_plus", DEFAULT_EMBEDDER)
    model = EmbedModel(model_id=f"google/{DEFAULT_EMBEDDER}", num_labels=18)
    trainer = HeadTrainer(model=model, dataset=dataset)
    trainer.train(epochs=100)


if __name__ == "__main__":
    main()
