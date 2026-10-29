# IT'S FUCKING USELESS
from .datagrabber import SynthDataXD

def main():
    grabber = SynthDataXD()

    prompts = grabber.create_prompts(n_per_class=20, seed=42)
    grabber.run(prompts, "dataset/synth_checkpoint.csv")
    grabber.build_synth_dataset(
        "dataset/train.csv",
        "dataset/synth_checkpoint_v2.csv",
        "dataset/train_synth_v2.csv",
    )

if __name__ == "__main__":
    main()
