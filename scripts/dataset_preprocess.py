# %% Imports
import sys
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss

from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
DATASET_PATH = PROJECT_ROOT / "dataset"

# %% load dataset
df_train = pd.read_csv(DATASET_PATH / 'train.csv')
df_test = pd.read_csv(DATASET_PATH / 'test.csv')

print(f"Train\n{df_train.head}\nTest\n{df_test.head}\n")
# %% plot train distribution
lengths = df_train["text"].str.len().dropna()
x = pd.Series(range(int(lengths.min()), int(lengths.max()) + 1))

plt.hist(lengths, bins=20, density=True, alpha=0.5, color="red", label="Observed")

for name, dist in [
        ("Exponential", ss.expon),
        ("Gamma", ss.gamma),
        ("Lognormal", ss.lognorm),
    ]:
    params = dist.fit(lengths)
    plt.plot(x, dist.pdf(x, *params), label=name)

plt.xlabel("Length")
plt.ylabel("Density")
plt.legend()
plt.show()
# %% look at test
print(df_test["text"])


# %% to parquet
df_test.to_parquet(f"{DATASET_PATH}/test.parquet", engine="pyarrow")
df_train.to_parquet(f"{DATASET_PATH}/train.parquet", engine="pyarrow")
