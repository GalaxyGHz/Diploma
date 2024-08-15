import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('translated_dataset.csv')

data = df.filter(items=["translation"])
data["length"] = df.translation.apply(lambda x: int(len(x)))
mean = np.round(np.mean(data["length"]), 2)
median = np.median(data["length"])
counts = data["length"].value_counts().rename_axis("Value").reset_index(name="Frequency")

minimum = min(counts.Value)
maximum= max(counts.Value)

# Plot
fig = plt.figure(figsize = (10, 6))
plt.bar(counts.Value, counts.Frequency, color=(0.5, 0.68, 0.99))
plt.xlabel("Character Length")
plt.ylabel("Frequency")
plt.xlim(minimum, maximum)
plt.title(label=f"Slovene Translation Of iSarcasmEval Dataset\n")

# Add vertical lines for mean and median
plt.axvline(mean, color=(1.0, 0.0, 0.0), linestyle="--", alpha=0.8, label=f"Mean: {mean}")
plt.axvline(median, color=(0.9, 0.5, 0.6), linestyle="--", alpha=0.9, label=f"Median: {median}")

# Show
plt.legend()
plt.savefig(f"./figures/text_lengths_translated.pdf")
plt.show()         
