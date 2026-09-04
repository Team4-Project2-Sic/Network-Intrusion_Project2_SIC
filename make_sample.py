import os
import pandas as pd

df = pd.read_csv("combinenew.csv", low_memory=False)
df.columns = df.columns.str.strip()

n_per_label = 2

sample = (
    df.groupby("Label", group_keys=False)
    .apply(lambda g: g.sample(min(n_per_label, len(g))), include_groups=False)
)

sample = sample.sample(frac=1)

output_dir = "samples"
os.makedirs(output_dir, exist_ok=True)

sample_unlabeled = sample.drop(columns=["Label"], errors="ignore")
output_path = os.path.join(output_dir, "sample_test_unlabeled.csv")
sample_unlabeled.to_csv(output_path, index=False)

print(f"Saved {output_path} (upload this to the app) - {len(sample_unlabeled)} rows")
print(sample["Label"].value_counts() if "Label" in sample.columns else "Sample created successfully.")