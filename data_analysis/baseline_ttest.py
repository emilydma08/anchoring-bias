import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

df = pd.read_csv("cleaned_data/baseline_data_cleaned.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

group_cols = ["Model", "Question_Num", "Anchor_Type"]
mean_df = df.groupby(group_cols, as_index=False)["Estimate"].mean()

pivot_df = (
    mean_df.pivot_table(
        index=["Model", "Question_Num"],
        columns="Anchor_Type",
        values="Estimate",
        aggfunc="mean",
    )
    .reset_index()
)

pivot_df = pivot_df.dropna(subset=["High", "Low"]).copy()
pivot_df["Diff"] = pivot_df["High"] - pivot_df["Low"]

results = []
for model, g in pivot_df.groupby("Model", dropna=False):
    n = len(g)
    high_mean = g["High"].mean() if n else np.nan
    low_mean = g["Low"].mean() if n else np.nan
    diff_mean = g["Diff"].mean() if n else np.nan

    if n < 2:
        results.append({
            "Model": model,
            "n_pairs": n,
            "High_mean": high_mean,
            "Low_mean": low_mean,
            "Mean_diff": diff_mean,
            "t": np.nan,
            "p": np.nan,
        })
        continue

    t, p = ttest_rel(g["High"], g["Low"])

    results.append({
        "Model": model,
        "n_pairs": n,
        "High_mean": high_mean,
        "Low_mean": low_mean,
        "Mean_diff": diff_mean,
        "t": t,
        "p": p,
    })

results_df = pd.DataFrame(results).sort_values("Model").reset_index(drop=True)

print("Matched rows total:", len(pivot_df))
print("\nPaired t-test (High vs Low) per Model:")
print(results_df.to_string(index=False))