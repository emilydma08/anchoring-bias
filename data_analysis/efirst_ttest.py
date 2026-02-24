import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

df1 = pd.read_csv("statistics/baseline_anchoring_effects.csv") 
df2 = pd.read_csv("statistics/efirst_anchoring_effects.csv")  

merged_df = pd.merge(
    df1,
    df2,
    on=["Model", "Question_Num"],
    suffixes=("_baseline", "_efirst")
)

results = []

for model, g in merged_df.groupby("Model"):
    n = len(g)
    
    if n < 2:
        results.append({
            "Model": model,
            "n_pairs": n,
            "Baseline_mean": g["Anchoring_Effect_baseline"].mean(),
            "efirst_mean": g["Anchoring_Effect_efirst"].mean(),
            "Mean_diff": np.nan,
            "t": np.nan,
            "p": np.nan,
        })
        continue

    t_stat, p_value = ttest_rel(
        g["Anchoring_Effect_baseline"],
        g["Anchoring_Effect_efirst"]
    )

    results.append({
        "Model": model,
        "n_pairs": n,
        "Baseline_mean": g["Anchoring_Effect_baseline"].mean(),
        "efirst_mean": g["Anchoring_Effect_efirst"].mean(),
        "Mean_diff": (g["Anchoring_Effect_baseline"] - g["Anchoring_Effect_efirst"]).mean(),
        "t": t_stat,
        "p": p_value,
    })

results_df = pd.DataFrame(results).sort_values("Model").reset_index(drop=True)

print("Paired t-test: Baseline vs efirst per Model")
print(results_df.to_string(index=False))