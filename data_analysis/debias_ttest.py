import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

df1 = pd.read_csv("statistics/baseline_anchoring_effects.csv") 
df2 = pd.read_csv("statistics/debias_anchoring_effects.csv")  

merged_df = pd.merge(
    df1,
    df2,
    on=["Model", "Question_Num"],
    suffixes=("_baseline", "_debias")
)

results = []

for model, g in merged_df.groupby("Model"):
    n = len(g)
    
    if n < 2:
        results.append({
            "Model": model,
            "n_pairs": n,
            "Baseline_mean": g["Anchoring_Effect_baseline"].mean(),
            "Debias_mean": g["Anchoring_Effect_debias"].mean(),
            "Mean_diff": np.nan,
            "t": np.nan,
            "p": np.nan,
        })
        continue

    t_stat, p_value = ttest_rel(
        g["Anchoring_Effect_baseline"],
        g["Anchoring_Effect_debias"]
    )

    results.append({
        "Model": model,
        "n_pairs": n,
        "Baseline_mean": g["Anchoring_Effect_baseline"].mean(),
        "Debias_mean": g["Anchoring_Effect_debias"].mean(),
        "Mean_diff": (g["Anchoring_Effect_baseline"] - g["Anchoring_Effect_debias"]).mean(),
        "t": t_stat,
        "p": p_value,
    })

results_df = pd.DataFrame(results).sort_values("Model").reset_index(drop=True)

print("Paired t-test: Baseline vs Debias per Model")
print(results_df.to_string(index=False))