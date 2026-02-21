import pandas as pd
import numpy as np

df = pd.read_csv("statistics/anchoring_effects.csv")

df["Anchoring_Effect"] = pd.to_numeric(df["Anchoring_Effect"], errors="coerce")

pivot_df = df.pivot_table(
    index=["Model", "Question_Num"],
    columns="Experiment",
    values="Anchoring_Effect"
).reset_index()

def paired_cohens_d(series1, series2):
    diffs = series1 - series2
    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)
    return mean_diff / std_diff if std_diff != 0 else np.nan

results = []

for model in pivot_df["Model"].unique():
    model_df = pivot_df[pivot_df["Model"] == model]
    
    # Baseline v. Debias
    df_debias = model_df.dropna(subset=["Baseline", "Debias"])
    if len(df_debias) >= 2: 
        d_debias = paired_cohens_d(df_debias["Baseline"], df_debias["Debias"])
    else:
        d_debias = np.nan
    
    # Baseline v. Estimate-First
    df_estimate = model_df.dropna(subset=["Baseline", "Estimate-First"])
    if len(df_estimate) >= 2:
        diffs_est = df_estimate["Baseline"] - df_estimate["Estimate-First"]
        print(f"\nModel: {model} - Baseline vs Estimate-First differences:\n", diffs_est.values)
        print(f"Std of differences: {diffs_est.std(ddof=1)}")
        d_estimate = paired_cohens_d(df_estimate["Baseline"], df_estimate["Estimate-First"])
    else:
        d_estimate = np.nan
    
    results.append({
        "Model": model,
        "Cohen_d_Baseline_vs_Debias": d_debias,
        "Cohen_d_Baseline_vs_EstimateFirst": d_estimate
    })

    print(f"\nModel: {model}")
    print("Baseline count:", model_df["Baseline"].notna().sum())
    print("Estimation count:", model_df["Estimate-First"].notna().sum())

    matched = model_df.dropna(subset=["Baseline", "Estimate-First"])
    print("Matched pairs:", len(matched))
    print("Matched question numbers:", matched["Question_Num"].tolist())

results_df = pd.DataFrame(results)
results_df.to_csv("cohens_d_results_all_models.csv", index=False)

print("\nCohen's d results for all models:")
print(results_df)
