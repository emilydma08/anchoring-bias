import pandas as pd
from scipy.stats import ttest_rel

df = pd.read_csv('cleaned_data/baseline_data_cleaned.csv')

df = df.drop(columns=["Unnamed: 0"], errors="ignore")

group_cols = ["Model", "Experiment", "Question_Num", "Anchor_Type"]
mean_df = df.groupby(group_cols, as_index=False)["Estimate"].mean()

pivot_df = mean_df.pivot_table(
    index=["Model", "Experiment", "Question_Num"],
    columns="Anchor_Type",
    values="Estimate"
).reset_index()

pivot_df = pivot_df.dropna(subset=["High", "Low"])

t_stat, p_value = ttest_rel(pivot_df["High"], pivot_df["Low"])

print("Number of matched rows:", len(pivot_df))
print("t-statistic:", t_stat)
print("p-value:", p_value)