import pandas as pd

df = pd.read_csv('cleaned_data/efirst_data_cleaned.csv')

df = df.drop(columns=["Unnamed: 0"], errors="ignore")

group_cols = ["Model", "Experiment", "Question_Num", "Anchor_Type"]
mean_df = df.groupby(group_cols, as_index=False)["Estimate"].mean()

pivot_df = mean_df.pivot_table(
    index=["Model", "Experiment", "Question_Num"],
    columns="Anchor_Type",
    values="Estimate"
).reset_index()

pivot_df = pivot_df.rename(columns={"High": "High_Mean", "Low": "Low_Mean"})

pivot_df["Anchoring_Effect"] = (
    pivot_df["High_Mean"] - pivot_df["Low_Mean"]
) / ((pivot_df["High_Mean"] + pivot_df["Low_Mean"]) / 2)

result_df = pivot_df[["Model", "Experiment", "Question_Num", "Anchoring_Effect"]]

result_df.to_csv("efirst_anchoring_effects.csv", index=False)

print(result_df.head())