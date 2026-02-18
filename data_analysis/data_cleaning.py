import pandas as pd

"""df = pd.read_csv('baseline_data.csv')

df["Estimate"] = pd.to_numeric(df["Estimate"], errors="coerce")

clean_df = df[
    df["Comparison"].isin(["HIGHER", "LOWER"]) &
    df["Estimate"].notna() & (df["Estimate"] % 1 == 0) & (df["Estimate"] < 100000000)
]

clean_df["Estimate"] = clean_df["Estimate"].astype("Int64")

print(len(clean_df))

clean_df.to_csv('baseline_data_clean.csv')"""

df = pd.read_csv('cleaned_data/baseline_data_toanalyze.csv')
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df.reset_index(drop=True, inplace=True)

print(df.head())
df.to_csv('baseline_data_cleaned.csv')