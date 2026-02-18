import pandas as pd

"""dfl = pd.read_csv('raw_data/efirst/llama_efirst_trials.csv')
dfd = pd.read_csv('raw_data/efirst/deepseek_efirst_trials.csv')
dfm = pd.read_csv('raw_data/efirst/mistral_efirst_trials.csv')

dfm['Trial_Id'] = dfm['Trial_Id'] + 400

combined_df = pd.concat([dfl, dfd, dfm], ignore_index=True)

combined_df.to_csv("efirst_data.csv", index=False)"""


df = pd.read_csv('raw_data/efirst/efirst_data.csv')

df["Estimate"] = pd.to_numeric(df["Estimate"], errors="coerce")

clean_df = df[
    df["Comparison"].isin(["HIGHER", "LOWER"]) &
    df["Estimate"].notna() & (df["Estimate"] % 1 == 0) & (df["Estimate"] < 100000000)
]

clean_df["Estimate"] = clean_df["Estimate"].astype("Int64")

print(len(clean_df))

clean_df.to_csv('efirst_data_cleaned.csv')

"""df = pd.read_csv('cleaned_data/baseline_data_toanalyze.csv')
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df.reset_index(drop=True, inplace=True)

print(df.head())
df.to_csv('baseline_data_cleaned.csv')"""