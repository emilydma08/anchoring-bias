import pandas as pd
from scipy.stats import ttest_rel

df1 = pd.read_csv("statistics/baseline_anchoring_effects.csv") 
df2 = pd.read_csv("statistics/efirst_anchoring_effects.csv")  

merged_df = pd.merge(
    df1,
    df2,
    on=["Model", "Question_Num"],
    suffixes=("_exp1", "_exp2")
)

merged_df = merged_df.dropna(subset=["Anchoring_Effect_exp1", "Anchoring_Effect_exp2"])

t_stat, p_value = ttest_rel(
    merged_df["Anchoring_Effect_exp1"],
    merged_df["Anchoring_Effect_exp2"]
)

print("Number of pairs used:", len(merged_df))
print("t-stat:", t_stat)
print("p-value:", p_value)

