import pandas as pd

df = pd.read_csv("data/rides.csv")
dup_df = pd.concat([df] * 10, ignore_index=True)
dup_df.to_csv("data/duplicates.csv", index=False)
