import pandas as pd

# metodologia crisp - dm
# CRoss
# Industry
# Standard
# Process
# Data
# Mining

df = pd.read_csv("bank.csv", sep= ";")


print(df.head(5))
print(df.describe())
