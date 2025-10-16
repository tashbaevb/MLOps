import pandas as pd
import numpy as np

n_weeks = 50
np.random.seed(2)

weeks = np.arange(1, n_weeks + 1)
demand = 50 + 0.5 * weeks + 10 * np.sin(2 * np.pi * weeks / 52) + np.random.normal(0, 5, n_weeks)

df = pd.DataFrame({
    "weeks": weeks,
    "demand": demand
})

csv_file = "data/dataset/synthetic_demand.csv"
df.to_csv(csv_file, index=False)
print(f"Saved dataset with {n_weeks} weeks: {csv_file}")
