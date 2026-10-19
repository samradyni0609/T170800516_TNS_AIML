
# Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

# Load dataset
data = load_diabetes(as_frame=True)
df = data.frame

# Q1: Display first 10 rows
print("First 10 rows:")
print(df.head(10))

# Q2: Dataset shape
print("\nRows and columns:", df.shape)

# Q3: Column names
print("\nColumns:", df.columns.tolist())

# Q4: Data types
print("\nData types:")
print(df.dtypes)

# Q5: Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Q6: Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Correlation matrix
correlation = df.corr(numeric_only=True)

print("\nCorrelation matrix:")
print(correlation)

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(correlation, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Histograms
df.hist(figsize=(12, 9), bins=20)
plt.tight_layout()
plt.show()

# Boxplots
df.drop(columns=["target"]).plot(
    kind="box", subplots=True, layout=(3, 3),
    figsize=(12, 8), sharex=False
)
plt.tight_layout()
plt.show()

# Save dataset
df.to_csv("diabetes_dataset.csv", index=False)
print("\nDataset saved successfully.")
