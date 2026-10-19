
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset from the project folder
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "rainfall.csv")
df.columns = df.columns.str.strip()

# Q1. First 10 rows
print(df.head(10))

# Q2. Number of rows and columns
print(df.shape)

# Q3. All column names
print(df.columns.tolist())

# Q4. Data types
print(df.dtypes)

# Q5. Missing values
print(df.isnull().sum())

# Q6. Mean, median and standard deviation
print("Mean:", df["rainfall"].mean())
print("Median:", df["rainfall"].median())
print("Standard Deviation:", df["rainfall"].std())

# Q7. Maximum and minimum temperature
print("Maximum:", df["temperature"].max())
print("Minimum:", df["temperature"].min())

# Q8. Numerical columns
num_df = df.select_dtypes(include=np.number)
print(num_df.head())


# Q9. Correlation matrix
corr = num_df.corr()
print(corr)

# Q10. Correlation heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# Q11. Correlation with rainfall
print(corr["rainfall"].sort_values(ascending=False))

# Q12. Pairs with correlation > 0.7
upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
pairs = corr.where(upper).stack()
high_pairs = pairs[pairs > 0.7]
print(high_pairs)

# Q13. Highly correlated features list
features = sorted(set(
    feature for pair in high_pairs.index for feature in pair
))
print(features)

# Q14. Drop one feature from each correlated pair
to_drop = sorted(set(pair[1] for pair in high_pairs.index))
reduced_df = num_df.drop(columns=to_drop)
print(reduced_df.head())

# Q15. Scatter plot: rainfall vs humidity
sns.scatterplot(data=df, x="humidity", y="rainfall")
plt.show()


# Q16. Histograms for all numerical features
num_df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Q17. Rainfall histogram with 30 bins
plt.hist(df["rainfall"].dropna(), bins=30)
plt.xlabel("Rainfall")
plt.ylabel("Frequency")
plt.show()

# Q18. Boxplots
num_df.plot(kind="box", subplots=True, layout=(-1, 3),
            figsize=(12, 8))
plt.tight_layout()
plt.show()

# Q19. Outliers using IQR
q1 = num_df.quantile(0.25)
q3 = num_df.quantile(0.75)
iqr = q3 - q1
outliers = (num_df < q1 - 1.5 * iqr) | (num_df > q3 + 1.5 * iqr)
print(outliers.sum())

# Q20. Skewness
print(num_df.skew())

# Q21. KDE plot for temperature
sns.kdeplot(data=df, x="temperature", fill=True)
plt.show()

# Q22. Compare mean and median
print("Mean:", df["rainfall"].mean())
print("Median:", df["rainfall"].median())

# Q23. Count right-skewed features
print("Right-skewed features:", (num_df.skew() > 0).sum())


# Q24. Average rainfall by weather condition
print(df.groupby("weather_condition")["rainfall"].mean())

# Q25. Top 3 rainfall records
print(df.nlargest(3, "rainfall"))

# Q26. Rainfall greater than average
avg = df["rainfall"].mean()
print(df[df["rainfall"] > avg])

# Q27. Average temperature for high vs low rainfall
high = df.loc[df["rainfall"] > avg, "temperature"].mean()
low = df.loc[df["rainfall"] <= avg, "temperature"].mean()
print("High rainfall temperature:", high)
print("Low rainfall temperature:", low)

# Q28. Create high_rainfall column
df["high_rainfall"] = (df["rainfall"] > avg).astype(int)
print(df.head())

# Q29. Count high-rainfall records
print(df["high_rainfall"].value_counts())

# Q30. Correlation for high-rainfall records
high_df = df[df["high_rainfall"] == 1]
print(high_df.select_dtypes(include=np.number).corr())


# Q31. Function to detect outliers using IQR
def detect_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    return data[
        (data[column] < q1 - 1.5 * iqr) |
        (data[column] > q3 + 1.5 * iqr)
    ]

print(detect_outliers(df, "rainfall"))

# Q32. Remove rainfall outliers
q1 = df["rainfall"].quantile(0.25)
q3 = df["rainfall"].quantile(0.75)
iqr = q3 - q1
clean_df = df[
    df["rainfall"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
].copy()
print(clean_df.shape)

# Q33. Min-Max normalization
normalized = pd.DataFrame(
    MinMaxScaler().fit_transform(num_df),
    columns=num_df.columns
)
print(normalized.head())

# Q34. Z-score standardization
standardized = pd.DataFrame(
    StandardScaler().fit_transform(num_df),
    columns=num_df.columns
)
print(standardized.head())

# Q35. Pairplot
sns.pairplot(df[["rainfall", "temperature", "humidity"]].dropna())
plt.show()

# Q36. Correlation > 0.5 with rainfall
correlations = num_df.corr()["rainfall"]
print(correlations[
    (correlations > 0.5) & (correlations.index != "rainfall")
])

# Q37. Check target imbalance
print(df["high_rainfall"].value_counts(normalize=True) * 100)

# Q38. Before vs after outlier removal
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df["rainfall"].dropna())
plt.title("Before")

plt.subplot(1, 2, 2)
sns.histplot(clean_df["rainfall"].dropna())
plt.title("After")
plt.tight_layout()
plt.show()

# Q39. Summary report
report = pd.DataFrame({
    "Mean": num_df.mean(),
    "Skewness": num_df.skew(),
    "Outlier_Count": outliers.sum()
})
print(report)
report.to_csv(BASE_DIR / "summary_report.csv")

# Q40. Reusable full EDA function
def full_eda(data):
    numerical = data.select_dtypes(include=np.number)
    print("Shape:", data.shape)
    print("\nMissing values:\n", data.isnull().sum())
    print("\nStatistics:\n", numerical.describe())
    print("\nCorrelation:\n", numerical.corr())
    print("\nSkewness:\n", numerical.skew())

    numerical.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.show()

    sns.heatmap(numerical.corr(), annot=True, cmap="coolwarm")
    plt.show()

full_eda(df)

