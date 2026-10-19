"""
Rainfall EDA Project
Run: python rainfall_eda.py
Place rainfall.csv in the same folder as this script.
Charts and reports are saved in the outputs/ folder.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "rainfall.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def find_column(data, candidates, required=True):
    """Find a column without being sensitive to capitalization or spaces."""
    normalized = {str(col).strip().lower().replace(" ", "_"): col for col in data.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {data.columns.tolist()}\n"
            "Update the column aliases near the start of the script."
        )
    return None


def detect_outliers_iqr(data, column):
    """Return rows that are outliers in the selected column using the IQR rule."""
    values = pd.to_numeric(data[column], errors="coerce")
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(values < lower) | (values > upper)]


def full_eda(data, rainfall_col, temperature_col=None, humidity_col=None):
    """Run a reusable basic + correlation + distribution EDA workflow."""
    numerical = data.select_dtypes(include=np.number)

    print("\n--- FULL EDA REPORT ---")
    print("\nFirst 5 rows:\n", data.head())
    print("\nDataset shape (rows, columns):", data.shape)
    print("\nData types:\n", data.dtypes)
    print("\nMissing values:\n", data.isnull().sum())
    print("\nNumerical summary:\n", numerical.describe())
    print("\nCorrelation matrix:\n", numerical.corr())
    print("\nSkewness:\n", numerical.skew())

    if not numerical.empty:
        numerical.hist(figsize=(14, 10))
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "full_eda_histograms.png", dpi=150)
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.heatmap(numerical.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "full_eda_correlation_heatmap.png", dpi=150)
        plt.show()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place your CSV file in this project folder and name it rainfall.csv."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()

    print("Loaded dataset:", DATA_PATH.name)
    print("Columns found:", df.columns.tolist())

    # Common column-name alternatives. Add your dataset's names here if needed.
    rainfall_col = find_column(
        df, ["rainfall", "rain", "precipitation", "rainfall_mm", "rain_mm"]
    )
    temperature_col = find_column(
        df, ["temperature", "temp", "temperature_c", "temp_c"], required=False
    )
    humidity_col = find_column(
        df, ["humidity", "relative_humidity", "humidity_percent"], required=False
    )
    weather_col = find_column(
        df, ["weather_condition", "weather", "condition", "weather_type"],
        required=False
    )

    num_df = df.select_dtypes(include=np.number)
    if rainfall_col not in num_df.columns:
        raise ValueError(
            f"The rainfall column '{rainfall_col}' is not numeric. "
            "Check the CSV values and remove units such as 'mm' from the data."
        )

    # Q1. Display first 10 rows
    print("\nQ1. First 10 rows:\n", df.head(10))

    # Q2. Number of rows and columns
    print("\nQ2. Rows and columns:", df.shape)

    # Q3. Print all column names
    print("\nQ3. Column names:", df.columns.tolist())

    # Q4. Data types of each column
    print("\nQ4. Data types:\n", df.dtypes)

    # Q5. Missing values in each column
    print("\nQ5. Missing values:\n", df.isnull().sum())

    # Q6. Mean, median and standard deviation of rainfall
    print("\nQ6. Rainfall mean:", df[rainfall_col].mean())
    print("Rainfall median:", df[rainfall_col].median())
    print("Rainfall standard deviation:", df[rainfall_col].std())

    # Q7. Maximum and minimum temperature
    if temperature_col:
        print("\nQ7. Maximum temperature:", df[temperature_col].max())
        print("Minimum temperature:", df[temperature_col].min())
    else:
        print("\nQ7. Skipped: no temperature column was detected.")

    # Q8. Numerical columns
    print("\nQ8. Numerical columns:\n", num_df.head())
    num_df.to_csv(OUTPUT_DIR / "numerical_columns.csv", index=False)

    # Q9. Correlation matrix
    corr = num_df.corr()
    print("\nQ9. Correlation matrix:\n", corr)
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # Q10. Correlation heatmap
    if not corr.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Q10. Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "q10_correlation_heatmap.png", dpi=150)
        plt.show()

    # Q11. Correlation of all numerical features with rainfall, sorted
    print("\nQ11. Correlation with rainfall:\n",
          corr[rainfall_col].sort_values(ascending=False))

    # Q12. Feature pairs with correlation > 0.7
    upper_triangle = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(upper_triangle).stack()
    high_pairs = pairs[pairs > 0.7]
    print("\nQ12. Feature pairs with correlation > 0.7:\n", high_pairs)

    # Q13. Features appearing in highly correlated pairs
    features_list = sorted({
        feature for pair in high_pairs.index for feature in pair
    })
    print("\nQ13. Highly correlated features list:", features_list)

    # Q14. Drop the second feature from each highly correlated pair
    to_drop = sorted({pair[1] for pair in high_pairs.index})
    reduced_df = num_df.drop(columns=to_drop, errors="ignore")
    print("\nQ14. Columns dropped:", to_drop)
    print("Reduced numerical dataset shape:", reduced_df.shape)
    reduced_df.to_csv(OUTPUT_DIR / "reduced_numerical_dataset.csv", index=False)

    # Q15. Scatter plot between rainfall and humidity
    if humidity_col:
        sns.scatterplot(data=df, x=humidity_col, y=rainfall_col)
        plt.title("Q15. Rainfall vs Humidity")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "q15_rainfall_humidity_scatter.png", dpi=150)
        plt.show()
    else:
        print("\nQ15. Skipped: no humidity column was detected.")

    # Q16. Histograms for all numerical features
    if not num_df.empty:
        num_df.hist(figsize=(14, 10), bins=20)
        plt.suptitle("Q16. Histograms of Numerical Features")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "q16_all_histograms.png", dpi=150)
        plt.show()

    # Q17. Rainfall histogram with 30 bins
    plt.figure()
    plt.hist(df[rainfall_col].dropna(), bins=30, edgecolor="black")
    plt.xlabel(rainfall_col)
    plt.ylabel("Frequency")
    plt.title("Q17. Rainfall Distribution (30 bins)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q17_rainfall_histogram.png", dpi=150)
    plt.show()

    # Q18. Boxplots for all numerical columns
    if not num_df.empty:
        axes = num_df.plot(kind="box", subplots=True, layout=(-1, 3),
                           figsize=(14, 4 * max(1, int(np.ceil(len(num_df.columns) / 3)))))
        plt.suptitle("Q18. Boxplots of Numerical Features")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "q18_boxplots.png", dpi=150)
        plt.show()

    # Q19. Outlier counts using IQR
    q1 = num_df.quantile(0.25)
    q3 = num_df.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (num_df < (q1 - 1.5 * iqr)) | (num_df > (q3 + 1.5 * iqr))
    print("\nQ19. Outlier count per numerical column:\n", outlier_mask.sum())

    # Q20. Skewness of numerical features
    print("\nQ20. Skewness:\n", num_df.skew())

    # Q21. KDE plot for temperature
    if temperature_col:
        sns.kdeplot(data=df, x=temperature_col, fill=True)
        plt.title("Q21. Temperature KDE")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "q21_temperature_kde.png", dpi=150)
        plt.show()
    else:
        print("\nQ21. Skipped: no temperature column was detected.")

    # Q22. Compare mean and median rainfall
    rainfall_mean = df[rainfall_col].mean()
    rainfall_median = df[rainfall_col].median()
    print("\nQ22. Rainfall mean:", rainfall_mean, "| median:", rainfall_median)

    # Q23. Count right-skewed numerical features
    skew_values = num_df.skew()
    print("\nQ23. Number of right-skewed features:", int((skew_values > 0).sum()))

    # Q24. Average rainfall grouped by weather condition
    if weather_col:
        print("\nQ24. Average rainfall by weather condition:\n",
              df.groupby(weather_col)[rainfall_col].mean())
    else:
        print("\nQ24. Skipped: no weather-condition column was detected.")

    # Q25. Top 3 records with highest rainfall
    print("\nQ25. Top 3 rainfall records:\n", df.nlargest(3, rainfall_col))

    # Q26. Rows where rainfall is greater than average
    print("\nQ26. Records with above-average rainfall:\n",
          df[df[rainfall_col] > rainfall_mean])

    # Q27. Average temperature for high vs low rainfall
    if temperature_col:
        high_temp = df.loc[df[rainfall_col] > rainfall_mean, temperature_col].mean()
        low_temp = df.loc[df[rainfall_col] <= rainfall_mean, temperature_col].mean()
        print("\nQ27. Mean temperature when rainfall is high:", high_temp)
        print("Mean temperature when rainfall is low:", low_temp)
    else:
        print("\nQ27. Skipped: no temperature column was detected.")

    # Q28. Add high_rainfall (1 if above mean, otherwise 0)
    df["high_rainfall"] = (df[rainfall_col] > rainfall_mean).astype(int)
    print("\nQ28. Dataset with high_rainfall column:\n", df.head())

    # Q29. Count high-rainfall records
    print("\nQ29. High-rainfall counts:\n", df["high_rainfall"].value_counts().sort_index())

    # Q30. Correlation for high-rainfall records
    high_df = df[df["high_rainfall"] == 1]
    print("\nQ30. Correlation for high-rainfall records:\n",
          high_df.select_dtypes(include=np.number).corr())

    # Q31. Reusable IQR outlier function
    print("\nQ31. Outliers in rainfall:\n", detect_outliers_iqr(df, rainfall_col))

    # Q32. Remove rainfall outliers
    rainfall_outliers = detect_outliers_iqr(df, rainfall_col)
    q1_r = df[rainfall_col].quantile(0.25)
    q3_r = df[rainfall_col].quantile(0.75)
    iqr_r = q3_r - q1_r
    lower_r, upper_r = q1_r - 1.5 * iqr_r, q3_r + 1.5 * iqr_r
    clean_df = df[df[rainfall_col].between(lower_r, upper_r)].copy()
    print(f"\nQ32. Rows before: {len(df)}, after removing rainfall outliers: {len(clean_df)}")
    clean_df.to_csv(OUTPUT_DIR / "rainfall_without_outliers.csv", index=False)

    # Q33. Min-Max normalization
    scaler_minmax = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler_minmax.fit_transform(num_df),
        columns=num_df.columns,
        index=num_df.index
    )
    print("\nQ33. Min-Max normalized data:\n", normalized.head())
    normalized.to_csv(OUTPUT_DIR / "minmax_normalized.csv", index=False)

    # Q34. Z-score standardization
    scaler_z = StandardScaler()
    standardized = pd.DataFrame(
        scaler_z.fit_transform(num_df),
        columns=num_df.columns,
        index=num_df.index
    )
    print("\nQ34. Z-score standardized data:\n", standardized.head())
    standardized.to_csv(OUTPUT_DIR / "zscore_standardized.csv", index=False)

    # Q35. Pairplot for available selected features
    selected = [col for col in [rainfall_col, temperature_col, humidity_col] if col]
    if len(selected) >= 2:
        pair_data = df[selected].dropna()
        if len(pair_data) > 500:
            pair_data = pair_data.sample(500, random_state=42)
        sns.pairplot(pair_data)
        plt.savefig(OUTPUT_DIR / "q35_pairplot.png", dpi=150)
        plt.show()
    else:
        print("\nQ35. Skipped: at least two selected numerical features are needed.")

    # Q36. Features with correlation > 0.5 with rainfall (excluding rainfall itself)
    rainfall_corr = corr[rainfall_col]
    related = rainfall_corr[
        (rainfall_corr > 0.5) & (rainfall_corr.index != rainfall_col)
    ].sort_values(ascending=False)
    print("\nQ36. Features with correlation > 0.5 with rainfall:\n", related)

    # Q37. Check class balance for the generated target variable
    counts = df["high_rainfall"].value_counts()
    percentages = df["high_rainfall"].value_counts(normalize=True).mul(100)
    print("\nQ37. Target counts:\n", counts)
    print("Target percentages:\n", percentages)
    if len(percentages) == 2 and percentages.min() < 30:
        print("Possible class imbalance: one class represents less than 30% of records.")
    else:
        print("No strong imbalance detected by the simple 30% rule.")

    # Q38. Distribution before and after removing rainfall outliers
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[rainfall_col].dropna(), ax=axes[0])
    axes[0].set_title("Before Removing Outliers")
    sns.histplot(clean_df[rainfall_col].dropna(), ax=axes[1])
    axes[1].set_title("After Removing Outliers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q38_before_after_outliers.png", dpi=150)
    plt.show()

    # Q39. Summary report
    report = pd.DataFrame({
        "mean": num_df.mean(),
        "skewness": num_df.skew(),
        "outliers_count": outlier_mask.sum()
    })
    print("\nQ39. Summary report:\n", report)
    report.to_csv(OUTPUT_DIR / "q39_summary_report.csv")

    # Q40. Reusable full EDA function
    full_eda(df, rainfall_col, temperature_col, humidity_col)

    print(f"\nAll questions completed. Results are saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
