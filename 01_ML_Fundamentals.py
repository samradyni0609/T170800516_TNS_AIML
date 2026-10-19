
# Machine Learning Fundamentals

ml_types = {
    "Supervised Learning": "Learns from labeled data.",
    "Unsupervised Learning": "Finds patterns in unlabeled data.",
    "Semi-Supervised Learning": "Uses labeled and unlabeled data.",
    "Reinforcement Learning": "Learns through rewards and penalties."
}

for name, description in ml_types.items():
    print(f"{name}: {description}")

algorithms = [
    "Linear Regression",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
    "K-Means Clustering",
    "Support Vector Machine"
]

print("\nCommon Machine Learning Algorithms:")
for algorithm in algorithms:
    print("-", algorithm)
