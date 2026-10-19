
# Supervised Learning: Classification and Regression

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score
)

# ---------------------------------
# Part A: Classification using Iris
# ---------------------------------

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

classifier = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print("CLASSIFICATION RESULTS")
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(
    y_test, predictions, target_names=iris.target_names
))

# ---------------------------------
# Part B: Regression using Diabetes
# ---------------------------------

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

print("\nREGRESSION RESULTS")
print("Mean Squared Error:",
      mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Predict a new sample
sample = X_test[:1]
print("Actual value:", y_test[0])
print("Predicted value:", regressor.predict(sample)[0])
