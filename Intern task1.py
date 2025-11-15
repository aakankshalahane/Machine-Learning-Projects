# ---------------------------------------------
# TASK 1: Predict Restaurant Aggregate Rating
# ---------------------------------------------

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
# Replace with your dataset filename
df = pd.read_csv("restaurant_data.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 3. Basic Data Understanding
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

# 4. Handle Missing Values
# Numeric: fill with median | Categorical: fill with mode
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\n✅ Missing values handled successfully.")

# 5. Encode Categorical Variables
# Label Encoding for simple categorical features
label_enc = LabelEncoder()
for col in cat_cols:
    df[col] = label_enc.fit_transform(df[col].astype(str))

print("\n✅ Categorical columns encoded.")

# 6. Define Features (X) and Target (y)
# Replace 'aggregate_rating' with your actual target column name
target = 'aggregate_rating'
X = df.drop(columns=[target])
y = df[target]

# 7. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# 8. Choose Regression Algorithms
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# 9. Train and Evaluate Models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append([name, mse, r2])
    print(f"\n🔹 {name}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

# 10. Compare Model Performances
results_df = pd.DataFrame(results, columns=["Model", "MSE", "R2_Score"])
print("\nModel Comparison:")
print(results_df)

# 11. Visualize R2 Scores
plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="R2_Score", data=results_df, palette="viridis")
plt.title("Model Comparison (R² Score)")
plt.show()

# 12. Interpret Feature Importance (for tree-based models)
best_model = models["Random Forest"]
importances = pd.Series(best_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=importances.index, palette="magma")
plt.title("Most Influential Features on Restaurant Rating")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.show()
