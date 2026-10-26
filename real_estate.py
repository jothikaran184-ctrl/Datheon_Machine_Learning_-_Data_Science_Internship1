# CALIFORNIA HOUSING PRICE PREDICTION
# Models: OLS, Ridge Regression, Lasso Regression
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

housing = fetch_california_housing(as_frame=True)

X = housing.data.copy()
y = housing.target.copy()

df = X.copy()
df["MedHouseVal"] = y

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values per column:")
print(X.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe().round(3))

# ------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

# Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(y, bins=40, kde=True)
plt.title("Distribution of Median House Values")
plt.xlabel("Median house value (units of $100,000)")
plt.tight_layout()
plt.show()

# Correlation heatmap on original features
plt.figure(figsize=(11, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
)
plt.title("California Housing Correlation Heatmap")
plt.tight_layout()
plt.show()

# Correlation matrix for original predictors only
feature_corr = X.corr().abs()
upper_triangle = feature_corr.where(
    np.triu(np.ones(feature_corr.shape), k=1).astype(bool)
)

high_corr_pairs = (
    upper_triangle.stack()
    .loc[lambda s: s > 0.80]
    .sort_values(ascending=False)
)

print("\nOriginal feature pairs with |correlation| > 0.80:")
print(high_corr_pairs if not high_corr_pairs.empty
      else "No pairs above threshold.")

# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# Split before fitting any preprocessing steps.
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))

# ------------------------------------------------------------
# 4. PREPROCESSING PIPELINE
# Impute -> create pairwise interactions -> standardize.
# PolynomialFeatures retains original features and adds
# pairwise interaction terms, without squared terms.
# ------------------------------------------------------------

def make_pipeline(regressor):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "interactions",
            PolynomialFeatures(
                degree=2,
                interaction_only=True,
                include_bias=False,
            ),
        ),
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


# ------------------------------------------------------------
# 5. DEFINE AND TUNE MODELS
# All hyperparameter selection uses training data only.
# ------------------------------------------------------------

cv = KFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# Ordinary Least Squares baseline
ols = make_pipeline(LinearRegression())
ols.fit(X_train, y_train)

# Ridge: L2 regularization
ridge_pipeline = make_pipeline(Ridge())

ridge_search = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid={
        "regressor__alpha": [
            0.01, 0.1, 1.0, 10.0, 100.0, 1000.0
        ]
    },
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

ridge_search.fit(X_train, y_train)

# Lasso: L1 regularization
lasso_pipeline = make_pipeline(
    Lasso(max_iter=20000, tol=1e-4)
)

lasso_search = GridSearchCV(
    estimator=lasso_pipeline,
    param_grid={
        "regressor__alpha": [
            0.0001, 0.001, 0.01, 0.1, 1.0, 10.0
        ]
    },
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

lasso_search.fit(X_train, y_train)

models = {
    "OLS": ols,
    "Ridge": ridge_search.best_estimator_,
    "Lasso": lasso_search.best_estimator_,
}

print("\nBest Ridge parameters:", ridge_search.best_params_)
print("Best Ridge CV RMSE:", -ridge_search.best_score_)

print("\nBest Lasso parameters:", lasso_search.best_params_)
print("Best Lasso CV RMSE:", -lasso_search.best_score_)

# ------------------------------------------------------------
# 6. TEST-SET EVALUATION
# RMSE and MAE are in units of $100,000.
# ------------------------------------------------------------

results = []
predictions = {}
residuals = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    err = y_test.to_numpy() - y_pred

    predictions[name] = y_pred
    residuals[name] = err

    results.append({
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    })

results_df = pd.DataFrame(results).sort_values("RMSE")

print("\n=== TEST-SET BENCHMARK ===")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "california_housing_model_comparison.csv",
    index=False,
)

# ------------------------------------------------------------
# 7. PREDICTED VS ACTUAL VALUES
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(
        y_test,
        y_pred,
        alpha=0.35,
        s=12,
    )

    low = min(y_test.min(), y_pred.min())
    high = max(y_test.max(), y_pred.max())

    ax.plot(
        [low, high],
        [low, high],
        "r--",
        linewidth=2,
    )

    ax.set_title(name)
    ax.set_xlabel("Actual median value")
    ax.set_ylabel("Predicted median value")

fig.suptitle("Predicted vs Actual House Values")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. RESIDUAL DISTRIBUTIONS
# Residual = actual - predicted
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (name, err) in zip(axes, residuals.items()):
    sns.histplot(
        err,
        bins=40,
        kde=True,
        stat="density",
        ax=ax,
    )

    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"{name} Residual Distribution")
    ax.set_xlabel("Residual (actual - predicted)")

plt.tight_layout()
plt.show()

# Residuals against fitted predictions
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_pred, residuals[name], alpha=0.35, s=12)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(f"{name}: Residuals vs Predictions")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. REGULARIZATION AND FEATURE SELECTION
# Count zero coefficients for the fitted Lasso model.
# ------------------------------------------------------------

lasso_model = models["Lasso"]
lasso_coef = lasso_model.named_steps["regressor"].coef_

print("\nLasso coefficient summary:")
print("Total engineered features:", len(lasso_coef))
print("Zero coefficients:", np.sum(np.isclose(lasso_coef, 0)))
print(
    "Non-zero coefficients:",
    np.sum(~np.isclose(lasso_coef, 0)),
)

# ------------------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------------------

best = results_df.iloc[0]

print("\n=== BEST MODEL BY TEST RMSE ===")
print("Model:", best["Model"])
print("RMSE:", round(best["RMSE"], 4))
print("MAE:", round(best["MAE"], 4))
print("R2:", round(best["R2"], 4))

print("\nSaved results to:")
print("california_housing_model_comparison.csv")