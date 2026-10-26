# ============================================================
# MULTI-CLASS CLASSIFICATION BENCHMARK: IRIS / WINE
# Models: Logistic Regression and Decision Tree
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns

# Use a non-interactive backend so the benchmark can run in terminals,
# CI environments, and VS Code without blocking on GUI windows.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def show_plot():
    """Display figures in interactive sessions but avoid blocking in headless runs."""
    if matplotlib.get_backend().lower() == "agg":
        plt.close()
        return
    plt.show()

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ------------------------------------------------------------
# 1. LOAD DATASET
# Choose "iris" or "wine"
# ------------------------------------------------------------

DATASET_NAME = "iris"

if DATASET_NAME.lower() == "iris":
    dataset = load_iris(as_frame=True)
elif DATASET_NAME.lower() == "wine":
    dataset = load_wine(as_frame=True)
else:
    raise ValueError("DATASET_NAME must be 'iris' or 'wine'.")

df = dataset.frame.copy()
feature_names = dataset.feature_names
target_names = dataset.target_names

# Separate features and target
X = df[feature_names]
y = df["target"]

# Use readable class names in visualizations
df["class"] = y.map(
    {i: name for i, name in enumerate(target_names)}
)

print(f"\nDataset: {DATASET_NAME.upper()}")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\nStatistical summary:")
print(X.describe())

# ------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")

# Class distribution
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="class", hue="class", legend=False)
plt.title(f"{DATASET_NAME.title()} - Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
show_plot()

# Pairplot: Iris has 4 features; Wine uses the first 4
# features to keep the plot readable.
pairplot_features = feature_names[:4]
pairplot_df = df[pairplot_features + ["class"]]

sns.pairplot(
    pairplot_df,
    hue="class",
    diag_kind="hist",
    corner=True,
)
plt.suptitle(
    f"{DATASET_NAME.title()} - Feature Pairplot",
    y=1.02,
)
show_plot()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    X.corr(),
    cmap="coolwarm",
    center=0,
    annot=(X.shape[1] <= 6),
    fmt=".2f",
)
plt.title(f"{DATASET_NAME.title()} - Feature Correlation")
plt.tight_layout()
show_plot()

# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT: 80% TRAIN, 20% TEST
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 4. DEFINE MODELS
# StandardScaler is fitted on training data only.
# Scaling is useful for Logistic Regression.
# Decision Trees do not require feature scaling, but using
# a pipeline keeps the benchmark structure consistent.
# ------------------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        )),
    ]),
}

# ------------------------------------------------------------
# 5. TRAIN, PREDICT AND BENCHMARK
# ------------------------------------------------------------

results = []
predictions = {}
fitted_models = {}

for model_name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predict unseen test samples
    y_pred = model.predict(X_test)

    fitted_models[model_name] = model
    predictions[model_name] = y_pred

    # Calculate classification metrics
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Macro)": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall (Macro)": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1-score (Macro)": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
    }

    results.append(metrics)

    print(f"\n{'=' * 55}")
    print(model_name)
    print("=" * 55)

    print(classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(target_names)),
        target_names=target_names,
        zero_division=0,
    ))

# Results table
results_df = pd.DataFrame(results)

print("\nBENCHMARK COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    f"{DATASET_NAME}_classification_benchmark.csv",
    index=False,
)

# ------------------------------------------------------------
# 6. METRIC COMPARISON PLOT
# ------------------------------------------------------------

metrics_to_plot = [
    "Accuracy",
    "Precision (Macro)",
    "Recall (Macro)",
    "F1-score (Macro)",
]

plot_df = results_df.melt(
    id_vars="Model",
    value_vars=metrics_to_plot,
    var_name="Metric",
    value_name="Score",
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=plot_df,
    x="Metric",
    y="Score",
    hue="Model",
)
plt.ylim(0, 1.05)
plt.title(f"{DATASET_NAME.title()} - Model Benchmark")
plt.xticks(rotation=15)
plt.tight_layout()
show_plot()

# ------------------------------------------------------------
# 7. CONFUSION MATRIX HEATMAPS
# ------------------------------------------------------------

for model_name, y_pred in predictions.items():

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(target_names)),
    )

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(f"{DATASET_NAME.title()} - {model_name}")
    plt.tight_layout()
    show_plot()

# ------------------------------------------------------------
# 8. DECISION BOUNDARY VISUALIZATION
# Use two features for a 2D visualization.
# These are separate visualization models; the benchmark
# models above use ALL features.
# ------------------------------------------------------------

boundary_features = feature_names[:2]

X_boundary = X[boundary_features].to_numpy()
y_array = y.to_numpy()

# Use the same train/test indices for the 2-feature models
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_boundary,
    y_array,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_array,
)

# Mesh over the 2D feature space
x_min, x_max = (
    X_boundary[:, 0].min() - 0.5,
    X_boundary[:, 0].max() + 0.5,
)
y_min, y_max = (
    X_boundary[:, 1].min() - 0.5,
    X_boundary[:, 1].max() + 0.5,
)

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)

grid = np.c_[xx.ravel(), yy.ravel()]

boundary_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        )),
    ]),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (model_name, model) in zip(
    axes, boundary_models.items()
):
    model.fit(Xb_train, yb_train)

    # Predict a class for each grid point
    Z = model.predict(grid).reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        cmap="viridis",
        levels=np.arange(len(target_names) + 1) - 0.5,
    )

    scatter = ax.scatter(
        Xb_test[:, 0],
        Xb_test[:, 1],
        c=yb_test,
        cmap="viridis",
        edgecolors="black",
        s=45,
    )

    ax.set_xlabel(boundary_features[0])
    ax.set_ylabel(boundary_features[1])
    ax.set_title(f"{model_name} - 2D Decision Boundary")

plt.tight_layout()
show_plot()

# ------------------------------------------------------------
# 9. BEST MODEL SUMMARY
# ------------------------------------------------------------

best_row = results_df.loc[results_df["F1-score (Macro)"].idxmax()]

print("\nBEST MODEL BY MACRO F1-SCORE")
print("Model:", best_row["Model"])
print("Accuracy:", round(best_row["Accuracy"], 4))
print("Macro Precision:", round(best_row["Precision (Macro)"], 4))
print("Macro Recall:", round(best_row["Recall (Macro)"], 4))
print("Macro F1-score:", round(best_row["F1-score (Macro)"], 4))

print(
    f"\nResults saved to "
    f"{DATASET_NAME}_classification_benchmark.csv"
)