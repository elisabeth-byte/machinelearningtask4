import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

def plot_cross_correlation_heatmap(X):
    # Correlation between input features
    matrix = X.corr(numeric_only=True)

    plt.figure(figsize=(20,16))

    sns.heatmap(
        matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Cross Correlation Matrix")
    plt.show()


def print_top_correlations_with_target(X, y):
    y_numeric = y.map({
        "Assembly": 0,
        "Pick-and-Place": 1,
        "Welding": 2
    })
    # Add target temporarily to X
    df_corr = X.copy()
    df_corr["task_type_Numeric"] = y_numeric

    # Correlation matrix
    corr = df_corr.corr(numeric_only=True)

    # Correlation between input features and output
    target_corr = corr["task_type_Numeric"].drop("task_type_Numeric")

    # Sort by strongest absolute correlation
    target_corr = target_corr.reindex(
        target_corr.abs().sort_values(ascending=False).index
    )

    print("Pearson correlations with Task Type:")
    print(target_corr)

def print_mutual_information(X, y):
    mi = mutual_info_classif(X, y, random_state=42)

    mi_scores = pd.Series(mi, index=X.columns)

    mi_scores = mi_scores.sort_values(ascending=False)

    print("Mutual Information with task_type:")
    print(mi_scores)