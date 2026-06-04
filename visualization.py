import matplotlib.pyplot as plt
import seaborn as sns


def plot_cross_correlation_heatmap(X):
    # Correlation between input features
    matrix = X.corr(numeric_only=True)

    plt.figure(figsize=(40,32))

    sns.heatmap(
        matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Cross Correlation Matrix")
    plt.show()


def print_top_correlations_with_target(X, y):

    # Add target temporarily to X
    df_corr = X.copy()
    df_corr["Fault_Type_Numeric"] = y

    # Correlation matrix
    corr = df_corr.corr(numeric_only=True)

    # Correlation between input features and output
    target_corr = corr["Fault_Type_Numeric"].drop("Fault_Type_Numeric")

    # Sort by strongest absolute correlation
    target_corr = target_corr.reindex(
        target_corr.abs().sort_values(ascending=False).index
    )

    print("Correlations with Fault Type:")
    print(target_corr)