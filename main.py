import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data, scaled_X
from models import train_lda, train_mlp, evaluate_model
from visualization import plot_cross_correlation_heatmap, print_top_correlations_with_target
from visualization import print_mutual_information

df = pd.read_csv('data/industrial_robot_control_6G_network.csv')

df_model = preprocess_data(df)
X = df_model.drop(columns="task_type")
y = df_model["task_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train_scaled, X_test_scaled = scaled_X(X_train, X_test)

#LDA
lda = train_lda(X_train_scaled, y_train)
lda_accuracy, lda_matrix = evaluate_model(lda, X_test_scaled, y_test)

print("LDA Accuracy:")
print(lda_accuracy)

print("LDA Confusion Matrix:")
print(lda_matrix)

#MLP
mlp = train_mlp(X_train_scaled, y_train)
mlp_accuracy, mlp_matrix = evaluate_model(mlp, X_test_scaled, y_test)

print("MLP Accuracy:")
print(mlp_accuracy)

print("MLP Confusion Matrix:")
print(mlp_matrix)


print_top_correlations_with_target(X, y)
# As you can see there is very little correlation
plot_cross_correlation_heatmap(X)
# Cross-correlation analysis shows generally weak relationships between most input features.
print_mutual_information(X, y)
# Slightly more information than Pearson, but still low.