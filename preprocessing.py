import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/industrial_robot_control_6G_network.csv')
#The
def preprocess_data(df):

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["minute"] = df["timestamp"].dt.minute

    df = df.drop(columns=[
        "robot_id",
        "position_coordinates",
        "timestamp",
    ])

    df = pd.get_dummies(
        df,
        columns=["sensor_id", "sensor_type", "network_type", "slice_id"],
        drop_first=True
    )

    return df

def scaled_X(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
