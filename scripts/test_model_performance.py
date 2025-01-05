import mlflow
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pytest

import warnings

warnings.filterwarnings("ignore")

# Helper function to load parameters
def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

# Helper function to load run_id
def load_run_id():
    with open("run_id.yaml", "r") as f:
        data = yaml.safe_load(f)
    return data["run_id"]

def test_model_performance():
    # Load params.yaml
    params = load_params()
    tracking_uri = params["register"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    # Get run_id and model details
    run_id = load_run_id()
    model_uri = f"runs:/{run_id}/model"

    # Get model URI and scaler path from params.yaml
    scaler_path = params["featureize"]["scaler_output_path"]

    # Load the model
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Create dummy data and corresponding dummy labels
    df = pd.read_csv(r'data/iris.csv')  # Use raw string or forward slashes
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target
    
    # Split the dataset into training and testing sets (same split used for training the model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the dummy data
    X_test_scaled = scaler.transform(X_test)

    # Ensure correct column names for the input
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    # Test model accuracy
    predictions = model.predict(X_test_scaled_df)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    # Define expected thresholds for the performance metrics
    expected_accuracy = 0.40
    expected_precision = 0.40
    expected_recall = 0.40
    expected_f1 = 0.40

    # Test if the model meets the expected thresholds
    assert accuracy >= expected_accuracy, f"Accuracy {accuracy:.2f} is below the expected threshold {expected_accuracy:.2f}."
    assert precision >= expected_precision, f"Precision {precision:.2f} is below the expected threshold {expected_precision:.2f}."
    assert recall >= expected_recall, f"Recall {recall:.2f} is below the expected threshold {expected_recall:.2f}."
    assert f1 >= expected_f1, f"F1 score {f1:.2f} is below the expected threshold {expected_f1:.2f}."

    print("Model performance tests passed.")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

if __name__ == "__main__":
    test_model_performance()
