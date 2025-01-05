import mlflow
import yaml
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

def test_model_performance():
    # Load params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Get model URI and scaler path from params.yaml
    tracking_uri = params["register"]["tracking_uri"]
    model_uri = params["register"]["model_uri"]
    scaler_path = params["featureize"]["scaler_output_path"]

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Load the model
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the scaler
    scaler = joblib.load(scaler_path)

    # Create dummy data and corresponding dummy labels
    dummy_data = pd.DataFrame({
        "sepal_length": np.random.uniform(4.3, 7.9, size=50),
        "sepal_width": np.random.uniform(2.0, 4.4, size=50),
        "petal_length": np.random.uniform(1.0, 6.9, size=50),
        "petal_width": np.random.uniform(0.1, 2.5, size=50)
    })
    dummy_labels = np.random.choice([0, 1, 2], size=50)  # 3 classes for the Iris dataset

    # Scale the dummy data
    scaled_data = scaler.transform(dummy_data)

    # Make predictions
    predictions = model.predict(scaled_data)

    # Calculate performance metrics
    accuracy = accuracy_score(dummy_labels, predictions)
    precision = precision_score(dummy_labels, predictions, average="weighted")
    recall = recall_score(dummy_labels, predictions, average="weighted")
    f1 = f1_score(dummy_labels, predictions, average="weighted")

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
