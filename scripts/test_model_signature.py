import mlflow
import yaml
import pandas as pd
import numpy as np
import joblib

def test_model_signature():
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

    # Create dummy data for the Iris dataset
    dummy_data = pd.DataFrame({
        "sepal_length": np.random.uniform(4.3, 7.9, size=5),
        "sepal_width": np.random.uniform(2.0, 4.4, size=5),
        "petal_length": np.random.uniform(1.0, 6.9, size=5),
        "petal_width": np.random.uniform(0.1, 2.5, size=5)
    })

    # Scale the data using the loaded scaler
    scaled_data = scaler.transform(dummy_data)

    # Validate model signature
    signature = model.metadata.signature
    if signature:
        # Check if inputs match signature
        expected_inputs = signature.inputs
        actual_inputs = list(dummy_data.columns)
        if all(col.name in actual_inputs for col in expected_inputs):
            print("Input signature matches dummy data.")
        else:
            print("Input signature does not match dummy data.")

        # Check if outputs match signature
        expected_outputs = signature.outputs
        print("Model signature validated successfully.")
        print(f"Expected Inputs: {expected_inputs}")
        print(f"Expected Outputs: {expected_outputs}")

        # Test prediction with scaled data
        predictions = model.predict(scaled_data)
        print("Predictions on dummy data:", predictions)
    else:
            print("Model signature is not defined.")

if __name__ == "__main__":
    test_model_signature()
