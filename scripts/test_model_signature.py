import pytest
import mlflow
import yaml
import pandas as pd
import numpy as np
import joblib

import warnings

# Suppress specific deprecation warnings
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


# Pytest fixture for setting up the model and scaler
@pytest.fixture(scope="module")
def setup_model_and_scaler():
    """Setup model and scaler for tests."""
    params = load_params()
    tracking_uri = params["register"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    # Get run_id and model details
    run_id = load_run_id()
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the scaler
    scaler_path = "models/scaler.pkl"
    scaler = joblib.load(scaler_path)

    return model, scaler


def test_model_signature(setup_model_and_scaler):
    """Test that the model signature matches expected schema."""
    model, _ = setup_model_and_scaler

    signature = model.metadata.signature
    assert signature is not None, "Model signature is not defined."

    # Validate input columns
    expected_inputs = signature.inputs
    actual_columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    assert len(expected_inputs) == len(actual_columns), "Input column count mismatch."
    for col in expected_inputs:
        assert col.name in actual_columns, f"Missing input column: {col.name}"

    print("Model signature validated successfully.")





def test_model_with_incorrect_schema(setup_model_and_scaler):
    """Test the model with incorrect input schema."""
    model, scaler = setup_model_and_scaler

    # Create a DataFrame with incorrect schema
    incorrect_dummy_data = pd.DataFrame({
        "sepal length (cm)": np.random.uniform(4.3, 7.9, size=5),
        "petal length (cm)": np.random.uniform(1.0, 6.9, size=5)
    })

    with pytest.raises(ValueError):
        scaled_data = scaler.transform(incorrect_dummy_data)
        model.predict(scaled_data)
    print("Schema validation test passed for incorrect schema.")



if __name__ == "__main__":
    pytest.main(["-v"])
