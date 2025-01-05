import mlflow
import yaml
import pytest
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore")


# Load parameters from YAML file
def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

# Load the run_id from the saved file
def load_run_id():
    with open('run_id.yaml', 'r') as f:
        data = yaml.safe_load(f)
    return data["run_id"]

# Function to load and test the model
def load_model(model_uri):
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        raise Exception(f"Failed to load the model: {e}")

# Pytest test function to test loading the model
def test_load_model():
    params = load_params()
    tracking_uri = params["register"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    # Get the saved run_id
    run_id = load_run_id()

    # Prepare the model URI
    model_name = params["register"]["model_name"]
    stage = params["register"]["staging_stage"]
    model_uri = f"runs:/{run_id}/model"

    # Test loading the model
    model = load_model(model_uri)
    assert model is not None, "Model loading failed"

# Run tests using pytest
if __name__ == "__main__":
    pytest.main()
