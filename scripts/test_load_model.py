import mlflow
import yaml

def test_load_model():
    # Load params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Get model URI from params.yaml
    tracking_uri = params["register"]["tracking_uri"]
    model_uri = params["register"]["model_uri"]

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Load the model
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load the model: {e}")

if __name__ == "__main__":
    test_load_model()
