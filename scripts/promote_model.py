import yaml
import mlflow
from mlflow.tracking import MlflowClient

# Load params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Read MLflow tracking URI and other registration parameters
tracking_uri = params["register"]["tracking_uri"]
model_uri = params["register"]["model_uri"]
model_name = params["register"]["model_name"]
stage = params["register"]["stage"]

# Set the tracking URI
mlflow.set_tracking_uri(tracking_uri)

# Initialize the MLflow client
client = MlflowClient()

# Get the model version (the latest one) for promotion
latest_version = client.get_registered_model(model_name).latest_versions[0].version
print(f"Latest version of {model_name}: {latest_version}")

# Transition the model to the specified stage
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage=stage
)

print(f"Model '{model_name}' promoted to stage '{stage}'.")
