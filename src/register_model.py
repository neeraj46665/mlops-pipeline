import yaml
import mlflow

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

params = load_params()
tracking_uri = params["register"]["tracking_uri"]
mlflow.set_tracking_uri(tracking_uri)

# Get the saved run_id
run_id = load_run_id()

# Register the model
model_name = params["register"]["model_name"]
stage = params["register"]["stage"]

model_uri = f"runs:/{run_id}/model"

# Register the model
result = mlflow.register_model(model_uri=model_uri, name=model_name)

# Transition model to the specified stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage=stage
)

print(f"Model {model_name} registered and transitioned to {stage} stage.")
