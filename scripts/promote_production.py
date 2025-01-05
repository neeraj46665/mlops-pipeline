import warnings
import yaml
import mlflow
from mlflow.tracking import MlflowClient

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Run tests on the model (you need to define your test logic here)
def run_tests():
    # Placeholder for actual testing logic
    print("Running tests on the model...")
    # Assuming tests pass for now, you can add actual test execution code
    return True

# Promote model to production after tests pass
def promote_to_production():
    # Load parameters
    params = load_params()
    tracking_uri = params["register"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    # Get the saved run_id
    run_id = load_run_id()

    # Initialize the MLflow client
    client = MlflowClient()

    # Prepare the model URI from the run_id
    model_uri = f"runs:/{run_id}/model"
    model_name = params["register"]["model_name"]
    stage = params["register"]["production_stage"]

    # Get the latest version in the "staging" stage
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version
    print(f"Latest version in staging: {latest_version_staging}")

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        print(f"Archiving production model version {version.version}")
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

# Main flow
def main():
    if run_tests():  # Step 1: Run tests
        promote_to_production()  # Step 2: Promote to production if tests pass
    else:
        print("Tests failed. Model promotion to production aborted.")

# Execute the main function
if __name__ == "__main__":
    main()
