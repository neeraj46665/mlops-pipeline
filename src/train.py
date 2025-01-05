import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Load parameters from YAML file
def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

params = load_params()

# Read MLflow tracking URI and other registration parameters
tracking_uri = params["register"]["tracking_uri"]
mlflow.set_tracking_uri(tracking_uri)

# Train the model with improved code
def train_model():
    # Load the parameters
    params = load_params()
    input_path = params['train']['input_path']
    model_output = params['train']['model_output']
    
    # Load the dataset
    df = pd.read_csv(input_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
    )
    
    # Initialize the model (Decision Tree Classifier)
    model = DecisionTreeClassifier(random_state=params['train']['random_state'])

    # Set up the hyperparameters grid for GridSearchCV
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],  # Updated to valid values
    }

    # Perform GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Evaluate the model on test data
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Log the model using MLflow
    mlflow.start_run()  # Start a new MLflow run

    # Infer the model signature
    # signature = infer_signature(X_train, best_model.predict(X_train))

    # Log the model with the inferred signature
    # mlflow.sklearn.log_model(best_model, "model", signature=signature)
    mlflow.sklearn.log_model(best_model, "model")

    # Optionally log parameters and metrics
    mlflow.log_param("max_depth", grid_search.best_params_["max_depth"])
    mlflow.log_param("min_samples_split", grid_search.best_params_["min_samples_split"])
    mlflow.log_metric("accuracy", accuracy)
    
    # Save the model to disk (optional, for local use as well)
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(best_model, model_output)
    print(f"Model trained and saved to '{model_output}'")

    run_id = mlflow.active_run().info.run_id
    print(f"Model logged with Run ID: {run_id}")

    # Save the run_id to a YAML file
    with open('run_id.yaml', 'w') as f:
        yaml.dump({"run_id": run_id}, f)

if __name__ == "__main__":
    train_model()
