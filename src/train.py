import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load parameters from YAML file
def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

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
    
    # Save the best model
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(best_model, model_output)
    print(f"Model trained and saved to '{model_output}'")

if __name__ == "__main__":
    train_model()
