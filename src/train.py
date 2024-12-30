import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def train_model():
    params = load_params()
    input_path = params['train']['input_path']
    model_output = params['train']['model_output']
    
    # Load the featureized data
    df = pd.read_csv(input_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
    )
    
    # Train the model
    model = RandomForestClassifier(
        max_depth=params['train']['max_depth'],
        n_estimators=params['train']['n_estimators'],
        random_state=params['train']['random_state']
    )
    model.fit(X_train, y_train)
    
    # Save the model
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(model, model_output)
    print(f"Model trained and saved to '{model_output}'")

if __name__ == "__main__":
    train_model()
