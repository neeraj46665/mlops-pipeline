import pandas as pd
import yaml
import joblib
from sklearn.metrics import accuracy_score

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def evaluate_model():
    params = load_params()
    model_path = params['evaluate']['model_path']
    test_data_path = params['evaluate']['test_data_path']
    
    # Load the model
    model = joblib.load(model_path)
    
    # Load the test data
    df = pd.read_csv(test_data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    evaluate_model()
