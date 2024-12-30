import pytest
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Test function to load the model and check its accuracy
def test_model_accuracy():
    # Load the dataset
    df = pd.read_csv('data/iris.csv')
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target
    
    # Split the dataset into training and testing sets (same split used for training the model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the pre-trained model
    model = joblib.load('models/iris_model.pkl')
    
    # Test model accuracy (you can improve this test by validating a specific result)
    accuracy = model.score(X_test, y_test)
    
    # Assert that the model's accuracy is greater than 80% (customizable based on your expectation)
    assert accuracy > 0.8, f"Model accuracy {accuracy} is less than expected"

# Test to check if the model is loaded properly
def test_model_loading():
    try:
        model = joblib.load('models\iris_model.pkl')
        assert model is not None
    except FileNotFoundError:
        pytest.fail("Model not found. Ensure that the model is trained and saved properly.")

test_model_loading()
test_model_accuracy()
