from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import joblib
import numpy as np
from mlflow.pyfunc import load_model
import yaml
import numpy as np
import pandas as pd

# Load the trained model and scaler
# MODEL_PATH = "models/iris_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_production_model():
    try:
        # Set the tracking URI to your MLflow server
        params = load_params()
        tracking_uri = params["register"]["tracking_uri"]
        mlflow.set_tracking_uri(tracking_uri)

        # Model name registered in MLflow
        model_name = params["register"]["model_name"]
        
        # Get the latest version of the model in the 'production' stage
        client = mlflow.tracking.MlflowClient()
        production_model = client.get_latest_versions(model_name, stages=["Production"])[0]
        model_version = production_model.version

        # Load the model using MLflow's pyfunc API
        model_uri = f"models:/{model_name}/{model_version}"
        model = load_model(model_uri)

        return model
    except Exception as e:
        raise RuntimeError(f"Error loading the model from MLflow: {str(e)}")


try:
    # Loading model using pickle
    # model = joblib.load(MODEL_PATH)
    model = load_production_model()

    scaler = joblib.load(SCALER_PATH)

    # Check if model is a valid scikit-learn model
    if not hasattr(model, "predict"):
        raise RuntimeError("Loaded model is not a valid scikit-learn model.")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

except Exception as e:
    raise RuntimeError(f"Error loading the model or scaler: {str(e)}")

# Initialize FastAPI app
app = FastAPI(title="Iris Species Prediction API")

# Request Body Schema
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Response Schema
class PredictionResponse(BaseModel):
    species: str

# Class labels for prediction
CLASS_LABELS = ['setosa', 'versicolor', 'virginica']

@app.get("/")
def home():
    return {"message": "Welcome to the Iris Species Prediction API. Use the /predict endpoint to make predictions."}

@app.post("/predict", response_model=PredictionResponse)
def predict(iris: IrisRequest):
    """
    Predict the species of the Iris flower based on input features.
    """
    # Convert input data to numpy array
    features = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])

    try:
        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)
        species = CLASS_LABELS[int(prediction[0])]
        return PredictionResponse(species=species)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the application and model status.
    """
    try:
        # Perform a dummy operation to ensure the model is loaded
        _ = model.predict(scaler.transform([[0, 0, 0, 0]]))
        return {"status": "Healthy", "message": "API and model are operational."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")



