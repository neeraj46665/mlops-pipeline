from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model and scaler
MODEL_PATH = "models/iris_model.pkl"
SCALER_PATH = "models/scaler.pkl"

try:
    # Loading model using pickle
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Check if model is a valid scikit-learn model
    if not hasattr(model, "predict"):
        raise RuntimeError("Loaded model is not a valid scikit-learn model.")
except FileNotFoundError:
    raise RuntimeError(f"Model or scaler not found at the specified paths.")
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
