import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. Initialize the FastAPI application
app = FastAPI(
    title="House Price Predictor API",
    description="API to predict house prices using a trained Linear Regression model.",
    version="1.0.0"
)

# 2. Enable CORS (Critical for React to talk to FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the pre-trained Machine Learning model
# We load it globally when the server starts so it's instantly ready for predictions
try:
    model = joblib.load('../models/linear_regression_model.joblib')
except Exception as e:
    print("Warning: Could not load the model. Have you run train.py yet?")
    model = None

# 4. Define the Expected Input (Pydantic Validation)
# This perfectly matches the 5 features from our PRD
class HouseFeatures(BaseModel):
    LotArea: int = Field(..., gt=0, description="Area of the lot in square feet")
    BedroomAbvGr: int = Field(..., ge=0, description="Number of bedrooms above ground")
    FullBath: int = Field(..., ge=0, description="Number of full bathrooms")
    HouseStyle: str = Field(..., description="Style of the house (e.g., 1Story, 2Story)")
    Age: int = Field(..., ge=0, description="Age of the house in years")

# 5. Define the Prediction Endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Convert the incoming JSON into a Pandas DataFrame
        # Our scikit-learn Pipeline expects a DataFrame with these exact column names
        input_data = pd.DataFrame([{
            'LotArea': features.LotArea,
            'BedroomAbvGr': features.BedroomAbvGr,
            'FullBath': features.FullBath,
            'HouseStyle': features.HouseStyle,
            'Age': features.Age
        }])
        
        # Ask the model for a prediction
        prediction = model.predict(input_data)
        
        # The model returns an array like [150000.50], we grab the first element
        predicted_price = float(prediction[0])
        
        # Return the result as JSON
        return {
            "predicted_price": round(predicted_price, 2),
            "currency": "USD"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 6. A simple health check endpoint
@app.get("/")
def read_root():
    return {"status": "API is running!", "model_loaded": model is not None}
