import pandas as pd
import numpy as np
import os
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import our custom modules
from data_validation import validate_data
from preprocessing import get_preprocessor

def train_model():
    """
    Trains a Linear Regression model on the housing dataset,
    evaluates it, tracks it with MLflow, and saves the model.
    """
    print("--- 1. Loading and Validating Data ---")
    df = pd.read_csv('../data/raw/train.csv')
    
    # Feature Engineering (Age calculation as done in EDA)
    df['Age'] = 2010 - df['YearBuilt']
    
    # Select only the features we care about
    features = ['LotArea', 'BedroomAbvGr', 'FullBath', 'HouseStyle', 'Age', 'SalePrice']
    df = df[features].copy()
    
    # Validate the data using our Pandera schema
    df = validate_data(df)
    
    # Separate Features (X) and Target (y)
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    print("\n--- 2. Splitting Data (80% Train, 20% Test) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training on {len(X_train)} houses. Testing on {len(X_test)} houses.")
    
    print("\n--- 3. Building the ML Pipeline ---")
    # We combine our preprocessor and the actual algorithm into one seamless pipeline
    preprocessor = get_preprocessor()
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    print("\n--- 4. Training the Model (with MLflow Tracking) ---")
    # Set up MLflow
    mlflow.set_experiment("House_Price_Prediction_LinearRegression")
    
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the Test set
        y_pred = model.predict(X_test)
        
        # Calculate Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        print(f"R-squared (R2) Score: {r2:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Save a standalone copy for our Web App
        os.makedirs('../models', exist_ok=True)
        model_path = '../models/linear_regression_model.joblib'
        joblib.dump(model, model_path)
        print(f"\nModel saved successfully to {model_path}")
        
if __name__ == "__main__":
    train_model()
