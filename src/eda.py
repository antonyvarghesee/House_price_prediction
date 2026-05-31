import pandas as pd
import os

def perform_eda(filepath):
    """
    Loads the Kaggle dataset, selects the features we need,
    and performs basic Exploratory Data Analysis.
    """
    print(f"Loading data from {filepath}...")
    
    # 1. Load the dataset
    df = pd.read_csv(filepath)
    
    print(f"\nOriginal Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Select our specific features and the target variable (SalePrice)
    # Mapping:
    # Area -> LotArea
    # Bedrooms -> BedroomAbvGr
    # Bathrooms -> FullBath
    # Floors -> HouseStyle (We will look at this, though it's categorical)
    # Age -> YearBuilt (We will calculate Age = CurrentYear - YearBuilt)
    # Price -> SalePrice
    
    features = ['LotArea', 'BedroomAbvGr', 'FullBath', 'HouseStyle', 'YearBuilt', 'SalePrice']
    
    # Check if all these columns exist in the downloaded dataset
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        return
        
    df_selected = df[features].copy()
    
    print(f"\nSelected Features Shape: {df_selected.shape[0]} rows, {df_selected.shape[1]} columns")
    
    # 3. Calculate 'Age' from 'YearBuilt'
    # For a real project, we usually use the year the house was sold.
    # We will use 2010 (the approximate year the Ames dataset ends) to calculate age.
    df_selected['Age'] = 2010 - df_selected['YearBuilt']
    
    # We can drop YearBuilt now since we have Age
    df_selected = df_selected.drop('YearBuilt', axis=1)
    
    print("\n--- First 5 Rows of our Data ---")
    print(df_selected.head())
    
    print("\n--- Missing Values Check ---")
    # This will show us if any of our chosen columns have missing data (NaN)
    print(df_selected.isnull().sum())
    
    print("\n--- Basic Statistics ---")
    # This shows mean, min, max, etc., for numerical columns
    print(df_selected.describe().round(2))

if __name__ == "__main__":
    # The Kaggle download usually provides 'train.csv' and 'test.csv'
    # 'train.csv' contains the SalePrice, so we use that for training/EDA
    TRAIN_DATA_PATH = "../data/raw/train.csv"
    
    if os.path.exists(TRAIN_DATA_PATH):
        perform_eda(TRAIN_DATA_PATH)
    else:
        print(f"Data file not found at {TRAIN_DATA_PATH}. Make sure data_ingestion.py ran successfully.")
