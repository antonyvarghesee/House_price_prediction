import pandas as pd
import numpy as np
import os

def generate_house_data(num_samples=1000, random_seed=42):
    """
    Generates a synthetic dataset for house prices.
    Features: Area, Bedrooms, Bathrooms, Floors, Age
    """
    np.random.seed(random_seed)
    
    # 1. Generate independent features (X)
    area = np.random.randint(800, 5000, num_samples)
    bedrooms = np.random.randint(1, 6, num_samples)
    bathrooms = np.random.randint(1, 4, num_samples)
    floors = np.random.randint(1, 3, num_samples)
    age = np.random.randint(0, 50, num_samples)
    
    # 2. Define the "true" relationship (Linear Regression equation)
    # Price = Base + (w1 * Area) + (w2 * Bedrooms) + (w3 * Bathrooms) + (w4 * Floors) - (w5 * Age) + Noise
    base_price = 50000
    price_per_sqft = 150
    price_per_bedroom = 25000
    price_per_bathroom = 15000
    price_per_floor = 20000
    depreciation_per_year = 1000
    
    # Calculate base target variable (y) without noise
    true_price = (
        base_price + 
        (area * price_per_sqft) + 
        (bedrooms * price_per_bedroom) + 
        (bathrooms * price_per_bathroom) + 
        (floors * price_per_floor) - 
        (age * depreciation_per_year)
    )
    
    # Add random noise (real world data is never perfectly linear)
    noise = np.random.normal(0, 25000, num_samples)
    price = true_price + noise
    
    # Ensure no negative prices (just in case)
    price = np.maximum(price, 50000)
    
    # 3. Create a DataFrame
    df = pd.DataFrame({
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Floors': floors,
        'Age': age,
        'Price': np.round(price, 2)
    })
    
    # 4. Introduce some "messiness" so we can practice Data Cleaning later
    # Add some missing values (NaN) to Area
    missing_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[missing_indices, 'Area'] = np.nan
    
    return df

if __name__ == "__main__":
    print("Generating synthetic house price data...")
    df = generate_house_data()
    
    # Create data/raw directory if it doesn't exist
    os.makedirs('../data/raw', exist_ok=True)
    
    # Save the dataset
    filepath = '../data/raw/house_prices.csv'
    df.to_csv(filepath, index=False)
    
    print(f"Dataset successfully generated and saved to {filepath}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
