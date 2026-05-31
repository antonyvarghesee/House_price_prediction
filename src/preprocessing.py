from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer that scales numerical
    features and one-hot encodes categorical features.
    """
    # 1. Identify which columns need which treatment
    numerical_features = ['LotArea', 'BedroomAbvGr', 'FullBath', 'Age']
    categorical_features = ['HouseStyle']
    
    # 2. Define what happens to numerical columns
    # StandardScaler transforms numbers so they have a mean of 0 and standard deviation of 1.
    # Linear Regression performs much better when all numbers are on the same scale!
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 3. Define what happens to text/categorical columns
    # OneHotEncoder creates a new binary (0 or 1) column for each category.
    # e.g., 'HouseStyle_1Story', 'HouseStyle_2Story'
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 4. Combine them into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return preprocessor

if __name__ == "__main__":
    # Quick test to see the preprocessor in action
    import pandas as pd
    
    print("Testing Preprocessor...")
    dummy_data = pd.DataFrame({
        'LotArea': [1000, 2000],
        'BedroomAbvGr': [3, 4],
        'FullBath': [2, 3],
        'HouseStyle': ['1Story', '2Story'],
        'Age': [10, 5]
    })
    
    print("Original Data:")
    print(dummy_data)
    
    preprocessor = get_preprocessor()
    processed_data = preprocessor.fit_transform(dummy_data)
    
    print("\nProcessed Data (Ready for Machine Learning):")
    # The output is a numpy array (matrix of numbers)
    print(processed_data)
