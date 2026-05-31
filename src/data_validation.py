import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from pandera import DataFrameModel

# Define a strict schema for our dataset
# This acts as a contract. If data doesn't match this, the code will crash (which is good!)
class HousePriceSchema(DataFrameModel):
    LotArea: Series[int] = pa.Field(ge=100) # Area must be >= 100 sq ft
    BedroomAbvGr: Series[int] = pa.Field(ge=0, le=20) # Bedrooms between 0 and 20
    FullBath: Series[int] = pa.Field(ge=0, le=10) # Bathrooms between 0 and 10
    HouseStyle: Series[str] = pa.Field(nullable=False) # Cannot be empty
    Age: Series[int] = pa.Field(ge=0, le=200) # Age must be realistic (0 to 200 years)
    
    # SalePrice is optional during validation because when a user inputs data on the website,
    # we won't have a SalePrice yet (we are predicting it!)
    SalePrice: Series[float] = pa.Field(ge=1000, nullable=True, coerce=True)

    class Config:
        strict = False # Allow other columns to exist, but strictly validate these

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the dataframe against the HousePriceSchema.
    Returns the validated dataframe. Throws an error if validation fails.
    """
    try:
        # We use schema.validate to check the DataFrame
        validated_df = HousePriceSchema.validate(df)
        print("Data Validation Passed! All rules are satisfied.")
        return validated_df
    except pa.errors.SchemaError as exc:
        print("Data Validation Failed!")
        print(f"Error Details: {exc}")
        raise

if __name__ == "__main__":
    # A quick test to show how it works
    import pandas as pd
    
    # Let's create some dummy data that perfectly matches the rules
    good_data = pd.DataFrame({
        'LotArea': [1000, 2500],
        'BedroomAbvGr': [3, 4],
        'FullBath': [2, 3],
        'HouseStyle': ['1Story', '2Story'],
        'Age': [10, 5],
        'SalePrice': [200000.0, 300000.0]
    })
    
    print("Testing Good Data...")
    validate_data(good_data)
    
    # Let's create data with a NEGATIVE area (which is impossible)
    bad_data = pd.DataFrame({
        'LotArea': [-500], # This breaks the ge=100 rule
        'BedroomAbvGr': [3],
        'FullBath': [2],
        'HouseStyle': ['1Story'],
        'Age': [10],
        'SalePrice': [200000.0]
    })
    
    print("\nTesting Bad Data...")
    try:
        validate_data(bad_data)
    except Exception:
        pass # We expect it to fail
