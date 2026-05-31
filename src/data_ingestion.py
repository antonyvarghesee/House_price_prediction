import os
import zipfile
import logging

# Set up simple logging so we can see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def download_kaggle_dataset(competition_name, output_dir):
    """
    Downloads and extracts a Kaggle dataset using the Kaggle API.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # We import kaggle here so it only authenticates when the function is run.
    # The Kaggle library will automatically look for ~/.kaggle/kaggle.json
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logging.info("Successfully authenticated with Kaggle API.")
    except Exception as e:
        logging.error(f"Failed to authenticate with Kaggle. Ensure kaggle.json is in ~/.kaggle/. Error: {e}")
        return
    
    logging.info(f"Downloading dataset for competition: {competition_name}...")
    
    # Download the files to the specified directory
    try:
        api.competition_download_files(competition_name, path=output_dir, force=True)
        logging.info("Download completed successfully.")
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        return

    # The file is downloaded as a zip archive, we need to extract it
    zip_path = os.path.join(output_dir, f"{competition_name}.zip")
    if os.path.exists(zip_path):
        logging.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        logging.info("Extraction complete.")
        
        # We can remove the zip file to save space
        os.remove(zip_path)
        logging.info("Cleaned up zip file.")
    else:
        logging.warning("No zip file found to extract. Perhaps files were downloaded uncompressed.")

if __name__ == "__main__":
    COMPETITION = "house-prices-advanced-regression-techniques"
    RAW_DATA_DIR = "../data/raw"
    
    download_kaggle_dataset(COMPETITION, RAW_DATA_DIR)
