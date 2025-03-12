import os
import gdown
import zipfile

# Google Drive file ID 
MODEL_ZIP_FILE_ID = "1TjtdVLCQ1So2TC6ylo6Rttyp2l_yXpvX"  

# Define storage paths
MODEL_DIR = "model"  # Directory where the model will be extracted
ZIP_PATH = "model.zip"  # Temporary zip file for downloading

def download_and_extract(file_id, output_dir, zip_path):
    """Download a zip file from Google Drive and extract it."""
    if not os.path.exists(output_dir):  # Check if model folder already exists
        print(f"Downloading {zip_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)

        print("Extracting model files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")  # Extract to current directory

        # Clean up the zip file after extraction
        os.remove(zip_path)
        print(f"✅ Model extracted successfully to '{output_dir}'.")
    else:
        print(f"✅ Model folder '{output_dir}' already exists. Skipping download.")

if __name__ == "__main__":
    download_and_extract(MODEL_ZIP_FILE_ID, MODEL_DIR, ZIP_PATH)
    print("\n🚀 Model is ready for inference.")
