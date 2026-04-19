import os
import sys

def main():
    print("Downloading HAI Security Dataset for ICS Anomaly Detection...")
    # Requires KAGGLE_API_TOKEN environment variable to be set
    ret = os.system('kaggle datasets download -d icsdataset/hai-security-dataset -p data --unzip')
    if ret != 0:
        print("Error: Failed to download dataset. Please check your Kaggle API token.")
        sys.exit(1)
    
    print("Dataset downloaded and extracted to data/ directory successfully.")

if __name__ == '__main__':
    main()
