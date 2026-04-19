import os
import sys
import json

def setup_kaggle_credentials():
    """
    Ensure Kaggle credentials are configured.

    The Kaggle CLI requires KAGGLE_USERNAME and KAGGLE_KEY to authenticate.
    These can be provided via:
      1. Environment variables: KAGGLE_USERNAME and KAGGLE_KEY
      2. A JSON file at ~/.kaggle/kaggle.json: {"username": "...", "key": "..."}

    To get your credentials:
      - Go to https://www.kaggle.com/settings
      - Under "API", click "Create New Token" to download kaggle.json
      - The file contains your username and key
    """
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Check environment variables
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        print("Using Kaggle credentials from environment variables.")
        return True

    # Check kaggle.json file
    if os.path.exists(kaggle_json):
        try:
            with open(kaggle_json, 'r') as f:
                creds = json.load(f)
            if 'username' in creds and 'key' in creds:
                print(f"Using Kaggle credentials from {kaggle_json}")
                return True
        except (json.JSONDecodeError, IOError):
            pass

    # No valid credentials found
    print("=" * 65)
    print("  Kaggle API credentials not found or incomplete!")
    print("=" * 65)
    print()
    print("The Kaggle CLI needs your USERNAME and API KEY (not a KGAT token).")
    print()
    print("How to get them:")
    print("  1. Go to https://www.kaggle.com/settings")
    print('  2. Under "API", click "Create New Token"')
    print("  3. This downloads a kaggle.json with your username + key.")
    print()
    print("Then choose one setup method:")
    print()
    print("  METHOD A — Environment variables (temporary):")
    print("    export KAGGLE_USERNAME=your_kaggle_username")
    print("    export KAGGLE_KEY=your_api_key_from_kaggle_json")
    print()
    print("  METHOD B — Config file (permanent):")
    print(f"    mkdir -p {kaggle_dir}")
    print(f"    mv ~/Downloads/kaggle.json {kaggle_json}")
    print(f"    chmod 600 {kaggle_json}")
    print()
    print("NOTE: The KGAT_xxx token from 'API Token' is NOT the same as the")
    print("      kaggle.json credentials. You need the 'Create New Token'")
    print("      button which downloads a JSON file with username + key.")
    print()
    return False

def main():
    if not setup_kaggle_credentials():
        sys.exit(1)

    print("Downloading HAI Security Dataset for ICS Anomaly Detection...")
    ret = os.system('kaggle datasets download -d icsdataset/hai-security-dataset -p data --unzip')
    if ret != 0:
        print("\nError: Download failed. Please verify your credentials and try again.")
        sys.exit(1)

    print("Dataset downloaded and extracted to data/ directory successfully.")

if __name__ == '__main__':
    main()
