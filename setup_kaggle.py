#!/usr/bin/env python3
"""Setup Kaggle for downloading deepfake datasets."""

import os

kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")

if not os.path.exists(kaggle_json):
    print("Kaggle API credentials not found.")
    print("\nTo setup Kaggle:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
    print("\nOr run with your credentials:")
    print("export KAGGLE_USERNAME=your_username")
    print("export KAGGLE_KEY=your_key")
    exit(1)
else:
    print("✓ Kaggle credentials found")
    import subprocess
    result = subprocess.run(["kaggle", "competitions", "list", "-s", "deepfake"], 
                          capture_output=True, text=True)
    print(result.stdout)
