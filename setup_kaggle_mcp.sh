#!/bin/bash
# Kaggle MCP Server Setup Script

echo "Setting up Kaggle MCP Server..."

# Check if Kaggle API credentials exist
KAGGLE_CONFIG="$HOME/.kaggle/kaggle.json"

if [ -f "$KAGGLE_CONFIG" ]; then
    echo "✓ Kaggle credentials found at $KAGGLE_CONFIG"
else
    echo "✗ Kaggle credentials not found!"
    echo ""
    echo "To set up Kaggle API:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Save kaggle.json to ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Install kaggle CLI if not present
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle CLI..."
    pip install kaggle
fi

# Test Kaggle CLI
echo "Testing Kaggle CLI..."
kaggle competitions list --sort-by deadline --limit 5

echo ""
echo "Kaggle MCP Server setup complete!"
echo ""
echo "To download deepfake datasets:"
echo "  kaggle competitions download -c deepfake-detection-challenge"
echo ""
echo "Available competitions:"
kaggle competitions list -s deepfake
