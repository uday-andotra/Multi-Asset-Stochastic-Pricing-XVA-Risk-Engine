#!/bin/bash
echo "======================================================"
echo " Starting Multi-Asset XVA Streamlit Dashboard"
echo "======================================================"

set -e

echo "[1/2] Installing required Python packages..."
pip install -r requirements.txt

echo "[2/2] Launching Streamlit Server..."
streamlit run app.py