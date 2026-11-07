# Instamart Customer Intelligence Capstone

## Overview
This repository contains the data pipeline, analytics, customer segmentation, support ticket classification, and automated refund agent for the Instamart Customer Intelligence capstone project. It processes raw transactional and customer data, stores it securely in a local SQLite database, and executes reproducible machine learning and LLM-driven workflows.

## Reproduce
To reproduce the data generation, cleaning, database creation, and subsequent analytics from a clean clone, run the following commands in order:

```bash
# 1. Create an isolated virtual environment
python -m venv venv

# 2. Activate the virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows (Command Prompt):
# venv\Scripts\activate.bat
# On Windows (PowerShell):
# venv\Scripts\Activate.ps1

# 3. Install required open-source dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate the raw synthetic datasets and policy documents
python generate_data.py
