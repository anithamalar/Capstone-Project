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

## Part A
### Raw Data Audit Summary
* **orders.csv rows**: 13,084
* **Exact duplicate rows**: 119
* **Pending status rows**: 656
* **Distinct order_id values**: 12,309
* **Non-numeric order_amount entries**: 120
* **Negative delivery_minutes**: 106
* **Blank delivery_minutes**: 2,541
* **Blank customer ages**: 61
* **Ages outside 10–100**: 6
* **Distinct raw city spellings**: 24
* **Distinct non-blank raw gender spellings**: 10
* **Blank genders**: 30


Customer cleaning result:

Missing ages count: 0 (Expected: 0)
Minimum age: 18.0, Maximum age: 60.0 (Expected: between 18 and 60)

City distribution after cleaning:
city
Pune         227
Delhi        210
Bengaluru    197
Chennai      191
Hyderabad    191
Mumbai       184
Name: count, dtype: int64

Gender distribution after cleaning:
gender
female     610
male       560
unknown     30
Name: count, dtype: int64
=================================================================

Order cleaning result :

Total rows: 12309 (Expected: 12309)
Status counts:
status
completed    11110
cancelled     1199
Name: count, dtype: int64
Pending rows count: 0 (Expected: 0)
Missing order amounts: 0 (Expected: 0)
Completed orders amount sum: INR 2,998,668.80 (Expected: INR 2,998,668.80)
=================================================================
