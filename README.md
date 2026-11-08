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

Running validation script against SQLite database...
✓ Checkpoint 1 Passed: Total orders = 12,309
✓ Checkpoint 2 Passed: Pending orders = 0
✓ Checkpoint 3 Passed: Completed orders = 11,110
✓ Checkpoint 4 Passed: Cancelled orders = 1,199
✓ Checkpoint 5 Passed: Missing order amounts = 0
✓ Checkpoint 6 Passed: Completed revenue sum matches INR 2,998,668.80
✓ Checkpoint 7 Passed: Missing customer ages = 0
✓ Checkpoint 8 Passed: Customer ages bounded between 18 and 60

ALL CHECKPOINTS PASSED

Connecting to SQLite database...

======================================================================
EXECUTING Q1: Customers with completed orders > 500 and total spend > 8000
======================================================================
Result Shape: (30, 2)
customer_id  total_spend
      C0082     12317.70
      C0130      9170.40
      C0147      9141.50
      C0167      8718.55
      C0267     12150.25
      C0291     12150.70
      C0308      9872.40
      C0332     10779.50
      C0407      9451.75
      C0424      8294.20
... and 20 more rows.

======================================================================
EXECUTING Q2: Revenue by city and product category
======================================================================
Result Shape: (30, 3)
     city      category  total_revenue
Bengaluru         Dairy       87075.05
Bengaluru   Electronics      233909.45
Bengaluru       Grocery      110915.65
Bengaluru Personal Care       20193.85
Bengaluru        Snacks       71549.40
  Chennai         Dairy       78301.80
  Chennai   Electronics      195508.80
  Chennai       Grocery       93817.45
  Chennai Personal Care       27675.40
  Chennai        Snacks       68325.70
... and 20 more rows.

======================================================================
EXECUTING Q3: Products priced above average catalogue price
======================================================================
Result Shape: (11, 3)
product_id        product_name  price_inr
      P014          Ghee 500ml        340
      P030   Body Lotion 400ml        350
      P032       Sunscreen 50g        399
      P033    Wireless Earbuds       1499
      P034 Power Bank 10000mAh       1299
      P035      USB-C Cable 1m        299
      P036   Phone Charger 20W        799
      P037          Smart Bulb        599
      P038   Bluetooth Speaker       1999
      P039         Phone Stand        349
... and 1 more rows.

======================================================================
EXECUTING Q4: Customers with no completed orders
======================================================================
Result Shape: (25, 1)
customer_id
      C0029
      C0037
      C0093
      C0115
      C0120
      C0151
      C0354
      C0363
      C0412
      C0461
... and 15 more rows.

======================================================================
EXECUTING Q5: Customers whose completed spend is above the customer average
======================================================================
Result Shape: (550, 2)
customer_id  total_spend
      C0002      4644.90
      C0003      3644.85
      C0007      2600.00
      C0008      3096.50
      C0009      8424.80
      C0011      3583.85
      C0012      3395.30
      C0014      8006.90
      C0016      3546.00
      C0017      7521.70
... and 540 more rows.

======================================================================
EXECUTING Q6: Ranking window functions by city and total spend
======================================================================
Result Shape: (1175, 6)
customer_id      city  total_spend  rnk  dense_rnk  row_num
      C0082 Bengaluru     12856.20    1          1        1
      C0525 Bengaluru     12719.70    2          2        2
      C0699 Bengaluru     12471.80    3          3        3
      C0543 Bengaluru     11195.10    4          4        4
      C1147 Bengaluru     10156.90    5          5        5
      C0407 Bengaluru      9945.85    6          6        6
      C0942 Bengaluru      8643.60    7          7        7
      C1126 Bengaluru      8595.70    8          8        8
      C0916 Bengaluru      8444.85    9          9        9
      C0098 Bengaluru      8151.20   10         10       10
... and 1165 more rows.

======================================================================
EXECUTING Q7: Running total of daily completed revenue
======================================================================
Result Shape: (183, 3)
       day  daily_rev  cumulative_revenue
2025-04-01   16876.50            16876.50
2025-04-02   19854.55            36731.05
2025-04-03   28048.35            64779.40
2025-04-04   14249.60            79029.00
2025-04-05   16031.50            95060.50
2025-04-06   26323.80           121384.30
2025-04-07   15606.90           136991.20
2025-04-08   17055.45           154046.65
2025-04-09   20403.35           174450.00
2025-04-10   15359.90           189809.90
... and 173 more rows.

All SQL queries executed successfully!
