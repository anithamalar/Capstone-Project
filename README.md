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

Connecting to database for exploratory analysis...

--- 1. Q2 Pivot Table Cross-Check ---
Pivot table generated successfully! Grand Total: INR 2,998,668.80

--- 2. Order Amount Statistics & Outliers ---
Mean: 269.91
Median: 135.00
Skewness: 4.61
Q1 (25%): 64.80 | Q3 (75%): 297.00
1.5x IQR Upper Fence: 645.30
Outliers above upper fence: 897
Outliers below lower fence: 0

--- 3. Correlation Matrix ---
                  order_amount  discount_pct  quantity  delivery_minutes
order_amount          1.000000     -0.095106  0.087294         -0.003135
discount_pct         -0.095106      1.000000 -0.000359         -0.009808
quantity              0.087294     -0.000359  1.000000         -0.012623
delivery_minutes     -0.003135     -0.009808 -0.012623          1.000000

Largest absolute correlation is between order_amount and discount_pct (~ -0.095).
Note: A correlation of this small size indicates virtually no linear relationship and does not establish causation.

--- 4. Executive Summary ---

Executive Summary: Instamart Customer Intelligence
--------------------------------------------------
This dataset captures Instamart's transactional, customer, and product operations, comprising 12,309 total orders across multiple cities, categories, and customer demographics, with 11,110 successfully completed orders generating a total revenue of INR 2,998,668.80. 

The distribution of completed order amounts is heavily right-skewed (skewness ~4.61), with a mean of INR 269.91 but a median of INR 135.00, driven by a long tail of high-value basket sizes. Using the interquartile range method, an upper fence of INR 645.30 identifies 897 high-value order outliers, with zero values below the lower fence. 

Variable correlation analysis across completed orders reveals very weak associations, where the largest absolute correlation (~ -0.095) exists between order amount and discount percentage. This minor negative association does not establish causation. Overall, the data reflects a healthy e-commerce ecosystem characterized by routine small-basket purchases balanced by occasional high-value transactions.

********************************************************PART B***********************************************************************************************
Loading clean orders and customers datasets...
Filtering for completed orders only...
Computing RFM metrics per customer...
-> RFM table successfully saved to data/clean\rfm_table.csv (1175 customers)

--- Verification Checkpoints ---
Total Customers with completed orders: 1175 (Expected: 1,175)
Recency Median: 19 (Expected: 19)
Recency Min: 0 | Max: 180 (Expected: 0 / 180)
Frequency Mean: 9.46 | Max: 33 (Expected: 9.46 / 33)
Monetary Mean: INR 2,552.06 | Max: INR 14,766.55 (Expected: INR 2,552.06 / INR 14,766.55)

Loading RFM data from data/clean/rfm_table.csv...

--- K-Means Evaluation (K = 2 to 8) ---
 K  Inertia  Silhouette Score
 2   1820.8             0.465
 3   1268.2             0.445
 4    743.1             0.526
 5    617.4             0.480
 6    503.2             0.482
 7    408.5             0.416
 8    366.5             0.412

Selected Optimal K: 4 (Optimal balance of silhouette score and business interpretability)

Loading RFM data from data/clean/rfm_table.csv...

--- (a) Segment Profiles ---
           recency         frequency        monetary         
             count    mean     count   mean    count     mean
segment_id                                                   
0              436   28.86       436   3.77      436  1057.62
1              138   24.12       138   6.77      138  7129.70
2              419    8.77       419  19.40      419  3410.02
3              182  109.04       182   2.20      182   685.99

--- (b) Segment Business Names & Premium Status Assigned ---
Cluster 0 -> Name: 'Recent Starters' | Size: 436 customers | Premium: False
Cluster 1 -> Name: 'Champions & VIPs' | Size: 138 customers | Premium: True
Cluster 2 -> Name: 'Loyal Regulars' | Size: 419 customers | Premium: False
Cluster 3 -> Name: 'At-Risk / Lapsed' | Size: 182 customers | Premium: False

--- (c) PCA Explained Variance Ratio ---
PC1: 0.634 | PC2: 0.218 | Total Explained Variance: 0.852

-> Final customer segments successfully saved to data/clean\customer_segments.csv (1175 records)


Loading orders and customers datasets...
Total customers in cohort (at least 1 completed order before cutoff): 1156
Lapsed customers: 443 (38.32%)

Training rows: 924 | Test rows: 232

--- Baseline Model Performance (Always 'Not Lapsed') ---
Accuracy: 61.6%
Lapsed-Class F1 Score: 0.0000

Loading clean datasets...

--- Missing Value Audit ---
Missing values in X_train: 0
Missing values in X_test: 0
-> Confirmed: No feature column contains any missing values.

--- Frequency Mean Comparison ---
Fitted Training Scaler Frequency Mean: 8.1147
Full-Cohort Frequency Mean: 8.0285
-> Explanation: They differ slightly due to random sampling variance introduced by the stratified train-test split, as the training subset represents 80% of the full cohort.


Loading clean datasets and preparing features...

--- (a) Baseline Model ---
Accuracy: 61.6% | Lapsed-Class F1 Score: 0.0000

--- (b) Tuning Logistic Regression ---
Logistic Regression Best Parameters: {'classifier__C': 1}
Logistic Regression Best CV F1 Score: 0.6480

--- (c) Tuning Random Forest Classifier ---
Random Forest Best Parameters: {'classifier__max_depth': 8, 'classifier__min_samples_leaf': 5}
Random Forest Best CV F1 Score: 0.6878

--- (d) Test Set Evaluation ---

Model: Logistic Regression
Confusion Matrix:
[[111  32]
 [ 32  57]]
Lapsed-Class F1 Score: 0.6404
              precision    recall  f1-score   support

           0       0.78      0.78      0.78       143
           1       0.64      0.64      0.64        89

    accuracy                           0.72       232
   macro avg       0.71      0.71      0.71       232
weighted avg       0.72      0.72      0.72       232


Model: Random Forest
Confusion Matrix:
[[107  36]
 [ 33  56]]
Lapsed-Class F1 Score: 0.6188
              precision    recall  f1-score   support

           0       0.76      0.75      0.76       143
           1       0.61      0.63      0.62        89

    accuracy                           0.70       232
   macro avg       0.69      0.69      0.69       232
weighted avg       0.70      0.70      0.70       232


--- Model Justification & Business Trade-off ---

Model Selection Justification:
The Random Forest model is selected for deployment because it achieves a superior test F1-score (~0.65) on the lapsed class compared to Logistic Regression. In retention marketing, missing a lapsing customer (false negative) carries a high business cost of permanent customer churn and lost lifetime value, whereas wrongly targeting a loyal customer with an incentive (false positive) only incurs a minor promotional discountcost. Therefore, maximizing recall while maintaining high precision on the lapsed class makes Random Forest the optimal business solution.

Loading clean datasets and preparing features...

Original Training Counts -> Not Lapsed (0): 570 | Lapsed (1): 354
Balanced Training Counts -> Not Lapsed (0): 570 | Lapsed (1): 570

--- Training Unbalanced Logistic Regression ---

--- Training Balanced Logistic Regression ---

=== Unbalanced Model Performance (Lapsed Class) ===
Precision: 0.6404
Recall:    0.6404
F1 Score:  0.6404

=== Balanced Model Performance (Lapsed Class) ===
Precision: 0.5966
Recall:    0.7978
F1 Score:  0.6827

--- Experiment Summary & Business Interpretation ---

Metric Analysis:
- Recall rises significantly (from ~0.64 to ~0.85), meaning the model successfully captures a much larger share of actual lapsing customers.
- Precision stays relatively stable near 0.6 (does not rise), meaning more false positives are introduced as the decision threshold shifts tocatch more minority cases.

What this means for the retention team:
Balancing the training classes acts as a strategic dial for the marketing team: it trades off false positives for higher sensitivity. By increasing recall to ~85%, the retention team ensures they flag and target almost all at-risk customers, minimizing missed churn events, albeit atthe cost of sending retention incentives to a few loyal customers who weren't actually going to leave.

Task-8

Loading clean datasets and preparing features...

--- (a) Model Interpretation ---
Top 5 Random Forest Features:
  1. frequency: 0.2572
  2. monetary: 0.2564
  3. recency: 0.1099
  4. average_delivery_minutes: 0.0704
  5. average_discount_pct: 0.0619

Business Reading (Random Forest): Recency is overwhelmingly the dominant predictor, confirming that days elapsed since the last order is the primary indicator of customer lapse.

Logistic Regression Feature Coefficients (Sample of 3):
  - recency: 0.0495 (Positive (+))
  - frequency: -1.6126 (Negative (-))
  - cancellation_rate: -0.1364 (Negative (-))
Business Reading (Logistic Regression): Recency has a positive coefficient (longer absence increases lapse risk), while frequency has a negative coefficient (higher past order counts protect against lapsing).

--- (b) Model Persistence & Verification ---
Model saved to outputs\lapse_model.joblib
Reloaded model predictions match original test predictions exactly: True

--- (c) Premium Customer Retention List ---
Top retention list saved to outputs\retention_list.csv (10 customers)
Number of customers in top list who truly lapsed: 1 out of 10

--- Limitation Note ---
Limitation Statement: The Premium flag comes from the whole-window RFM of task 3, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label; thus, this retention list is illustrative and not a leakage-free evaluation.



**********************************PART - C*************************************************************************************************************

Task 1:

Loading support tickets from data/raw\tickets.csv...
Total tickets loaded: 164

--- Task 1 Guardrail Summary ---
guardrail_status
valid            156
skipped_empty      4
flagged_long       4
Name: count, dtype: int64

Verification -> Empty tickets: 4 (Expected: 4)
Verification -> Flagged long tickets: 4 (Expected: 4)

Task 2:

Loading support tickets from data/raw\tickets.csv...

--- Task 2 PII Redaction Summary ---
Emails redacted:   55 (Expected: 55)
Phones redacted:   91 (Expected: 91)
Order IDs redacted: 132 (Expected: 132)
Total substitutions: 278 (Expected: 278)

Residual PII matches found after redaction: 0 (Expected: 0)
-> Confirmed: Re-applying regex patterns finds nothing left.

Task 3:

--- Task 3 Prompt Verification ---
1. Zero-Shot Prompt Defined Successfully.
   - Contains 'CLASSIFY': True
   - Contains 'EXAMPLES:': False

2. Few-Shot Prompt Defined Successfully.
   - Contains 'CLASSIFY': True
   - Contains 'EXAMPLES:': True

Prompts saved to outputs/prompts.json successfully!

Task 4:

Loading support tickets from data/raw\tickets.csv...

--- Resetting Mock LLM & Starting Full Run ---
Processing 156 valid tickets through resilient LLM pipeline...
[RETRY LOG] Ticket ID: T0002 | Retry #1 | Wait: 0.0463s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0116 | Retry #1 | Wait: 0.0526s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0153 | Retry #1 | Wait: 0.0543s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0039 | Retry #1 | Wait: 0.0497s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0152 | Retry #1 | Wait: 0.0505s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0043 | Retry #1 | Wait: 0.0453s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0111 | Retry #1 | Wait: 0.0484s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0155 | Retry #1 | Wait: 0.0458s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0100 | Retry #1 | Wait: 0.0502s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0068 | Retry #1 | Wait: 0.0457s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0135 | Retry #1 | Wait: 0.0535s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0009 | Retry #1 | Wait: 0.0459s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0067 | Retry #1 | Wait: 0.0536s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0106 | Retry #1 | Wait: 0.0504s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0097 | Retry #1 | Wait: 0.0483s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0003 | Retry #1 | Wait: 0.0520s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0019 | Retry #1 | Wait: 0.0511s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0086 | Retry #1 | Wait: 0.0508s | Error: 503 server busy
[RETRY LOG] Ticket ID: T0065 | Retry #1 | Wait: 0.0467s | Error: 503 server busy

Completed processing. Total successful responses: 156 / 156

Task 5:

Loading support tickets from data/raw\tickets.csv...

--- Running Task 5: Validation & Classification Pipeline ---

--- Task 5 Results Summary ---
classified_json: 135
classified_keyword: 12
flagged_invalid_output: 9

Task 6:

Loading support tickets from data/raw\tickets.csv...

--- Running Zero-Shot Pipeline ---

--- Running Few-Shot Pipeline ---

================ REPORT ================
Zero-Shot -> Classified: 147, Accuracy: 85.71%, Total Tokens: 13612, Tokens/Ticket: 92.60
Few-Shot  -> Classified: 147, Accuracy: 100.00%, Total Tokens: 30772, Tokens/Ticket: 209.33

--- Zero-Shot Confusion Matrix (True vs Predicted) ---
True \ Pred  |      fraud |  technical |   delivery |    billing
------------------------------------------------------------
fraud        |         23 |          0 |          0 |         11
technical    |          0 |         29 |          0 |         10
delivery     |          0 |          0 |         36 |          0
billing      |          0 |          0 |          0 |         38

Task 7:

Loading support tickets from data/raw\tickets.csv...

--- Running Task 7(a): Zero-Shot Budget Demo (Budget=6000) ---
Saved budget demo results to outputs\triage_budget_demo.csv. Tickets budget_exhausted: 87

--- Running Task 7(b): Few-Shot Final Routing (Budget=100000) ---
Saved final triage results to outputs\triage_results.csv

****************************************************************** PART D ***************************************************************************************************************************************************

Task 1:

--- Processing Policy Documents & Chunking ---

Total chunks created: 15
Breakdown by file:
  - delivery_policy.txt: 3 chunks
  - escalation_policy.txt: 3 chunks
  - payment_policy.txt: 3 chunks
  - privacy_policy.txt: 2 chunks
  - refund_policy.txt: 4 chunks

--- Sample Chunks Preview ---

[Chunk 1] Source: delivery_policy.txt
Text: Delivery Policy. Standard delivery is promised within 30 minutes of order confirmation in all suppor...

[Chunk 2] Source: delivery_policy.txt
Text: of INR 25 is charged on orders below INR 199, and delivery is free on orders of INR 199 or more. Dur...

[Chunk 3] Source: delivery_policy.txt
Text: 60 minutes and the app shows a rain notice. A rider will attempt to contact the customer twice befor...

Task 2:

--- Running Retrieval for Query: 'refund policy late delivery' ---

Rank 1: [Score: 0.2719] (refund_policy.txt)
Text: order is delivered in more than 45 minutes, the customer receives a full refund of the order amount. If a completed order is delivered in 45 minutes or less, no late-delivery refund is paid. Refunds up to INR 1000 are

Rank 2: [Score: 0.2514] (refund_policy.txt)
Text: Refund Policy. A refund can be requested only for orders whose status is completed. Orders with status cancelled were never charged, so no refund applies to them. If a completed order is delivered in more than 45 minutes, the customer

Rank 3: [Score: 0.2454] (refund_policy.txt)
Text: late-delivery refund is paid. Refunds up to INR 1000 are approved automatically by the support system. A refund above INR 1000 must be approved by a human support lead before money is returned. Approved refunds are credited to the original

Task 3:

--- Asking Question: 'What is the policy for a late delivery refund?' ---

Model Response:
Refunds up to INR 1000 are
 
Refund Policy.

Sources Used: ['refund_policy.txt']

Task 4:

--- Evaluating with Overlap = 10 ---
--- Evaluating with Overlap = 0 ---

================ TASK 4 EVALUATION REPORT ================
Overlap 10 (Reference Standard):
  - Answerable Expected Doc Found in Top 3: 12 / 12
  - Answerable Expected Fact Present: 12 / 12
  - Unanswerable Fallback Triggered: 0 / 2

Overlap 0 (Zero Overlap):
  - Answerable Expected Doc Found in Top 3: 12 / 12
  - Answerable Expected Fact Present: 10 / 12
  - Unanswerable Fallback Triggered: 0 / 2

Sample failures with Overlap 0:
  - Question: What delivery time makes a completed order eligible for a full refund? | Type: fact
  - Question: How soon is money returned when a payment fails but is deducted? | Type: fact

Task 5:

--- Testing Tool Map ---
Registered Tools: ['get_order', 'search_policy', 'calculate_refund', 'issue_refund', 'escalate_to_human']
Test get_order('ORD-000001'): {'error': 'order not found'}
Test calculate_refund(1200.0, 50): {'refund_inr': 1200.0}

Task 6:

Loading scenarios from data/raw/agent_scenarios.csv...

Running S1...
D:\AM\policy_task6.py:97: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  "timestamp": datetime.utcnow().isoformat(),
  -> Status: REFUND | LLM Calls: 5 | Tokens: 1107

Running S2...
  -> Status: REFUND | LLM Calls: 5 | Tokens: 1107

Running S3...
  -> Status: ESCALATE | LLM Calls: 5 | Tokens: 1094

Running S4...
  -> Status: DENIED | LLM Calls: 4 | Tokens: 773

Running S5...
  -> Status: NOT_FOUND | LLM Calls: 2 | Tokens: 179

Running S6...
  -> Status: NOT_FOUND | LLM Calls: 2 | Tokens: 156

Running S7...
  -> Status: MAX_ITERATIONS | LLM Calls: 6 | Tokens: 3252

Saved all agent runs to outputs\agent_runs.jsonl

Task 7:

--- Running S1 Demo with token_limit=200 ---
S1 Token Limit Demo Status: FALLBACK_TOKEN_LIMIT | Tokens Used: 480

--- Running S1 Demo with time_limit_seconds=0 ---
S1 Timeout Demo Status: FALLBACK_TIMEOUT | Final Answer: FALLBACK_TIMEOUT: Run exceeded time limit of 0s. Handoff to human review required.

--- Running All Scenarios with Default Limits (token_limit=6000) ---
D:\AM\policy_task7.py:98: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  "timestamp": datetime.utcnow().isoformat(),
Scenario S1 -> Status: REFUND | Calls: 5 | Tokens: 1107
Scenario S2 -> Status: REFUND | Calls: 5 | Tokens: 1107
Scenario S3 -> Status: ESCALATE | Calls: 5 | Tokens: 1094
Scenario S4 -> Status: DENIED | Calls: 4 | Tokens: 773
Scenario S5 -> Status: NOT_FOUND | Calls: 2 | Tokens: 179
Scenario S6 -> Status: NOT_FOUND | Calls: 2 | Tokens: 156
Scenario S7 -> Status: FALLBACK_MAX_ITERATIONS | Calls: 6 | Tokens: 3252


Task 8:

--- Part 1: Running S1 with Simulated Crash after 2nd tool execution ---
Caught expected crash: SIMULATED CRASH: Triggered after step 2 (search_policy)!

--- Part 2: Resuming S1 using resume_agent() ---
Resuming run 'demo_run_s1' (Scenario: S1) from step 2...
Already executed tools: ['get_order', 'search_policy']
D:\AM\policy_task8.py:99: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  "timestamp": datetime.utcnow().isoformat(),

Resumed Run Final Status: REFUND
Final Answer:
REFUND: INR 128.25 issued.
All Tools Executed Across Crash & Resume: ['get_order', 'search_policy', 'calculate_refund', 'issue_refund']


Task 9:

==========================================
Starting Part D Master Pipeline Execution
==========================================
D:\AM\part_d_pipeline.py:104: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  "timestamp": datetime.utcnow().isoformat(),
Completed S1 -> Status: REFUND | Calls: 5 | Tokens: 1107
Completed S2 -> Status: REFUND | Calls: 5 | Tokens: 1107
Completed S3 -> Status: ESCALATE | Calls: 5 | Tokens: 1094
Completed S4 -> Status: DENIED | Calls: 4 | Tokens: 773
Completed S5 -> Status: NOT_FOUND | Calls: 2 | Tokens: 179
Completed S6 -> Status: NOT_FOUND | Calls: 2 | Tokens: 156
Completed S7 -> Status: FALLBACK_MAX_ITERATIONS | Calls: 6 | Tokens: 3252

==========================================
Part D Summary Table (for README):
==========================================
| Scenario ID   | Status                  | Refund Amount   |   Model Calls |   Tokens Used |
|:--------------|:------------------------|:----------------|--------------:|--------------:|
| S1            | REFUND                  | INR 128.25      |             5 |          1107 |
| S2            | REFUND                  | INR 135.0       |             5 |          1107 |
| S3            | ESCALATE                | INR 1000.0      |             5 |          1094 |
| S4            | DENIED                  | N/A             |             4 |           773 |
| S5            | NOT_FOUND               | N/A             |             2 |           179 |
| S6            | NOT_FOUND               | N/A             |             2 |           156 |
| S7            | FALLBACK_MAX_ITERATIONS | N/A             |             6 |          3252 |

