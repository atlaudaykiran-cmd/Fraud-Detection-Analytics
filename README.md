# Fraud-Detection-Analytics
# CODTECH-Task4

Name: ATLA UDAY KIRAN

Company: CODTECH IT SOLUTIONS
ID: CITS5172
Domain: DATA ANALYTICS
Duration: 18th JUNE 2026 to 16th JULY 2026
Mentor: NEELA SANTHOSH KUMAR

## 🔍 Project Overview: Fraud Detection Analytics

As part of my role at CODTECH IT SOLUTIONS, I built a Fraud Detection
Analytics system on a synthetic financial transactions dataset. This project
combines exploratory data analysis with machine learning classification models
to identify fraudulent transactions automatically.

### Objective

To detect fraudulent financial transactions by analyzing patterns in
transaction data — including amount, time, location, and transaction type —
and to build machine learning models that can flag suspicious activity with
high accuracy.

### Steps Taken

**1. Data Generation and Loading:**
- Generated a synthetic dataset of 5,000 financial transactions across 2024.
- Features include Transaction Type, Amount, Hour of Day, Location (Domestic/International), Card Type, Number of Transactions per Day, Account Age, Failed Attempts, and Distance from Home.
- Fraud probability was built using realistic risk factors (large amounts, international location, late-night hours, failed login attempts).
- Final fraud rate: ~5.7% (realistic class imbalance).

**2. Exploratory Data Analysis (EDA):**
- Visualized the overall Fraud vs Legitimate transaction split.
- Compared fraud rates across different transaction types.
- Analyzed transaction amount distribution for fraudulent vs legitimate transactions.
- Plotted fraud rate by hour of day to identify high-risk time windows.

**3. Preprocessing:**
- Label-encoded categorical features (TransactionType, CardType, Location).
- Applied class weighting to handle the class imbalance problem.
- Split data 75/25 for training and testing, stratified by fraud label.
- Standardized features for the Logistic Regression model.

**4. Model Training and Evaluation:**
- Trained **Logistic Regression** (with class balancing) and **Random Forest Classifier**.
- Evaluated with accuracy, precision, recall, F1-score, ROC-AUC, and Average Precision.
- Plotted ROC curves and Precision-Recall curves for both models.
- Generated confusion matrix to analyze false positive/negative trade-offs.
- Extracted feature importance from the Random Forest model to identify the top fraud signals.

### Key Insights

- **International transactions** and **large amounts (> $3,000)** are the strongest predictors of fraud.
- Fraud rate is highest between **midnight and 4 AM**, indicating late-night activity as a major risk signal.
- **International** transaction type has the highest per-category fraud rate.
- **Failed login attempts** and **high daily transaction counts** also significantly elevate fraud risk.
- Logistic Regression achieved a better recall for fraud (67%) vs Random Forest (1%), making it more suitable for fraud detection despite lower accuracy — catching fraud matters more than overall accuracy.
- The Precision-Recall curve highlights the classic precision-recall trade-off in imbalanced classification problems.

### Conclusion

This project demonstrated the full fraud detection pipeline — from data
exploration and feature engineering to model training and evaluation. The
results highlight that in fraud detection, **recall is more important than
accuracy** — missing a fraud is more costly than a false alarm. Class
weighting and threshold tuning are critical techniques for imbalanced datasets.

---

## Output Graphs

### 1. Fraud vs Legitimate Transaction Distribution
<img width="660" height="550" alt="01_fraud_distribution" src="https://github.com/user-attachments/assets/7a39ae1d-3bbf-4a2d-b989-68cf846b5da6" />

### 2. Fraud Rate (%) by Transaction Type
<img width="990" height="550" alt="02_fraud_by_transaction_type" src="https://github.com/user-attachments/assets/ac77a13f-3947-428a-b34f-85ab969d8cd6" />

### 3. Transaction Amount Distribution: Fraud vs Legitimate
<img width="990" height="550" alt="03_amount_distribution" src="https://github.com/user-attachments/assets/1865b440-ffd1-4134-bf2f-4211a3d19dd8" />

### 4. Fraud Rate by Hour of Day
<img width="1210" height="550" alt="04_fraud_by_hour" src="https://github.com/user-attachments/assets/fb558e92-f7ed-45fc-8de1-18e0f11bf0e2" />

### 5. Confusion Matrix — Random Forest
<img width="660" height="550" alt="05_confusion_matrix" src="https://github.com/user-attachments/assets/85f6303d-bfc5-4b11-8a64-e61b2b79e407" />

### 6. ROC Curve — Fraud Detection
<img width="770" height="660" alt="06_roc_curve" src="https://github.com/user-attachments/assets/5d142df0-9728-4e98-9163-16ee9f5a5e5b" />

### 7. Precision-Recall Curve
<img width="770" height="660" alt="07_precision_recall_curve" src="https://github.com/user-attachments/assets/b44357e1-f083-4ee2-b181-f9d3a79765fa" />

### 8. Feature Importance — Random Forest
<img width="990" height="660" alt="08_feature_importance" src="https://github.com/user-attachments/assets/9b6cf995-7840-4242-a779-b071f68fdbf7" />


## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python generate_data.py
python fraud_detection_analytics.py
```

## Files in this Repository

| File | Description |
|---|---|
| `Fraud_Detection_Analytics.ipynb` | Full notebook with code and analysis |
| `fraud_detection_analytics.py` | Main fraud detection analysis and modeling script |
| `data/transactions.csv` | Generated dataset (5,000 transactions) |
| `output/` | All generated chart images and model summary |

