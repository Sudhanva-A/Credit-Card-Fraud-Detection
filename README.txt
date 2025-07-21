# Credit Card Fraud Detection App

An end-to-end machine learning web app that detects fraudulent credit card transactions using multiple classification algorithms. Built to handle imbalanced datasets and deliver real-time predictions with export capabilities.

## Features
- Upload transaction features (V1–V28, Time, Amount)
- Switch between ML models: XGBoost, Random Forest, Logistic Regression, KNN
- Real-time fraud detection stats (accuracy, fraud %)
- Export predictions as CSV

## Technologies
- Flask (Python Backend)
- HTML/CSS + JavaScript (Frontend)
- Scikit-learn, XGBoost
- SMOTE for class imbalance handling
- Joblib for model serialization

##  Models Trained & Results

| Model              | Accuracy | AUC Score | Notes                               |
|-------------------|----------|-----------|-------------------------------------|
| **Logistic Regression** | ~95%     | 0.97      | Fast, interpretable, good baseline |
| **K-Nearest Neighbors** | ~96%     | 0.96      | Slower on large datasets            |
| **Random Forest**       | ~96%     | 0.97      | Robust and stable                   |
| **XGBoost**             | ~97%     | 0.98      | Best overall performer              |

- **Best model:** XGBoost (saved in `best-models/` as `.joblib` file)
- **Threshold optimized:** Optimal fraud probability threshold = `0.3`

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection-app.git
cd credit-card-fraud-detection-app
pip install -r requirements.txt
python app.py

