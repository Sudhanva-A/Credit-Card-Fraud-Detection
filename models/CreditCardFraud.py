import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from joblib import dump, load
import os

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Loading and Preprocessing
try:
    df = pd.read_csv("creditcard.csv")
    print("Fraud cases:", df['Class'].sum())  # Should show >0
    print("Null values:", df['Class'].isnull().sum())
    # Verify dataset contains both classes
    class_dist = df['Class'].value_counts()
    print("Class distribution:\n", class_dist)
    
    if len(class_dist) < 2:
        raise ValueError("Dataset must contain both fraud (1) and non-fraud (0) cases")

    # Create target and features
    y = df['Class']
    X = df.drop('Class', axis=1)

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    sm = SMOTE(random_state=2)
    X_sm, y_sm = sm.fit_resample(X_train_scaled, y_train)
    print("\nAfter SMOTE:\n", pd.Series(y_sm).value_counts())

except Exception as e:
    print(f"\nError during data loading/preprocessing: {str(e)}")
    exit()

# 2. Enhanced Model Evaluation Function
def evaluate_model(model, X_test, y_test, model_name=""):
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        print(f"\n=== {model_name} Evaluation ===")
        print("Classification Report:")
        print(metrics.classification_report(y_test, y_pred))
        
        # Confusion Matrix
        plt.figure(figsize=(6,4))
        sns.heatmap(metrics.confusion_matrix(y_test, y_pred), 
                    annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legit', 'Fraud'],
                    yticklabels=['Legit', 'Fraud'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
        auc_score = metrics.roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
        plt.plot([0,1], [0,1], 'r--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
        
        return auc_score
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return None

# 3. Logistic Regression with Enhanced Validation
print("\n" + "="*50)
print("Training Logistic Regression...")
try:
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
    logreg.fit(X_sm, y_sm)
    
    # Cross-validation focusing on recall
    cv_scores = cross_val_score(logreg, X_sm, y_sm, cv=5, scoring='recall')
    print(f"Cross-validated Recall: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    logreg_auc = evaluate_model(logreg, X_test_scaled, y_test, "Logistic Regression")

except Exception as e:
    print(f"Logistic Regression failed: {str(e)}")

# 4. Optimized KNN Training
print("\n" + "="*50)
print("Training KNN Classifier...")
try:
    knn = KNeighborsClassifier()
    
    # Parameter grid with reduced options for faster tuning
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['distance']  # Typically better for imbalanced data
    }
    
    knn_grid = GridSearchCV(knn, param_grid, cv=3, scoring='recall', n_jobs=-1)
    knn_grid.fit(X_sm[:10000], y_sm[:10000])  # Use subset for faster tuning
    
    print(f"Best parameters: {knn_grid.best_params_}")
    best_knn = knn_grid.best_estimator_
    knn_auc = evaluate_model(best_knn, X_test_scaled, y_test, "KNN")

except Exception as e:
    print(f"KNN training failed: {str(e)}")

# 5. Random Forest with Feature Importance
print("\n" + "="*50)
print("Training Random Forest...")
try:
    rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=3)
    
    # Optimized parameter grid
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 20],
        'min_samples_split': [5, 10]
    }
    
    rf_grid = GridSearchCV(rf, param_grid, cv=3, scoring='recall', n_jobs=-1)
    rf_grid.fit(X_sm[:20000], y_sm[:20000])  # Use subset for faster training
    
    print(f"Best parameters: {rf_grid.best_params_}")
    best_rf = rf_grid.best_estimator_
    rf_auc = evaluate_model(best_rf, X_test_scaled, y_test, "Random Forest")
    
    # Feature Importance
    feature_imp = pd.Series(best_rf.feature_importances_, index=X.columns)
    plt.figure(figsize=(10,6))
    feature_imp.nlargest(15).plot(kind='barh')
    plt.title('Top 15 Important Features')
    plt.show()

except Exception as e:
    print(f"Random Forest training failed: {str(e)}")

# 6. XGBoost with Early Stopping
print("\n" + "="*50)
print("Training XGBoost...")
print("\n=== XGBoost ===")
try:
    xgb = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=len(y_sm[y_sm==0])/len(y_sm[y_sm==1]),
        early_stopping_rounds=10,
        random_state=42
    )
    xgb.fit(
        X_sm, y_sm,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )
    xgb_auc = evaluate_model(xgb, X_test_scaled, y_test, "XGBoost")
except Exception as e:
    print(f"XGBoost training failed: {str(e)}")
    xgb_auc = 0  # Prevents save error

# 7. Model Persistence
print("\n" + "="*50)
print("Saving Models with Verification")
try:
    # Create directory if needed (NEW: exist_ok prevents errors)
    os.makedirs('best-models', exist_ok=True)
    
    # Save ALL models with descriptive names (NEW)
    model_dict = {
        'LogisticRegression': logreg,
        'KNN': best_knn,
        'RandomForest': best_rf,
        'XGBoost': xgb
    }
    
    # Save each model individually (NEW)
    for name, model in model_dict.items():
        model_path = f'best-models/{name.lower()}.joblib'
        dump(model, model_path, compress=3)  # NEW: Compression added
        print(f"Saved {name} to {model_path}")

    # Determine and save the best model (NEW: with verification)
    model_aucs = {
        'LogisticRegression': logreg_auc or 0,
        'KNN': knn_auc or 0,
        'RandomForest': rf_auc or 0,
        'XGBoost': xgb_auc or 0
    }
    best_model_name = max(model_aucs, key=model_aucs.get)
    best_model_path = 'best-models/best_rf_model.joblib'
    
    # Save with compression and verify (NEW)
    dump(model_dict[best_model_name], best_model_path, compress=3)
    print(f"\n🏆 Best model is {best_model_name} (AUC: {model_aucs[best_model_name]:.4f})")
    
    # Verify the file can be loaded (NEW)
    try:
        test_model = load(best_model_path)
        print("✅ Best model verified and loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to verify best model: {str(e)}")
        raise

    # Save scaler (NEW: with verification)
    scaler_path = 'best-models/scaler.joblib'
    dump(scaler, scaler_path, compress=3)
    try:
        test_scaler = load(scaler_path)
        print("✅ Scaler verified and loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to verify scaler: {str(e)}")
        raise

    # Recover a real fraud sample from the test set
    try:
        fraud_index = np.where(y_test.values == 1)[0][0]  # first fraud example
        fraud_scaled = X_test_scaled[fraud_index].reshape(1, -1)
        fraud_original = scaler.inverse_transform(fraud_scaled)[0]
        fraud_sample_list = [round(float(x), 4) for x in fraud_original]

        print("\n🚨 Real fraud sample (unscaled, ready for HTML):")
        print(fraud_sample_list)
    except Exception as e:
        print(f"⚠️ Failed to extract fraud sample: {str(e)}")

    # Save metadata (NEW)
    with open('best-models/model_info.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Validation AUC: {model_aucs[best_model_name]:.4f}\n")
        f.write(f"Features: {list(X.columns)}\n")
    print("📝 Saved model metadata")

except Exception as e:
    print(f"\n🔥 Error saving models: {str(e)}")
    print("Current directory contents:", os.listdir())
    if os.path.exists('best-models'):
        print("'best-models' contents:", os.listdir('best-models'))
    exit(1)


# 8. Threshold Analysis
print("\n" + "="*50)
print("Threshold Analysis for Fraud Detection")
try:
    if 'best_rf' in locals():
        y_prob = best_rf.predict_proba(X_test_scaled)[:,1]
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        print("\nThreshold | Recall | Precision")
        print("-----------------------------")
        for thresh in thresholds:
            y_pred = (y_prob > thresh).astype(int)
            recall = metrics.recall_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            print(f"{thresh:.2f}      | {recall:.4f} | {precision:.4f}")
        
        # Optimal threshold based on business needs
        optimal_thresh = 0.2
        y_pred_optimal = (y_prob > optimal_thresh).astype(int)
        
        print(f"\nPerformance at threshold {optimal_thresh}:")
        print(metrics.classification_report(y_test, y_pred_optimal))

except Exception as e:
    print(f"Threshold analysis failed: {str(e)}")
print("\n" + "="*50)
print("Fraud Detection Pipeline Completed!")