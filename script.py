# 📌 Customer Churn Prediction - Capstone Project (Voting Classifier Version)

# ============== 1. Import Libraries ==============
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

# ============== 2. Load & Inspect Data ==============
df = pd.read_csv("Telco-Customer-Churn.csv")
print(df.head())
print(df.info())
print(df['Churn'].value_counts())

# ============== 3. Data Preprocessing ==============
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode binary categorical features
binary_cols = ['Churn', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'gender']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

# One-hot encode multi-class categorical variables
df = pd.get_dummies(df, drop_first=True)

# Features & Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ============== 4. Baseline Model (Logistic Regression) ==============
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print("\nLogistic Regression Performance:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = abs(log_reg.coef_[0])
sorted_idx = np.argsort(feature_importance)[::-1]
top_features = pd.DataFrame({
    'Feature': X.columns[sorted_idx],
    'Importance': feature_importance[sorted_idx]
})
print("\nTop 10 Important Features:\n", top_features.head(10))

plt.figure(figsize=(8,5))
plt.barh(top_features['Feature'][:10], top_features['Importance'][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Features - Logistic Regression")
plt.show()

# ============== 5. Voting Classifier (Ensemble of ML Models) ==============
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_reg), ('rf', rf), ('gb', gb)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

print("\nVoting Classifier Performance:")
print(classification_report(y_test, y_pred_voting))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_voting))

# ============== 6. Model Comparison ==============
print("\nModel Comparison:")
print("Logistic Regression → Accuracy:", accuracy_score(y_test, y_pred),
      "| Recall:", recall_score(y_test, y_pred),
      "| Precision:", precision_score(y_test, y_pred))

print("Voting Classifier → Accuracy:", accuracy_score(y_test, y_pred_voting),
      "| Recall:", recall_score(y_test, y_pred_voting),
      "| Precision:", precision_score(y_test, y_pred_voting))

# ============== 7. Save Model ==============
import joblib
joblib.dump(voting_clf, "customer_churn_voting_model.pkl")

print("\nModel saved as customer_churn_voting_model.pkl")