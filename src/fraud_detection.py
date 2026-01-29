import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("data/credit_card_fraud.csv")

print("\nDataset Loaded Successfully")
print(df.head())

print("\nFraud vs Non-Fraud Count:")
print(df["is_fraud"].value_counts())

# ===============================
# FEATURES & TARGET
# ===============================
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# ===============================
# TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# BASELINE MODEL
# ===============================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n📌 Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

# ===============================
# RANDOM FOREST MODEL
# ===============================
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n🔥 Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# ===============================
# FEATURE IMPORTANCE
# ===============================
importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/feature_importance.png")
plt.show()

# ===============================
# SAVE MODEL
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest_model.pkl")

print("\n✅ Random Forest model saved successfully")
