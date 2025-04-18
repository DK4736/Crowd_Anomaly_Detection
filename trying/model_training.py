import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    roc_auc_score, precision_recall_curve
)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from trying.data_split import X, y  # Ensure this has labels like "person", "dog", etc.

# 🔹 Define normal and anomaly classes
normal_class = "person"  # everything else will be treated as anomaly

# 🔹 Convert to binary labels: 0 = normal, 1 = anomaly
y_bin = np.where(y == normal_class, 0, 1)

# 🔹 Select only numeric features
X = X.select_dtypes(include=[np.number])

# 🔹 Separate anomalies and normals
anomalies = X[y_bin == 1]
normals = X[y_bin == 0]

print(f"Anomalies: {len(anomalies)} | Normals: {len(normals)}")
if len(anomalies) == 0:
    raise ValueError("❌ No anomalies found in dataset. Check your labels or class conversion.")

# 🔹 Train/test split
anomalies_train, anomalies_test = train_test_split(anomalies, test_size=0.5, random_state=42)
normals_train, normals_test = train_test_split(normals, test_size=0.2, random_state=42)

X_train = pd.concat([normals_train], ignore_index=True)  # IsolationForest trained only on normal
y_train = np.zeros(len(X_train))

X_test = pd.concat([normals_test, anomalies_test], ignore_index=True)
y_test = np.concatenate([np.zeros(len(normals_test)), np.ones(len(anomalies_test))])

print("✅ Data split completed:")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print("Class distribution in test set:")
print(pd.Series(y_test).map({0: normal_class, 1: "anomaly"}).value_counts())

# 🔹 Split validation set from train
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 🔹 Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_train_final)

# 🔹 Get anomaly scores (the higher, the more normal — we invert for anomaly probability)
val_scores = model.decision_function(X_val)
test_scores = model.decision_function(X_test)

# 🔹 Normalize and invert scores
scaler = MinMaxScaler()
val_probs = 1 - scaler.fit_transform(val_scores.reshape(-1, 1)).flatten()
test_probs = 1 - scaler.transform(test_scores.reshape(-1, 1)).flatten()

# 🔹 Tune threshold on validation set
best_f1 = 0
best_thresh = 0.5
for thresh in np.arange(0.0, 1.0, 0.01):
    val_preds = (val_probs > thresh).astype(int)
    score = f1_score(y_val, val_preds)
    if score > best_f1:
        best_f1 = score
        best_thresh = thresh

print(f"\n🔍 Best Threshold from Validation: {best_thresh:.2f} | F1: {best_f1:.4f}")

# 🔹 Final test evaluation
y_pred_bin = (test_probs > best_thresh).astype(int)
accuracy = accuracy_score(y_test, y_pred_bin)
f1 = f1_score(y_test, y_pred_bin)
roc_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else np.nan
report = classification_report(y_test, y_pred_bin, zero_division=0)

print("\n📊 Final Evaluation on Test Set:")
print(f"Threshold: {best_thresh:.2f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("Classification Report:")
print(report)

print(f"Number of anomalies in test set: {sum(y_test)}")
print(f"Predictions: {np.unique(y_pred_bin)}")
print(f"True values: {np.unique(y_test)}")

# 🔹 Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, test_probs)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
