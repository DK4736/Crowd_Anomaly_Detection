from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from trying.data_split import X_train, y_train, X_test, y_test
from sklearn.metrics import roc_auc_score
import numpy as np

# Drop non-numeric columns
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Ensure y_test is binary: 0 = anomaly, 1 = normal
y_test_binary = (y_test == 1).astype(int)

# Train Isolation Forest model
sup_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
sup_model.fit(X_train)

# Get raw anomaly scores: higher = more normal
raw_scores = sup_model.decision_function(X_test)

# Normalize scores to [0, 1]: higher = more normal
scaler = MinMaxScaler()
norm_scores = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()

# Convert to anomaly probabilities: higher = more anomalous
anomaly_probs = 1 - norm_scores

# Predict binary labels: 0 = anomaly, 1 = normal
y_pred_binary = (anomaly_probs > 0.5).astype(int)

# Evaluate model performance
accuracy = accuracy_score(y_test_binary, y_pred_binary)
f1 = f1_score(y_test_binary, y_pred_binary)
report = classification_report(y_test_binary, y_pred_binary, zero_division=0)
roc_auc = roc_auc_score(y_test_binary, anomaly_probs)

# Print the ROC AUC score

print(report)

# Print the results
print(f"📈 Supervised Evaluation with Anomaly Scores:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"F1 Score (threshold=0.5): {f1:.4f}")
print("Classification Report:")
print(report)
