from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trying.data_split import X_train, y_train, X_test, y_test

# Save original test set
X_test_full = X_test.copy()

# Keep only numeric features
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Train Isolation Forest
unsup_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
unsup_model.fit(X_train)

# Predict: -1 for anomaly, 1 for normal ➝ map to 0 (anomaly), 1 (normal)
unsup_preds_test = np.where(unsup_model.predict(X_test) == -1, 0, 1)
unsup_preds_train = np.where(unsup_model.predict(X_train) == -1, 0, 1)

# Count anomalies
anomaly_count = np.sum(unsup_preds_test == 0)
print(f"🔍 [Unsupervised] Detected anomalies: {anomaly_count} / {len(unsup_preds_test)}")

# Print anomalous frame IDs
if 'frame_id' in X_test_full.columns:
    print("📌 Sample Anomalous Frames:")
    print(X_test_full[unsup_preds_test == 0]['frame_id'].head())
else:
    print("⚠️ 'frame_id' column not found in test set.")

# Visualize training anomalies (just first two features)
plt.figure(figsize=(8, 6))
plt.scatter(
    X_train.iloc[:, 0], X_train.iloc[:, 1],
    c=unsup_preds_train, cmap='coolwarm', edgecolors='k'
)

# Custom legend
colors = ['blue', 'red']
labels = ['Normal', 'Anomaly']
for color, label in zip(colors, labels):
    plt.scatter([], [], c=color, label=label)

plt.legend(title="Point Type")
plt.title("Unsupervised Anomaly Detection (Training Data)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()
