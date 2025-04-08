from trying.data_split import X_train, y_train, X_test, y_test  # Ensure the correct import

# 🔹 Train the unsupervised model
unsup_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
unsup_model.fit(X_train)

# 🔹 Get unsupervised predictions
unsup_preds = unsup_model.predict(X_test)  # Predictions on test data

# 🔹 Save original test set with frame IDs
X_test_full = X_test.copy()

# 🔹 Compare supervised and unsupervised results
combined_df = X_test_full.copy()

# Add the true labels, unsupervised predictions, and supervised predictions to the dataframe
combined_df['true_label'] = y_test.values
combined_df['unsupervised_pred'] = (unsup_preds == -1).astype(int)  # Converting -1 to 1 for anomalies
combined_df['supervised_pred'] = final_preds  # Ensure final_preds is defined earlier in your code
combined_df['anomaly_score'] = sup_scores_scaled  # Ensure sup_scores_scaled is defined earlier in your code

# Optional: View where the supervised and unsupervised models disagree
print("🔄 Where Supervised and Unsupervised disagree:")
print(combined_df[combined_df['unsupervised_pred'] != combined_df['supervised_pred']].head())

# Optional: Save to CSV for further analysis or visualization
combined_df.to_csv("anomaly_comparison.csv", index=False)
