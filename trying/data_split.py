import pandas as pd
from sklearn.model_selection import train_test_split

from trying.featureengineering import df_features

# Assuming df_features is your DataFrame containing the data
# You can load your data into df_features like this:
# df_features = pd.read_csv("your_data.csv")

# Remove the target column 'class' for the features
X = df_features.drop(columns=['class'])  # Features (all columns except 'class')
y = df_features['class']  # Target column is 'class'

# Check the class distribution
class_counts = y.value_counts()
print(f"Class distribution:\n{class_counts}")

# If a class has only one sample, we need to handle it separately
if class_counts.min() < 2:
    print("⚠️ At least one class has only one sample. It will cause issues with stratified splitting.")

# Filter out classes with only one sample
min_class_count = class_counts[class_counts > 1]
X_filtered = X[y.isin(min_class_count.index)]
y_filtered = y[y.isin(min_class_count.index)]

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Display the first few rows of the split data to check if it's correct
print("Filtered Training features:")
print(X_train.head())
print("Filtered Training target:")
print(y_train.head())
print("Filtered Testing features:")
print(X_test.head())
print("Filtered Testing target:")
print(y_test.head())
