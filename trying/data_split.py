import pandas as pd
from sklearn.model_selection import train_test_split

# 🔍 Load engineered features
df = pd.read_csv("engineered_features.csv")

# 🎯 Rename label column
if 'label' not in df.columns:
    raise KeyError("Missing 'label' column in features!")
df.rename(columns={'label': 'class'}, inplace=True)

X = df.drop(columns=['class'])
y = df['class']

# 📉 Remove classes with only 1 sample (necessary for stratification)
class_counts = y.value_counts()
valid_classes = class_counts[class_counts > 1].index
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# 🧪 Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ✅ Summary
print("✅ Data split completed:")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print("Class distribution in test set:")
print(pd.Series(y_test).value_counts())
