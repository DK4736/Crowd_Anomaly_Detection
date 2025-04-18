import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lsfr import lsfr
import glob
import os

# 📁 Load CSV files
csv_files = glob.glob(r"C:\Users\21\Desktop\Crowd_Anomaly_Detection-main\Detection_Results\Training\*.csv")
if not csv_files:
    raise FileNotFoundError("No CSV files found!")

# Combine all CSVs into a single DataFrame
df_list = []
for file in csv_files:
    df = pd.read_csv(file).dropna(axis=1, how='all')  # Drop completely empty columns
    if df.empty:
        print(f"⚠️ Warning: {file} is empty and was skipped.")
        continue
    df['frame_id'] = os.path.splitext(os.path.basename(file))[0]
    df_list.append(df)

# Concatenate all data into a single DataFrame
df = pd.concat(df_list, ignore_index=True)
if df.empty:
    raise ValueError("After combining, the resulting DataFrame is empty!")

print(f"✅ Loaded {len(df)} rows from {len(df_list)} files.")

# 🧠 Generate LSFR sequences
seed = [1, 0, 0, 1]
taps = [0, 2]
sequence_length = 100

# Ensure that LSFR sequences match the number of rows in the DataFrame
if len(df) != len(csv_files):
    print("⚠️ Warning: The number of rows in the final DataFrame does not match the number of CSV files.")

# Generate LSFR sequences
lsfr_sequences = [lsfr(seed, taps, sequence_length) for _ in range(len(df))]
df_lsfr = pd.DataFrame(lsfr_sequences)

# 🔢 Add LSFR features: num_ones, num_zeros, transition_count
df_lsfr['num_ones'] = df_lsfr.sum(axis=1)
df_lsfr['num_zeros'] = sequence_length - df_lsfr['num_ones']
df_lsfr['transition_count'] = df_lsfr.apply(lambda row: np.count_nonzero(np.diff(row)), axis=1)

# 🔗 Combine LSFR features with the original DataFrame
df_combined = pd.concat([df.reset_index(drop=True), df_lsfr.reset_index(drop=True)], axis=1)

# Fill missing values in the combined DataFrame
df_combined.fillna(0, inplace=True)

# 🧪 Normalize numeric features
non_numeric_columns = df_combined.select_dtypes(exclude=[np.number]).columns
numeric_columns = df_combined.drop(columns=non_numeric_columns)

# Convert column names to strings to avoid TypeError
numeric_columns.columns = numeric_columns.columns.astype(str)

# Apply MinMax scaling only on numeric columns
scaler = MinMaxScaler()
scaled = scaler.fit_transform(numeric_columns)
scaled_df = pd.DataFrame(scaled, columns=numeric_columns.columns)

# Combine the scaled numeric data with the non-numeric columns
final_df = pd.concat([scaled_df, df_combined[non_numeric_columns].reset_index(drop=True)], axis=1)

# ✅ Save the final DataFrame to CSV
output_file = "engineered_features.csv"
final_df.to_csv(output_file, index=False)
print(f"✅ Feature engineering completed and saved as {output_file}")
