import pandas as pd
import numpy as np
from lsfr import lsfr  # Import LSFR generation function
from sklearn.preprocessing import MinMaxScaler
import glob  # Import glob module for file matching
import os  # Import os module for extracting filenames

# Use glob to get all the CSV files in the directory
csv_files = glob.glob(r"C:\Users\21\Desktop\Crowd_Anomaly_Detection-main\Detection_Results\Training\*.csv")

# ✅ Debug: Check if files are found
print("Found CSV files:", csv_files)
if not csv_files:
    raise ValueError("No CSV files found in the specified directory!")

# Read all CSV files into a list of DataFrames, extracting frame_id from filenames
df_list = []
for file in csv_files:
    df = pd.read_csv(file)

    # Extract filename without extension as frame_id (or use a numeric ID)
    frame_id = os.path.splitext(os.path.basename(file))[0]
    df["frame_id"] = frame_id  # Assign ID to each row in that file

    df_list.append(df)

# ✅ Debug: Check if data is loaded
if not df_list:
    raise ValueError("No data found! CSV files may be empty.")

# Remove columns that are completely empty or have all NaN values
df_list = [df.dropna(axis=1, how='all') for df in df_list]

# ✅ Ensure we only concatenate if data exists
if df_list:
    df = pd.concat(df_list, ignore_index=True)
else:
    raise ValueError("No valid data to concatenate!")

print("Data successfully loaded and concatenated.")

# Example of generating LSFR sequences
seed = [1, 0, 0, 1]  # Example seed
taps = [0, 2]  # Example taps
length = 100  # Length of the sequence
num_sequences = len(df)  # Number of sequences to generate (can be adjusted)

# Generate LSFR sequences
sequences = [lsfr(seed, taps, length) for _ in range(num_sequences)]
df_lsfr = pd.DataFrame(sequences)

# Add feature extraction logic (e.g., number of ones, zeros, transitions)
df_lsfr['num_ones'] = df_lsfr.apply(lambda row: row.sum(), axis=1)
df_lsfr['num_zeros'] = df_lsfr.apply(lambda row: len(row) - row.sum(), axis=1)
df_lsfr['transition_count'] = df_lsfr.apply(lambda row: np.count_nonzero(np.diff(row)), axis=1)

# Combine the LSFR features with the original dataframe (df)
df_combined = pd.concat([df, df_lsfr], axis=1)

# Check for missing values and handle them (if any)
if df_combined.isnull().any().any():
    print("Missing values found. Filling missing values...")
    df_combined = df_combined.fillna(0)  # Optionally replace NaNs with 0

# Ensure all column names are strings (to avoid the TypeError)
df_combined.columns = df_combined.columns.astype(str)

# Identify non-numeric columns and exclude them from scaling
non_numeric_columns = df_combined.select_dtypes(exclude=[np.number]).columns

# Exclude non-numeric columns from scaling
df_features = df_combined.drop(columns=non_numeric_columns)

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_features)
scaled_df = pd.DataFrame(scaled_data, columns=df_features.columns)

# Combine the scaled data back with the non-numeric columns (if any)
final_df = pd.concat([scaled_df, df_combined[non_numeric_columns]], axis=1)

# ✅ Export df_features for import in other scripts
df_features = final_df

print("Feature engineering completed successfully!")
print(final_df.head())  # Check the final result
