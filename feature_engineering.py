import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load CSV files and merge them into one dataframe
csv_files = glob.glob("Detection_Results/Training/*.csv")  # Ensure correct path
if not csv_files:
    raise FileNotFoundError("No CSV files found in 'Detection_Results/Training' directory.")

df_list = [pd.read_csv(file) for file in csv_files if pd.read_csv(file).shape[0] > 0]  # Ignore empty files
if not df_list:
    raise ValueError("All CSV files are empty. Please check your data.")

df = pd.concat(df_list, ignore_index=True)


# Compute bounding box features
df["bbox_width"] = df["xmax"] - df["xmin"]
df["bbox_height"] = df["ymax"] - df["ymin"]
df["bbox_area"] = df["bbox_width"] * df["bbox_height"]
df["bbox_aspect_ratio"] = df["bbox_height"] / df["bbox_width"]
df["centroid_x"] = (df["xmin"] + df["xmax"]) / 2
df["centroid_y"] = (df["ymin"] + df["ymax"]) / 2

print("Spatial Features Sample:")
print(df.head())

# Ensure frame_id exists
if "frame_id" in df.columns:
    frame_groups = df.groupby("frame_id")
    frame_features = []

    for frame_id, group in frame_groups:
        num_persons = len(group)
        avg_bbox_area = np.mean(group["bbox_area"]) if num_persons > 0 else 0

        # Compute crowd density
        image_area = 1920 * 1080  # Assuming HD resolution
        density = sum(group["bbox_area"]) / image_area

        # Compute distances between centroids
        centroids = np.array([(row["centroid_x"], row["centroid_y"]) for _, row in group.iterrows()])
        avg_distance = np.mean([np.linalg.norm(p1 - p2)
                                for i, p1 in enumerate(centroids)
                                for p2 in centroids[i + 1:]]) if len(centroids) > 1 else 0

        frame_features.append([frame_id, num_persons, avg_bbox_area, density, avg_distance])

    df_features = pd.DataFrame(frame_features, columns=["frame_id", "num_persons", "avg_bbox_area", "density", "avg_distance"])

    # Compute changes between consecutive frames
    df_features = df_features.sort_values(by="frame_id")
    df_features["Δnum_persons"] = df_features["num_persons"].diff().fillna(0)
    df_features["Δavg_bbox_area"] = df_features["avg_bbox_area"].diff().fillna(0)
    df_features["Δdensity"] = df_features["density"].diff().fillna(0)
    df_features["Δavg_distance"] = df_features["avg_distance"].diff().fillna(0)

    print("Temporal Features Sample:")
    print(df_features.head())

    # Normalize the features
    scaler = MinMaxScaler()
    df_features_scaled = pd.DataFrame(scaler.fit_transform(df_features.iloc[:, 1:]), columns=df_features.columns[1:])
    df_features_scaled["frame_id"] = df_features["frame_id"]
    print("Normalized Features Sample:")
    print(df_features_scaled.head())

else:
    print("⚠️ Warning: 'frame_id' not found in the dataset. Skipping temporal feature extraction.")

# Compute speed and direction change per person
df["speed_x"] = df["centroid_x"].diff().fillna(0)
df["speed_y"] = df["centroid_y"].diff().fillna(0)
df["speed"] = np.sqrt(df["speed_x"]**2 + df["speed_y"]**2)
df["direction_change"] = np.arctan2(df["speed_y"], df["speed_x"]).diff().fillna(0)

print("Movement Features Sample:")
print(df.head())
