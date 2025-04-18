import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 🔸 Load and merge all CSV detection files
csv_files = glob.glob("Detection_Results/Training/*.csv")
if not csv_files:
    raise FileNotFoundError("❌ No CSV files found in Detection_Results/Training")

# Read non-empty CSVs
df_list = [pd.read_csv(file) for file in csv_files if pd.read_csv(file).shape[0] > 0]
if not df_list:
    raise ValueError("❌ All detection CSV files are empty.")

df = pd.concat(df_list, ignore_index=True)

# 🔸 Optional: Filter only person detections (label: "person")
df = df[df["label"] == "person"].reset_index(drop=True)

# Compute spatial features
df["bbox_width"] = df["xmax"] - df["xmin"]
df["bbox_height"] = df["ymax"] - df["ymin"]
df["bbox_area"] = df["bbox_width"] * df["bbox_height"]
df["bbox_aspect_ratio"] = df["bbox_height"] / df["bbox_width"]
df["centroid_x"] = (df["xmin"] + df["xmax"]) / 2
df["centroid_y"] = (df["ymin"] + df["ymax"]) / 2

print("\n📦 Spatial Features Sample:")
print(df.head())

# Compute frame-level features
if "frame_id" in df.columns:
    frame_features = []
    for frame_id, group in df.groupby("frame_id"):
        num_persons = len(group)
        avg_bbox_area = group["bbox_area"].mean()
        image_area = 1920 * 1080  # Customize if different resolution
        density = group["bbox_area"].sum() / image_area

        # Centroid distance
        centroids = group[["centroid_x", "centroid_y"]].values
        if len(centroids) > 1:
            distances = [np.linalg.norm(p1 - p2) for i, p1 in enumerate(centroids) for p2 in centroids[i+1:]]
            avg_distance = np.mean(distances)
        else:
            avg_distance = 0

        frame_features.append([frame_id, num_persons, avg_bbox_area, density, avg_distance])

    df_features = pd.DataFrame(frame_features, columns=["frame_id", "num_persons", "avg_bbox_area", "density", "avg_distance"])
    df_features = df_features.sort_values("frame_id")

    # 🔸 Temporal feature deltas
    for col in ["num_persons", "avg_bbox_area", "density", "avg_distance"]:
        df_features[f"Δ{col}"] = df_features[col].diff().fillna(0)

    print("\n🕒 Temporal Features Sample:")
    print(df_features.head())

    # Normalize features
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features.iloc[:, 1:]), columns=df_features.columns[1:])
    df_scaled["frame_id"] = df_features["frame_id"].values
    print("\n📊 Normalized Features Sample:")
    print(df_scaled.head())
else:
    print("⚠️ 'frame_id' not found. Skipping temporal features.")

# Optional: Person movement features across frames
# Assuming you have an object ID column (if not, motion isn't trackable per person!)
# For now: Approximate using frame-wise diffs
df["speed_x"] = df.groupby("frame_id")["centroid_x"].diff().fillna(0)
df["speed_y"] = df.groupby("frame_id")["centroid_y"].diff().fillna(0)
df["speed"] = np.sqrt(df["speed_x"]**2 + df["speed_y"]**2)
df["direction_change"] = np.arctan2(df["speed_y"], df["speed_x"]).diff().fillna(0)

print("\n🚶 Movement Features Sample:")
print(df[["frame_id", "centroid_x", "centroid_y", "speed", "direction_change"]].head())
