import cv2
import pandas as pd
import os

# 🔹 Load the anomaly detection results from CSV
csv_file = "detected_anomalies.csv"  # Path to your CSV file
image_folder = "C:\\Users\21\Desktop\Crowd_Anomaly_Detection-main\ProcessedImages\training_videos_images"  # Folder containing frames/images
output_folder = "output"  # Folder to save annotated images

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(csv_file)

# 🔹 Loop through anomalies and process images
for index, row in df.iterrows():
    img_path = os.path.join(image_folder, f"{index}.jpg")  # Using index as frame_id

    # Check if the image file exists
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    # 🔹 Read the image
    img = cv2.imread(img_path)

    # 🔹 Get bounding box coordinates (assuming values are normalized 0-1)
    h, w, _ = img.shape  # Get original image size
    xmin = int(row["xmin"] * w)
    ymin = int(row["ymin"] * h)
    xmax = int(row["xmax"] * w)
    ymax = int(row["ymax"] * h)

    # 🔹 Draw bounding box
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red box
    cv2.putText(img, "Anomaly", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 🔹 Save or show the result
    output_path = os.path.join(output_folder, f"annotated_{index}.jpg")
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

    # (Optional) Show image
    cv2.imshow("Detected Anomaly", img)
    cv2.waitKey(100)  # Show for 100ms
    cv2.destroyAllWindows()
