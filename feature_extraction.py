import os
import cv2
import pandas as pd
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Load YOLOv5 Model (pretrained on COCO dataset)
def load_yolov5_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model


# Detect objects in an image
def detect_objects(image_path, model, frame_id):
    """
    Detect objects in the given image using the YOLOv5 model.
    Args:
        image_path (str): Path to the image.
        model: Loaded YOLOv5 model.
        frame_id (int): Unique frame identifier.
    Returns:
        pd.DataFrame: DataFrame containing detected objects with labels, confidence scores, and bounding boxes.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return pd.DataFrame()  # Return empty DataFrame if the image fails to load

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(image_rgb)

    # Extract detections as a pandas DataFrame
    detections = results.pandas().xyxy[0]  # Bounding box coordinates and labels

    # 🔹 Add frame_id column
    detections["frame_id"] = frame_id

    return detections


# Save detections to a CSV file
def save_detections_to_csv(detections, output_csv):
    """
    Save object detections to a CSV file.
    Args:
        detections (pd.DataFrame): DataFrame containing detection results.
        output_csv (str): Path to the output CSV file.
    """
    detections.to_csv(output_csv, index=False)
    logging.info(f"Saved detections to {output_csv}")


# Process all images in a folder
def process_images_in_folder(folder_path, model, output_folder="Detection_Results"):
    os.makedirs(output_folder, exist_ok=True)

    # 🔹 Sort images by name to maintain order
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    logging.info(f"Found {len(images)} images in {folder_path}")

    for frame_id, image_name in enumerate(images):  # Assign frame_id based on sorted order
        image_path = os.path.join(folder_path, image_name)
        detections = detect_objects(image_path, model, frame_id)

        if detections.empty:
            logging.warning(f"No objects detected in {image_path}")
            continue

        output_csv = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_detections.csv")
        save_detections_to_csv(detections, output_csv)


# Main Execution
if __name__ == "__main__":
    # Load the YOLOv5 model
    model = load_yolov5_model()

    # Paths to Training and Testing Image Folders
    training_image_folder = r"C:\Users\21\Desktop\Crowd_Anomaly_Detection-main\ProcessedImages\training_videos_images"
    testing_image_folder = r"C:\Users\21\Desktop\Crowd_Anomaly_Detection-main\ProcessedImages\testing_videos_images"

    # Process Training Images
    logging.info("Processing training images...")
    process_images_in_folder(training_image_folder, model, output_folder="Detection_Results/Training")

    # Process Testing Images
    logging.info("Processing testing images...")
    process_images_in_folder(testing_image_folder, model, output_folder="Detection_Results/Testing")
