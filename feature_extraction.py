import os
import cv2
import pandas as pd
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Load YOLOv8 Model (pretrained on COCO dataset)
def load_yolov8_model():
    model = YOLO("yolov8s.pt")  # You can switch to yolov8n.pt for faster speed
    return model


# Detect objects in an image
def detect_objects(image_path, model, frame_id):
    """
    Detect objects in the given image using the YOLOv8 model.
    Args:
        image_path (str): Path to the image.
        model: Loaded YOLOv8 model.
        frame_id (int): Unique frame identifier.
    Returns:
        pd.DataFrame: DataFrame containing detected objects with labels, confidence scores, and bounding boxes.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return pd.DataFrame()

    results = model(image_path)[0]

    if results.boxes is None or len(results.boxes) == 0:
        logging.warning(f"No objects detected in {image_path}")
        return pd.DataFrame()

    boxes = results.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    labels = [model.names[int(c)] for c in cls]

    df = pd.DataFrame(xyxy, columns=["xmin", "ymin", "xmax", "ymax"])
    df["confidence"] = conf
    df["class_id"] = cls
    df["label"] = labels
    df["frame_id"] = frame_id

    return df


# Save detections to a CSV file
def save_detections_to_csv(detections, output_csv):
    detections.to_csv(output_csv, index=False)
    logging.info(f"Saved detections to {output_csv}")


# Process all images in a folder
def process_images_in_folder(folder_path, model, output_folder="Detection_Results"):
    os.makedirs(output_folder, exist_ok=True)

    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    logging.info(f"Found {len(images)} images in {folder_path}")

    for frame_id, image_name in enumerate(images):
        image_path = os.path.join(folder_path, image_name)
        detections = detect_objects(image_path, model, frame_id)

        if detections.empty:
            continue

        output_csv = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_detections.csv")
        save_detections_to_csv(detections, output_csv)


# Main Execution
if __name__ == "__main__":
    model = load_yolov8_model()

    training_image_folder = r"C:\Users\21\Desktop\Crowd_Anomaly_Detection-main\ProcessedImages\training_videos_images"
    testing_image_folder = r"C:\Users\21\Desktop\Crowd_Anomaly_Detection-main\ProcessedImages\testing_videos_images"

    logging.info("Processing training images...")
    process_images_in_folder(training_image_folder, model, output_folder="Detection_Results/Training")

    logging.info("Processing testing images...")
    process_images_in_folder(testing_image_folder, model, output_folder="Detection_Results/Testing")
