import tensorflow as tf
import numpy as np
import os
import imageio.v3 as iio  # Use imageio for reading video frames
from tensorflow.keras.applications import resnet50


def ExtractFrames(video_path, frame_rate=1, save_dir=None):
    """
    Extract frames from a video at a specified frame rate.

    Args:
        video_path (str): Path to the input video file.
        frame_rate (int): Number of frames to extract per second.
        save_dir (str, optional): Directory to save the extracted frames.

    Returns:
        list: A list of extracted frames as NumPy arrays.
    """
    frames = []
    reader = iio.imopen(video_path, "r")
    fps = reader.meta["fps"]  # Get video frames per second
    frame_interval = int(fps // frame_rate)  # Interval between frames

    for idx, frame in enumerate(reader):
        if idx % frame_interval == 0:  # Extract frame based on interval
            frames.append(frame)
            if save_dir:  # Save frames if save_dir is specified
                os.makedirs(save_dir, exist_ok=True)
                frame_path = os.path.join(save_dir, f"{os.path.basename(video_path)}_frame_{idx}.jpg")
                iio.imwrite(frame_path, frame)

    reader.close()
    return frames


def PreprocessFrames(frames, res_shape=(224, 224)):
    """
    Preprocess video frames for a TensorFlow/Keras model.

    Args:
        frames (list): List of frames as NumPy arrays.
        res_shape (tuple): Target image size (height, width).

    Returns:
        tf.Tensor: Batch of preprocessed frames.
    """
    preprocessed_frames = []
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    for frame in frames:
        # Convert frame to TensorFlow tensor
        img = tf.convert_to_tensor(frame, dtype=tf.float32) / 255.0  # Normalize to [0, 1]
        img = tf.image.resize(img, res_shape)  # Resize to target dimensions
        img = (img - mean) / std  # Normalize using ImageNet mean and std
        preprocessed_frames.append(img)

    return tf.stack(preprocessed_frames)  # Stack frames into a batch


def PredictFrameClasses(model, processed_frames):
    """
    Predict classes for a batch of video frames.

    Args:
        model (tf.keras.Model): Pre-trained TensorFlow/Keras model.
        processed_frames (tf.Tensor): Batch of preprocessed frames.

    Returns:
        list: List of predicted class labels for the frames.
    """
    predictions = model(processed_frames)
    decoded_predictions = resnet50.decode_predictions(predictions.numpy(), top=1)
    return [pred[0][1] for pred in decoded_predictions]  # Extract top class label


def ProcessMultipleVideos(directory, frame_rate=1):
    """
    Process all videos in a directory and predict classes for their frames.

    Args:
        directory (str): Path to the directory containing video files.
        frame_rate (int): Number of frames to extract per second.

    Returns:
        dict: A dictionary mapping video file names to their predicted classes.
    """
    results = {}
    model = resnet50.ResNet50(weights="imagenet")  # Load pre-trained model once
    video_files = [f for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mov'))]

    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        print(f"Processing video: {video_file}")

        # Step 1: Extract frames
        frames = ExtractFrames(video_path, frame_rate, save_dir="frames")

        # Step 2: Preprocess frames
        preprocessed_frames = PreprocessFrames(frames)

        # Step 3: Predict classes
        predicted_classes = PredictFrameClasses(model, preprocessed_frames)

        # Save results for the current video
        results[video_file] = predicted_classes

    return results


if __name__ == "__main__":
    # Directory containing the videos
    video_directory = "Datasets/AvenueDataset/training_videos"  # Replace with your directory path

    # Process all videos in the directory
    print("Processing multiple videos...")
    video_results = ProcessMultipleVideos(video_directory, frame_rate=1)

    # Print the results
    for video, classes in video_results.items():
        print(f"\nResults for {video}:")
        for i, pred_class in enumerate(classes):
            print(f"  Frame {i + 1}: Predicted class: {pred_class}")
