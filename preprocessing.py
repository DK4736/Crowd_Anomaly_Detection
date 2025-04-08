import numpy as np
import glob
import os
import math
import cv2
from tqdm import tqdm
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Frame Extraction (No changes needed here)
def Frame_Extractor(
    v_file,
    path="./",
    ext=".avi",
    frames_dir="train_1",
    extract_rate="all",
    frames_ext=".jpg",
):
    os.makedirs(frames_dir, exist_ok=True)
    if ext not in v_file:
        v_file += ext
    cap = cv2.VideoCapture(path + v_file)

    frameRate = cap.get(5)  # Frame rate
    os.makedirs(frames_dir + "/" + v_file, exist_ok=True)
    count = 0
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if isinstance(extract_rate, int):
            if extract_rate > frameRate:
                raise ValueError("`extract_rate` cannot be greater than the frame rate.")
            if frameId % extract_rate == 0:
                filename = f"{frames_dir}/{v_file}/_frame{count}{frames_ext}"
                count += 1
                cv2.imwrite(filename, frame)
        elif extract_rate == "all":
            filename = f"{frames_dir}/{v_file}/{v_file}_frame{count}{frames_ext}"
            count += 1
            cv2.imwrite(filename, frame)
        else:
            raise ValueError("`extract_rate` must be 'all' or an integer.")
    cap.release()

# Read File Names (No changes needed here)
def ReadFileNames(path, frames_ext=".tif"):
    directories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    onlyfiles, file_names = [], []
    for directory in directories:
        files = glob.glob(os.path.join(path, directory, f"*{frames_ext}"))
        file_names.append([os.path.basename(f) for f in files])
        onlyfiles.append(files)
    return onlyfiles, file_names, directories

# Write to JSON (No changes needed here)
def ToJson(obj, name, path="./", json_dir=False):
    os.makedirs(path, exist_ok=True)
    json_path = f"{path}/JSON" if json_dir else path
    os.makedirs(json_path, exist_ok=True)
    with open(f"{json_path}/{name}.json", "w") as f:
        json.dump(obj, f)

# Process Image (Modified for RGB)
def ProcessImg(img_name, read_path, write=True, write_path=None, res_shape=(128, 128)):
    img = cv2.imread(read_path)  # Read image in BGR format
    img = cv2.resize(img, res_shape)
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if write:
        os.makedirs(write_path, exist_ok=True)
        # Convert back to BGR for saving using OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{write_path}/{img_name}", img_bgr)
    return img_rgb  # Return RGB image array

# Global Normalization (Modified for RGB)
def GlobalNormalization(img_list, name=None, path="Train_Data", save_data=True):
    img_arr = np.array(img_list, dtype=np.float32)  # Convert list to NumPy array
    logging.info(f"Array shape: {img_arr.shape}, dtype: {img_arr.dtype}")
    # Normalize the data
    mean = img_arr.mean()
    std = img_arr.std()
    img_arr = (img_arr - mean) / std
    img_arr = np.clip(img_arr, 0, 1)
    if save_data:
        if name is None:
            raise ValueError("Provide a valid name for saving data.")
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/{name}.npy", img_arr)  # Save as .npy file
        logging.info(f"Data saved successfully at {path}/{name}.npy")
    return img_arr
# Video to Frames (No changes needed here)
def Vid2Frame(vid_path, frames_dir, ext_vid=".avi", frames_ext=".tif"):
    vids = glob.glob(os.path.join(vid_path, f"*{ext_vid}"))
    for vid in tqdm(vids, desc="Extracting frames"):
        path, v_file = os.path.split(vid)
        Frame_Extractor(v_file, path=path + "/", ext=ext_vid, frames_dir=frames_dir, extract_rate="all", frames_ext=frames_ext)

# Fit Preprocessing (No changes needed except handling RGB)
def Fit_Preprocessing(path, frames_ext):
    onlyfiles, file_names, dirs = ReadFileNames(path, frames_ext)
    img_list = []
    for i in tqdm(range(len(onlyfiles)), desc="Processing images"):
        images = onlyfiles[i]
        for count, img in enumerate(images):
            img_name = f"{dirs[i]}_{file_names[i][count]}"
            write_path = f"ProcessedImages/{os.path.basename(path)}"
            rgb_img = ProcessImg(img_name, read_path=img, write=True, write_path=write_path, res_shape=(227, 227))
            img_list.append(rgb_img)
    return img_list

# Main Execution (No changes needed)
if __name__ == "__main__":
    dataset_path = "Datasets/AvenueDataset/"
    vid_paths = [os.path.join(dataset_path, "training_videos"), os.path.join(dataset_path, "testing_videos")]
    frame_paths = []

    # Extract frames from videos
    for vid_path in vid_paths:
        frames_dir = f"{vid_path}_images"
        Vid2Frame(vid_path, frames_dir, ext_vid=".avi", frames_ext=".tif")
        frame_paths.append(frames_dir)

    # Preprocess frames
    for path in frame_paths:
        img_list = Fit_Preprocessing(path, frames_ext=".tif")
        name = f"Test_{os.path.basename(path)}" if "test" in path.lower() else f"Train_{os.path.basename(path)}"
        GlobalNormalization(img_list, name=name, path="Test_Data" if "test" in path.lower() else "Train_Data", save_data=True)