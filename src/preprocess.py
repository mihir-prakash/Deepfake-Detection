import os
import cv2

def extract_frames(video_path, output_dir, interval=30):
    """
    Extract frames from a video at a fixed interval.
    """
    os.makedirs(output_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    video.release()

def preprocess_huggingface_dataset(dataset, output_dir):
    """
    Extract frames from Hugging Face Wild Deepfake dataset.
    """
    for item in dataset:
        video_path = item['video_path']
        video_output_dir = os.path.join(output_dir, os.path.basename(video_path))
        extract_frames(video_path, video_output_dir)
