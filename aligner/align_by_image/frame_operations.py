import numpy as np
import os
from pathlib import Path
import cv2
from retinaface import RetinaFace  # Assuming RetinaFace is installed
from tqdm import tqdm  # For the progress bar
from utilities import conversion_audio_sample_rate as sr


def crop_to_center_percent(frame, perc=1):
    """
    Crop the middle 20% of the given frame.
    """
    if perc == 1:
        return frame

    margin = (1 - perc) / 2
    height, width = frame.shape[:2]
    x_margin = int(width * margin)  # 40% margin to get 20% center
    y_margin = int(height * margin)
    return frame[y_margin : height - y_margin, x_margin : width - x_margin]


def crop_with_noise(frame, box):
    """
    Crop the frame from (x, y) with width w and height h. If the bounding box
    extends out of bounds, generate random noise to fill the missing areas.
    """
    x, y, x2, y2 = box
    w = int(x2 - x)
    h = int(y2 - y)
    x = int(x)
    y = int(y)
    y2 = int(y2)
    x2 = int(x2)

    frame_height, frame_width = frame.shape[:2]

    # Determine the actual cropping area that fits within the frame
    x_start = max(x, 0)
    y_start = max(y, 0)
    x_end = min(x2, frame_width)
    y_end = min(y2, frame_height)

    # Create an empty image of size (h, w) filled with random noise
    result_crop = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    # Determine the region within result_crop where the actual frame content will be placed
    crop_x_start = max(-x, 0)
    crop_y_start = max(-y, 0)

    # Fill the valid area of the result with the actual cropped frame content
    result_crop[
        crop_y_start : crop_y_start + (y_end - y_start),
        crop_x_start : crop_x_start + (x_end - x_start),
    ] = frame[y_start:y_end, x_start:x_end]

    return result_crop


def extract_frame(video_path=None, timestamp=None, frame_idx=None, cap=None):
    """
    Extract a frame from a video at the given timestamp.
    """
    cap_as_arg = not not cap
    if cap is None:
        cap = cv2.VideoCapture(video_path)

    if timestamp is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(fps * timestamp)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not cap_as_arg:
        cap.release()
    return frame if ret else None
