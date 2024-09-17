#############################
# Video processing ##
#############################

import cv2
import imagehash
from PIL import Image
import numpy as np


def extract_frames(video_file, region=None, resolution=1000):
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    duration_ms = int(video.get(cv2.CAP_PROP_FRAME_COUNT) * 1000 / fps)
    frames = []

    success, frame = video.read()

    height, width, _ = frame.shape
    print("%region", region)

    if not region:
        region = (0, 0, 1, 1)

    region = (
        int(region[0] * width),
        int(region[1] * height),
        int(region[2] * width),
        int(region[3] * height),
    )

    print("region", region)
    middle_frame_timestamp = 1000 * (duration_ms / 1000 // 16)

    for timestamp_ms in range(
        0, duration_ms, resolution
    ):  # Step through the video by seconds
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        success, frame = video.read()
        print("READING FRAME", timestamp_ms / 1000, end="\r")
        if success:
            frame = frame[
                region[1] : region[1] + region[3], region[0] : region[0] + region[2]
            ]
            frames.append(frame)
            if timestamp_ms == middle_frame_timestamp:
                cv2.imwrite("middle_frame.jpg", frame)

    video.release()
    return frames


def compute_hashes(frames):
    return [
        imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        for frame in frames
    ]


SIMILARITY_THRESH = 0.5


def find_first_occurrences(react_hashes, resolution):
    timestamps = []
    seen_hashes = {}

    for i, react_hash in enumerate(react_hashes):
        current_time = i * resolution / 1000
        if react_hash in seen_hashes:
            # If the current time is at least 1 second later than the first occurrence, record the timestamp
            if current_time - seen_hashes[react_hash] < SIMILARITY_THRESH:
                timestamps.append(current_time)
                seen_hashes[react_hash] = current_time
        else:
            # If the frame has not been seen before, record its hash and the current time
            seen_hashes[react_hash] = current_time
            timestamps.append(current_time)

    return timestamps


import os
import subprocess
import tempfile


def crop_video(reaction_vid, region):
    # Uses ffmpeg to crop a given video the specified region.
    # The cropped video should be written to a temporary file.
    # Returns a path to the cropped video.
    # reaction_vid: the target video to crop
    # region: the area of the video to crop, as a tuple of
    # (% from left edge, % from top, % of total width, % of total height)

    nam, ext = os.path.splitext(os.path.basename(reaction_vid))
    fname = f"{nam}-cropped{ext}"
    temp_file = os.path.join(tempfile.gettempdir(), fname)

    if not os.path.exists(temp_file):
        # Region parameters (assumed to be percentages)
        left_percent, top_percent, width_percent, height_percent = region

        # ffmpeg command for cropping (using the "crop" filter)
        # Note: ffmpeg takes crop parameters as: width:height:x:y
        command = f'ffmpeg -y -i "{reaction_vid}" -vf "crop=in_w*{width_percent}:in_h*{height_percent}:in_w*{left_percent}:in_h*{top_percent}" "{temp_file}"'

        print("Cropping to", temp_file)
        print(command)

        # Run the command
        subprocess.run(command, shell=True, check=True)

    return temp_file


def downsample(reaction_vid, target_resolution):
    # Create a temp file for the downscaled video
    nam, ext = os.path.splitext(os.path.basename(reaction_vid))
    fname = f"{nam}-downsampled-{target_resolution}{ext}"
    temp_file = os.path.join(tempfile.gettempdir(), fname)
    if not os.path.exists(temp_file):
        # Command to downscale the video
        command = f'ffmpeg -y -i "{reaction_vid}" -vf "scale={target_resolution}:-2" "{temp_file}"'
        print("Downsampling to", temp_file)
        print(command)
        # Run the command
        subprocess.run(command, shell=True, check=True)

    return temp_file


def filter_by_video_analysis(reaction_vid, region, resolution=1):
    cropped_video = crop_video(reaction_vid, region)
    downsampled_reaction_vid = downsample(cropped_video, target_resolution=80)
    react_frames = extract_frames(downsampled_reaction_vid, resolution=resolution)
    react_hashes = compute_hashes(react_frames)

    timestamps = find_first_occurrences(react_hashes, resolution)
    print(timestamps)

    base, ext = os.path.splitext(reaction_vid)
    segments = create_video_segments(reaction_vid, timestamps, ext, resolution)
    concatenate_video_segments(segments, f"{base}-vidstripped-{resolution}{ext}")


# # Knox Hill / The Hunger
# width = 1792
# height = 977

# left = 1189
# right = 1642

# top = 646
# bottom = 950

# region = (left / width, top / height, (right - left) / width, (bottom - top) / height)

# reaction_vid = os.path.join("Ren - The Hunger", "reactions", "CAN HE RAP THO？! ｜ Ren - The Hunger knox-trimmed.mp4")
# resolution = 1

# filter_by_video_analysis(reaction_vid, region, resolution)


# # h8tful jay
# width = 1785
# height = 1005

# left = 200
# right = 726

# top = 563
# bottom = 986

# region = (left / width, top / height, (right - left) / width, (bottom - top) / height)

# reaction_vid = os.path.join("Ren - The Hunger", "reactions", "REN-HUNGER [REACTION] h8tful jay-trimmed.mp4")
# resolution = 1

# filter_by_video_analysis(reaction_vid, region, resolution)


# # Anthony Ray
# width = 1790
# height = 1007

# left = 200
# right = 832

# top = 50
# bottom = 532

# region = (left / width, top / height, (right - left) / width, (bottom - top) / height)

# reaction_vid = os.path.join("Ren - The Hunger", "reactions", "Rapper REACTS to REN - THE HUNGER anthony ray-truncated.mp4")
# resolution = 1

# filter_by_video_analysis(reaction_vid, region, resolution)


# # Stevie Knight
# width = 1790
# height = 1007

# left = 250
# right = 877

# top = 235
# bottom = 795

# region = (left / width, top / height, (right - left) / width, (bottom - top) / height)

# reaction_vid = os.path.join("Ren - The Hunger", "reactions", "stevie_knight-truncated.mp4")
# resolution = 1

# filter_by_video_analysis(reaction_vid, region, resolution)


# # thatsingerreacts
# width = 1790
# height = 1007

# left = 1198
# right = 1590

# top = 40
# bottom = 405

# region = (left / width, top / height, (right - left) / width, (bottom - top) / height)

# reaction_vid = os.path.join("Ren - The Hunger", "reactions", "Ren - The Hunger ｜ Reaction thatsingerreacts.mp4")
# resolution = 1

# filter_by_video_analysis(reaction_vid, region, resolution)
