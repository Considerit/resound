import glob, os, math
import tempfile
from typing import List, Tuple
import soundfile as sf
import numpy as np
import subprocess
import cv2
import json
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    ColorClip,
    CompositeVideoClip,
)
from decimal import Decimal, getcontext
import os
import subprocess
import glob
import ffmpeg
import io


def sec_to_time(sec):
    minutes = math.floor(sec / 60)
    seconds = round(sec % 60, 1)  # Round seconds to 1 decimal place
    return f"{minutes}:{seconds:04.1f}"  # Ensure seconds are always two places


import pickle, json


def save_object_to_file(output_file, object, check_collisions=False):
    if check_collisions:
        data_on_disk = read_object_from_file(output_file)
        for k, v in data_on_disk.items():
            if k not in object:
                object[k] = v

    # Open the file in binary mode for pickle, text mode for JSON
    if output_file.endswith(".pckl"):
        with open(output_file, "wb") as f:
            pickle.dump(object, f)
    else:
        with open(output_file, "w") as f:  # Open in text mode for JSON
            json.dump(object, f, indent=4)


def read_object_from_file(input_file):
    if os.path.exists(input_file):
        # print(f"reading {input_file}")
        if input_file.endswith(".pckl"):
            with open(input_file, "rb") as f:  # Open in binary mode for pickle
                data = pickle.load(f)
        else:
            with open(input_file, "r") as f:  # Open in text mode for JSON
                data = json.load(f)
    else:
        return None

    return data


conversion_frame_rate = 60
conversion_audio_sample_rate = 44100


def samples_per_frame():
    return int(conversion_audio_sample_rate / conversion_frame_rate)


def universal_frame_rate():
    return conversion_frame_rate


def extract_audio(
    video_file,
    output_dir=None,
    sample_rate=None,
    preserve_silence=True,
    convert_to_mono=True,
    keep_file=False,
):
    if output_dir is None:
        output_dir = os.path.dirname(video_file)
    if sample_rate is None:
        sample_rate = conversion_audio_sample_rate

    # Construct the output file path (if needed)
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.wav")

    # ffmpeg command to extract the audio
    command = [
        "ffmpeg",
        "-i",
        video_file,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
    ]

    # If preserving silence, add the apad filter
    if preserve_silence:
        duration_command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_file,
        ]
        duration = float(subprocess.check_output(duration_command).decode("utf-8").strip())
        pad_duration = int(duration * sample_rate)  # Calculate the number of samples to pad
        command.extend(["-af", f"apad=whole_len={pad_duration}"])

    audio_input = None
    if not keep_file:
        # If keep_file is False, process the audio in memory
        command.extend(["-f", "wav", "-"])  # Output audio as WAV to stdout

        # Capture the audio in a pipe
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Load the audio from memory into a BytesIO buffer
        audio_input = io.BytesIO(result.stdout)
        audio_input.seek(0)  # Rewind the BytesIO object to the beginning
        output_file = None

    else:
        audio_input = output_file
        if not os.path.exists(output_file):
            # If keep_file is True, write the audio to a file
            command.append(output_file)
            # Run the ffmpeg command, passing arguments as a list and not using shell=True
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Read the audio from the file and return
    try:
        audio_data, sample_rate = sf.read(audio_input)
        if audio_data.ndim > 1 and convert_to_mono:
            audio_data = np.mean(audio_data, axis=1)
    except Exception as e:
        print(f"FAILED TO LOAD AUDIO FROM {video_file}", e)
        raise (e)
    return (audio_data, sample_rate, output_file)


def get_frame_rate(video_file: str) -> float:
    cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1"
    args = cmd.split(" ") + [video_file]
    ffprobe_output = subprocess.check_output(args).decode("utf-8")
    # The output has the form 'num/den' so we compute the ratio to get the frame rate as a decimal number
    num, den = map(int, ffprobe_output.split("/"))
    return num / den


def compute_precision_recall(output_segments, ground_truth, tolerance=0.5):
    tp = 0  # true positives
    fp = 0  # false positives
    fn = 0  # false negatives

    # Create a list to keep track of which ground truth segments have been matched
    matched = [False] * len(ground_truth)

    for out_start, out_end, base_start, base_end, filler in output_segments:
        out_start = float(out_start)
        out_end = float(out_end)

        if filler:
            continue

        match_found = False
        for i, (gt_start, gt_end) in enumerate(ground_truth):
            if abs(out_start - gt_start) <= tolerance and abs(out_end - gt_end) <= tolerance:
                match_found = True
                matched[i] = True
                break
        if match_found:
            tp += 1
        else:
            fp += 1

    # Any unmatched ground truth segments are false negatives
    fn = len(ground_truth) - sum(matched)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    print("ground truth:", ground_truth)

    print(f"\nPrecision = {precision}, Recall = {recall}\n")
    return precision, recall


def is_close(a, b, rel_tol=None, abs_tol=None):
    if rel_tol is None and abs_tol is None:
        rel_tol = Decimal("1e-09")
        abs_tol = Decimal("0.0")
    elif rel_tol is None:
        rel_tol = abs_tol
    elif abs_tol is None:
        abs_tol = rel_tol

    # Convert inputs to Decimal if necessary
    if isinstance(a, float):
        a = Decimal(str(a))
    if isinstance(b, float):
        b = Decimal(str(b))

    # Convert tolerance values to Decimal
    if isinstance(rel_tol, float):
        rel_tol = Decimal(str(rel_tol))
    if isinstance(abs_tol, float):
        abs_tol = Decimal(str(abs_tol))

    difference = abs(a - b)
    # print(f"difference={difference}  of a {a} and b {b}")
    return difference <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


from pympler import muppy, summary

import sys


def get_size(obj, seen=None):
    """Recursively find the size of an object and its attributes."""
    # Keep track of objects we've already seen to avoid infinite recursion
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    # If the object is a container, add the sizes of its items
    if isinstance(obj, (list, tuple, set, dict)):
        size += sum(get_size(i, seen) for i in obj)
    elif hasattr(obj, "__dict__") and isinstance(obj.__dict__, dict):
        size += sum(get_size(v, seen) for v in obj.__dict__.values())

    return size


def print_memory_consumption():
    # Get all objects currently in memory
    all_objects = muppy.get_objects()

    # Now, sort individual objects by their size
    largest_objs = sorted(all_objects, key=lambda x: get_size(x), reverse=True)

    for obj in largest_objs[:5]:
        print(object_description(obj), get_size(obj))


def object_description(obj):
    """Provide a short description of the object along with a small part of its content."""
    if isinstance(obj, (list, tuple)):
        preview = ", ".join([str(x) for x in obj[:3]])
        return f"{type(obj).__name__} of length {len(obj)}: [{preview}]..."
    elif isinstance(obj, dict):
        preview_items = list(obj.items())[:3]
        preview = ", ".join([f"{k}: {v}" for k, v in preview_items])
        return f"Dict of size {len(obj)}: {{{preview}}}..."
    elif isinstance(obj, str):
        return f"String of length {len(obj)}: {obj[:100]}..."  # Print only first 100 chars
    elif isinstance(obj, bytes):
        # For simplicity, display the first few bytes in hexadecimal
        return f"Bytes of length {len(obj)}: {obj[:10].hex()}..."
    else:
        return str(type(obj))


###### For input

from pynput import keyboard
import threading

# A dictionary to store key-callback mappings
input_events = {}


# Function to set the callback for a specific key
def on_press_key(key, callback):
    global input_events
    input_events[key] = callback
    # print('set callback for', key)


# Function that will be called whenever a key is pressed
def on_press(key):
    try:
        # Attempt to get the character of the key
        char = key.char
    except AttributeError:
        char = None

    # If the character is in our input events, call the associated function
    if char in input_events:
        input_events[char]()


def get_video_fps(video_file):
    """Get the FPS of the video using OpenCV."""
    if not os.path.exists(video_file):
        print(f"File not found: {video_file}")
        return None

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return None

    # Get the FPS property
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()

    if fps > 0:
        return fps
    else:
        print(f"Error: Could not retrieve FPS for {video_file}")
        return None


def reencode_to_30fps(video_file, output_file):
    """Reencode the video at 30 FPS using ffmpeg."""
    try:
        ffmpeg.input(video_file).output(output_file, r=30).run(overwrite_output=True)
        print(f"Reencoded {video_file} to {output_file} at 30 FPS.")
    except ffmpeg.Error as e:
        print(f"Error reencoding {video_file}: {e}")


def check_and_fix_fps(video_file):
    """Check if the FPS is even, and if not, reencode the video at 30 FPS."""
    fps = get_video_fps(video_file)
    if fps is None:
        return False

    if int(fps) % 2 != 0:
        print(f"FPS {fps} of {video_file} is not even. Reencoding to 30 FPS.")
        output_file = os.path.splitext(video_file)[0] + "_30fps.mp4"
        reencode_to_30fps(video_file, output_file)

        # Rename to .mp4 if reencoded to 30 FPS
        os.rename(output_file, os.path.splitext(video_file)[0] + ".mp4")
        return True

    return False


# Start the listener
listener = keyboard.Listener(on_press=on_press)
listener_thread = threading.Thread(target=listener.start)
listener_thread.daemon = True
listener_thread.start()

# set up profiling on demand
import cProfile
import pstats

# import asyncio

profile_when_possible = False
profiler = None


def toggle_profiling():
    global profile_when_possible
    global profiler

    profile_when_possible = True

    if profile_when_possible and not profiler:
        print("ACTIVATE PROFILING")
        profiler = cProfile.Profile()
        profiler.enable()


def print_profiling():
    global profile_when_possible
    global profiler
    if profile_when_possible and profiler:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")  # 'tottime' for total time
        stats.print_stats()
        profiler.enable()
        profile_when_possible = False


on_press_key("¬", toggle_profiling)  # option-i
