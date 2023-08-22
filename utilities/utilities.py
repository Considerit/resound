import glob, os
import tempfile
from typing import List, Tuple
import soundfile as sf
import numpy as np
import subprocess
import cv2
import json
from silence import get_edge_silence
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip
from decimal import Decimal, getcontext



# def create_reaction_audio_from_path(reaction_audio, base_audio, segments):
#     # Calculate the total length required for the new audio data

#     total_length = 0
#     for reaction_start, reaction_end, current_start, current_end, is_filler in segments:
#         if not is_filler:
#             total_length += reaction_end - reaction_start
#         else:
#             total_length += current_end - current_start
    
#     # Preallocate the array
#     segmented_audio_data = np.empty(total_length, dtype=base_audio.dtype)  # Assuming both audios are of the same dtype
#     insert_at = 0

#     for reaction_start, reaction_end, current_start, current_end, is_filler in segments:
#         if is_filler:
#             segment_len = current_end - current_start
#             segmented_audio_data[insert_at:insert_at+segment_len] = base_audio[current_start:current_end]
#         else: 
#             segment_len = reaction_end - reaction_start
#             segmented_audio_data[insert_at:insert_at+segment_len] = reaction_audio[reaction_start:reaction_end]
        
#         insert_at += segment_len

#     return segmented_audio_data








conversion_frame_rate = 60 
conversion_audio_sample_rate = 44100

def samples_per_frame():
    return int(conversion_audio_sample_rate / conversion_frame_rate)

def universal_frame_rate(): 
    return conversion_frame_rate



def prepare_reactions(song_directory: str):

    base_video_path_webm = os.path.join(song_directory, f"{os.path.basename(song_directory)}.webm")
    base_video_path_mp4 = os.path.join(song_directory, f"{os.path.basename(song_directory)}.mp4")

    #############
    # Make sure to remove silence at the beginning or end of the base video
    if os.path.exists(base_video_path_webm):
        if not os.path.exists(base_video_path_mp4):
            base_video = base_video_path_webm

            # Check if base_video is in webm format, and if corresponding mp4 doesn't exist, convert it
            base_video_name, base_video_ext = os.path.splitext(base_video_path_webm)

            start, end = get_edge_silence(base_video_path_webm)
            if start > conversion_audio_sample_rate / 5 or end > conversion_audio_sample_rate / 5:
                command = f"ffmpeg -i \"{base_video_path_webm}\" -y -vcodec libx264 -vf \"setpts=PTS\" -c:v libx264 -r {conversion_frame_rate} -ar {conversion_audio_sample_rate} -ss {start} -to {end} \"{base_video_path_mp4}\""
                subprocess.run(command, shell=True, check=True)
                os.remove(base_video_path_webm)
        else: 
            os.remove(base_video_path_webm)


    if os.path.exists(base_video_path_mp4):
        base_video_path = base_video_path_mp4
    elif os.path.exists(base_video_path_webm):
        base_video_path = base_video_path_webm
    else:
        print("ERROR! base audio not found")
        raise Exception()

    reaction_dir = os.path.join(song_directory, 'Reactions')

    print("Processing reactions in: ", reaction_dir)

        

    # Get all reaction video files
    webm_videos = glob.glob(os.path.join(reaction_dir, "*.webm"))
    mp4_videos = glob.glob(os.path.join(reaction_dir, "*.mp4"))
    react_videos = webm_videos + mp4_videos


    # # Process each reaction video
    # for react_video in react_videos:

    #     react_video_name, react_video_ext = os.path.splitext(react_video)

    #     react_video_mp4 = react_video_name + '.mp4'
    #     if not os.path.exists(react_video_mp4):


    #         with VideoFileClip(react_video) as clip:
    #             width, height = clip.size


    #         resize_command = ""
    #         if width > 1920 or height > 1080:
    #             # Calculate aspect ratio
    #             aspect_ratio = width / height
    #             if width > height:
    #                 new_width = 1920
    #                 new_height = int(new_width / aspect_ratio)
    #             else:
    #                 new_height = 1080
    #                 new_width = int(new_height * aspect_ratio)
                
    #             resize_command = f"-vf scale={new_width}:{new_height}"


    #         # Generate the ffmpeg command
    #         command = f'ffmpeg -y -i "{react_video}" {resize_command} -c:v libx264 -r {conversion_frame_rate} -ar {conversion_audio_sample_rate} -c:a aac "{react_video_mp4}"'
            
    #         print(command)
    #         subprocess.run(command, shell=True, check=True)

    #         if os.path.exists(react_video):
    #             os.remove(react_video)

    # ################

    # # Get all reaction video files
    # react_videos = glob.glob(os.path.join(reaction_dir, "*.mp4"))
    

    return base_video_path, react_videos





def extract_audio(video_file, output_dir=None, sample_rate=None, preserve_silence=False):
    if output_dir is None:
        output_dir = os.path.dirname(video_file)
    if sample_rate is None:
        sample_rate = conversion_audio_sample_rate

    # Construct the output file path
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.wav")
    
    if not os.path.exists(output_file):
        # Construct the ffmpeg command
        command = f'ffmpeg -i "{video_file}" -vn -acodec pcm_s16le -ar {sample_rate} -ac 2'
        
        # If preserving silence is desired, get the video duration and apply the atrim filter
        if preserve_silence:
            duration_command = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_file}"'
            duration = float(subprocess.check_output(duration_command, shell=True).decode('utf-8').strip())
            pad_duration = int(duration * sample_rate)  # Calculate the number of samples to pad
            command += f" -af apad=whole_len={pad_duration}"
        
        command += f' "{output_file}"'
        print(command)
        # Execute the command
        subprocess.run(command, shell=True, check=True)    

    audio_data, sample_rate = sf.read(output_file)
    if audio_data.ndim > 1:  # convert to mono
        audio_data = np.mean(audio_data, axis=1)

    return (audio_data, sample_rate, output_file)




def get_frame_rate(video_file: str) -> float:
    cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1"
    args = cmd.split(" ") + [video_file]
    ffprobe_output = subprocess.check_output(args).decode('utf-8')
    # The output has the form 'num/den' so we compute the ratio to get the frame rate as a decimal number
    num, den = map(int, ffprobe_output.split('/'))
    return num / den


def compute_precision_recall(output_segments, ground_truth, tolerance=0.5):
    tp = 0  # true positives
    fp = 0  # false positives
    fn = 0  # false negatives

    # Create a list to keep track of which ground truth segments have been matched
    matched = [False]*len(ground_truth)

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
        rel_tol = Decimal('1e-09')
        abs_tol = Decimal('0.0')
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
    elif hasattr(obj, '__dict__') and isinstance(obj.__dict__, dict):
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


