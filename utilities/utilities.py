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



def trim_and_concat_video(video_file: str, video_segments: List[Tuple[float, float]],  filler_video: str, output_file: str, ext: str, extend_by=0, use_fill=True):

    print(f"Frame rate: {conversion_frame_rate}, Audio rate: {conversion_audio_sample_rate}")
    temp_dir, _ = os.path.splitext(output_file)

    if not os.path.exists(temp_dir):
       os.makedirs(temp_dir)

    check_compatibility(filler_video, video_file)

    video_segments = [list(s) for s in video_segments if ((not s[4]) and s[0] < s[1]) or (s[4] and s[2] < s[3] )] # integrity check

    # process video  target_resolution=(1080, 1920)
    reaction_video = VideoFileClip(video_file)
    width = reaction_video.w
    height = reaction_video.h

    base_video = VideoFileClip(filler_video)

    #########################################################
    # Append 15 seconds (or extend_by) of each reaction video
    if extend_by > 0:
        fill_length = 0
        for (start, end, filler_start, filler_end, filler) in reversed(video_segments):
            if not filler: 
                break
            fill_length += filler_end - filler_start

        if end + fill_length + extend_by <= reaction_video.duration:  # Make sure not to exceed the reaction video duration
            video_segments.append((end + fill_length, end + fill_length + extend_by, filler_end, filler_end + fill_length + extend_by, False))
    
    ##################

    clips = []
    for segment in video_segments:

        start_frame, end_frame, filler_start, filler_end, filler = segment

        if (filler and use_fill) or not filler:
            if filler: 
                start_frame = filler_start
                end_frame = filler_end
                video = base_video
            else:
                video = reaction_video

            end_frame = min(end_frame, video.duration)

            subclip = video.subclip(float(start_frame), float(end_frame))
            if filler:
                subclip = subclip.resize(height=height, width=width)
                subclip = subclip.without_audio()
                subclip = subclip.set_fps(reaction_video.fps)
        else: 
            clip_duration = float(filler_end - filler_start)
            subclip = ColorClip(size=(width, height), color=(0,0,0)).set_duration(clip_duration).set_fps(reaction_video.fps)


        print(f'Adding frames from {start_frame}s to {end_frame}s filler? {filler}')
        clips.append(subclip)


    # Concatenate the clips together
    final_clip = concatenate_videoclips(clips)
    final_clip.set_fps(30)

    # Get the duration of each clip
    final_clip_duration = final_clip.duration
    base_video_duration = base_video.duration


    # If final_clip is longer than base_video, trim it
    if final_clip_duration > base_video_duration + extend_by:
        print("...chopping down final clip")
        final_clip = final_clip.subclip(0, base_video_duration + extend_by)
        
    # If final_clip is shorter than base_video, pad it with black frames
    elif final_clip_duration < base_video_duration:
        # Create a black clip with the remaining duration
        print("...adding black clip to end")
        black_clip = (ColorClip((base_video.size), col=(0,0,0))
                      .set_duration(base_video_duration - final_clip_duration)
                      .set_fps(final_clip.fps)
                      .set_start(final_clip_duration)
                      .without_audio())

        # Combine the final and black clip using CompositeVideoClip
        final_clip = CompositeVideoClip([final_clip, black_clip])


    final_clip.resize(height=height, width=width)

    # Generate the final output video
    final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")


    # Close the video files
    reaction_video.close()
    base_video.close()


    output_video = VideoFileClip(output_file)
    print(f'DONE! Duration={output_video.duration} (compare to {base_video_duration})')

    return output_file


import json

def extract_video_info(video_file):
    command = [
        'ffprobe', '-i', video_file, 
        '-hide_banner', 
        '-print_format', 'json', 
        '-show_format', 
        '-show_streams'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    json_start_idx = result.stdout.find('{')
    json_output = result.stdout[json_start_idx:]
    
    try:
        video_info = json.loads(json_output)
    except json.JSONDecodeError:
        print("Error while extracting information from video. FFprobe says:")
        print(result.stdout)
        return None
    
    return video_info

def check_compatibility(video1, video2):
    video1_info = extract_video_info(video1)
    video2_info = extract_video_info(video2)

    if not video1_info or not video2_info:
        return

    video1_video_stream = next(s for s in video1_info['streams'] if s['codec_type'] == 'video')
    video2_video_stream = next(s for s in video2_info['streams'] if s['codec_type'] == 'video')

    video1_audio_stream = next(s for s in video1_info['streams'] if s['codec_type'] == 'audio')
    video2_audio_stream = next(s for s in video2_info['streams'] if s['codec_type'] == 'audio')

    video_attributes_to_check = [
        'codec_name', 'width', 'height', 'pix_fmt', 
        'avg_frame_rate', 'color_range', 'color_space', 
        'color_transfer', 'color_primaries', 'level', 
        'profile', 'sample_aspect_ratio', 'display_aspect_ratio',
        'field_order'
    ]

    audio_attributes_to_check = [
        'codec_name', 'sample_rate', 'channels', 
        'channel_layout', 'sample_fmt'
    ]

    for attribute in video_attributes_to_check:
        v1_attr = video1_video_stream.get(attribute, "Unavailable")
        v2_attr = video2_video_stream.get(attribute, "Unavailable")
        if v1_attr != v2_attr:
            print(f"Different {attribute}: {video1} has {v1_attr}, while {video2} has {v2_attr}")

    if video1_audio_stream and video2_audio_stream:
        for attribute in audio_attributes_to_check:
            v1_attr = video1_audio_stream.get(attribute, "Unavailable")
            v2_attr = video2_audio_stream.get(attribute, "Unavailable")
            if v1_attr != v2_attr:
                print(f"Different {attribute}: {video1} has {v1_attr}, while {video2} has {v2_attr}")
    elif not video1_audio_stream:
        print(f"{video1} does not contain an audio stream.")
    elif not video2_audio_stream:
        print(f"{video2} does not contain an audio stream.")







conversion_frame_rate = 60 
conversion_audio_sample_rate = 44100

def samples_per_frame():
    return int(conversion_audio_sample_rate / conversion_frame_rate)

def universal_frame_rate(): 
    return conversion_frame_rate

def prepare_reactions(song_directory: str):

    base_audio_path_webm = os.path.join(song_directory, f"{os.path.basename(song_directory)}.webm")
    base_audio_path_mp4 = os.path.join(song_directory, f"{os.path.basename(song_directory)}.mp4")

    reaction_dir = os.path.join(song_directory, 'Reactions')

    print("Processing reactions in: ", reaction_dir)

    ############
    # Make sure all webm files have been converted to mp4
    base_video = glob.glob(base_audio_path_webm)
    if len(base_video) > 0: 
        base_video = base_video[0]

        # Check if base_video is in webm format, and if corresponding mp4 doesn't exist, convert it
        base_video_name, base_video_ext = os.path.splitext(base_video)

        base_video_mp4 = base_video_name + '.mp4'
        if not os.path.exists(base_video_mp4):
            start, end = get_edge_silence(base_video)
            command = f"ffmpeg -i \"{base_video}\" -y -vcodec libx264 -vf \"setpts=PTS\" -c:v libx264 -r {conversion_frame_rate} -ar {conversion_audio_sample_rate} -ss {start} -to {end} \"{base_video_mp4}\""
            subprocess.run(command, shell=True, check=True)
        os.remove(base_video)
        

    # Get all reaction video files
    react_videos = glob.glob(os.path.join(reaction_dir, "*.webm"))

    # Process each reaction video
    for react_video in react_videos:

        react_video_name, react_video_ext = os.path.splitext(react_video)

        react_video_mp4 = react_video_name + '.mp4'
        if not os.path.exists(react_video_mp4):


            with VideoFileClip(react_video) as clip:
                width, height = clip.size

            resize_command = ""
            if width > 1920 or height > 1080:
                # Calculate aspect ratio
                aspect_ratio = width / height
                if width > height:
                    new_width = 1920
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = 1080
                    new_width = int(new_height * aspect_ratio)
                
                resize_command = f"-vf scale={new_width}:{new_height}"

            # Generate the ffmpeg command
            command = f'ffmpeg -y -i "{react_video}" {resize_command} -c:v libx264 -r {conversion_frame_rate} -ar {conversion_audio_sample_rate} -c:a aac "{react_video_mp4}"'



            # TODO: if video is also greater than 1920x1080 resolution, resize it while maintaining aspect ratio            
            command = f'ffmpeg -y -i "{react_video}" -c:v libx264 -r {conversion_frame_rate} -ar {conversion_audio_sample_rate} -c:a aac "{react_video_mp4}"'
            subprocess.run(command, shell=True, check=True)
        


        os.remove(react_video)

    ################

    # Get all reaction video files
    react_videos = glob.glob(os.path.join(reaction_dir, "*.mp4"))
    return base_audio_path_mp4, react_videos




def download_and_parse_reactions(song):
    song_directory = os.path.join('Media', song)
    
    if not os.path.exists(song_directory):
       # Create a new directory because it does not exist
       os.makedirs(song_directory)

    manifest_file = os.path.join(song_directory, "manifest.json")
    if not os.path.exists(manifest_file):
        raise f"{manifest_file} does not exist"

    song_data = json.load(open(manifest_file))

    song_file = os.path.join(song_directory, song)
    
    if not os.path.exists(song_file + '.mp4') and not os.path.exists(song_file + '.webm'):
        v_id = song_data["main_song"]["id"]

        cmd = f"yt-dlp -o \"{song_file + '.webm'}\" https://www.youtube.com/watch\?v\={v_id}\;"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    else: 
        print(f"{song_file} exists")


    full_reactions_path = os.path.join(song_directory, 'reactions')
    if not os.path.exists(full_reactions_path):
       # Create a new directory because it does not exist
       os.makedirs(full_reactions_path)

    for _, reaction in song_data["reactions"].items():
        if reaction.get("download"):
            v_id = reaction["id"]
            output = os.path.join(full_reactions_path, reaction['reactor'] + '.webm')
            extracted_output = os.path.join(full_reactions_path, reaction['reactor'] + '.mp4')

            if not os.path.exists(output) and not os.path.exists(extracted_output) and not os.path.exists(os.path.join(full_reactions_path, 'tofix', reaction['reactor'] + '.mp4')):
                cmd = f"yt-dlp -o \"{output}\" https://www.youtube.com/watch\?v\={v_id}\;"
                print(cmd)
                subprocess.run(cmd, shell=True, check=True)


    prepare_reactions(song)



def extract_audio(video_file: str, output_dir: str = None, sample_rate: int = 44100) -> list:
    if output_dir is None:
        output_dir = os.path.dirname(video_file)

    # Construct the output file path
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.wav")
    
    # If the file has already been extracted, return the existing path
    if not os.path.exists(output_file):
        # Construct the ffmpeg command
        command = f'ffmpeg -i "{video_file}" -vn -acodec pcm_s16le -ar {sample_rate} -ac 2 "{output_file}"'
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


