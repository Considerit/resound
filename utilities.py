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



def create_audio_extraction(reaction_audio, base_audio, segments):
    segmented_audio_data = []

    for reaction_start, reaction_end, current_start, current_end, is_filler in segments:
        if is_filler:
            segment = base_audio[current_start:current_end]
        else: 
            segment = reaction_audio[reaction_start:reaction_end]
        segmented_audio_data.append(segment)

    segmented_audio_data = np.concatenate(segmented_audio_data)
    return segmented_audio_data



def trim_and_concat_video(video_file: str, video_segments: List[Tuple[float, float]],  filler_video: str, output_file: str, ext: str, extend_by=0, write_audio=False):

    print(f"Frame rate: {conversion_frame_rate}, Audio rate: {conversion_audio_sample_rate}")
    temp_dir, _ = os.path.splitext(output_file)

    if not os.path.exists(temp_dir):
       # Create a new directory because it does not exist
       os.makedirs(temp_dir)


    if write_audio:
        # Process audio
        temp_audio_files = []    
        for i, segment in enumerate(video_segments):
            start, end, filler_start, filler_end, filler = segment

            if end <= start: 
                continue

            temp_audio_file = os.path.join(temp_dir, f"temp_{i}.wav")
            temp_audio_files.append(temp_audio_file)

            if filler: 
                command = f'ffmpeg -y -i "{filler_video}" -ss {filler_start} -to {filler_end} -vn -ar {conversion_audio_sample_rate} -ac 2 -c:a pcm_s16le "{temp_audio_file}"'
            else:
                command = f'ffmpeg -y -i "{video_file}" -ss {start} -to {end} -vn -ar {conversion_audio_sample_rate} -ac 2 -c:a pcm_s16le "{temp_audio_file}"'

            subprocess.run(command, shell=True, check=True)
            

        audio_output_file = os.path.join(temp_dir, f"{os.path.basename(output_file).split('.')[0]}.wav")
        concat_audio_file = os.path.join(temp_dir, "concat-audio.txt")
        with open(concat_audio_file, 'w') as f:
            for temp_file in temp_audio_files:
                f.write(f"file '{os.path.basename(temp_file)}'\n")
        
        command = f'ffmpeg -y -f concat -safe 0 -i "{concat_audio_file}" -c copy "{audio_output_file}"'
        print(command)
        subprocess.run(command, shell=True, check=True)

        if os.path.isfile(concat_audio_file):
            os.remove(concat_audio_file)

    # process video  target_resolution=(1080, 1920)
    reaction_video = VideoFileClip(video_file)
    width = reaction_video.w
    height = reaction_video.h

    base_video = VideoFileClip(filler_video)
    
    #########################################################
    # Append 15 seconds (or extend_by) of each reaction video
    fill_length = 0
    for (start, end, filler_start, filler_end, filler) in reversed(video_segments):
        if not filler: 
            break
        fill_length += filler_end - filler_start
    ##################

    if end + fill_length + extend_by <= reaction_video.duration:  # Make sure not to exceed the reaction video duration
        video_segments.append((end, end + fill_length + extend_by, filler_end, filler_end + fill_length + extend_by, False))

    clips = []
    for i, segment in enumerate(video_segments):
        start_frame, end_frame, filler_start, filler_end, filler = segment
        if filler: 
            start_frame = filler_start
            end_frame = filler_end
            video = base_video
        else:
            video = reaction_video

        if end_frame <= start_frame: 
            continue

        end_frame = min(end_frame, video.duration)

        subclip = video.subclip(float(start_frame), float(end_frame))
        if filler:
            subclip = subclip.resize(height=height, width=width)
            subclip = subclip.without_audio()

        print(f'\nAdding frames from {start_frame}s to {end_frame}s filler? {filler}\n')
        clips.append(subclip)


    # Concatenate the clips together
    final_clip = concatenate_videoclips(clips)
    final_clip.set_fps(30)

    # Get the duration of each clip
    final_clip_duration = final_clip.duration
    base_video_duration = base_video.duration


    # If final_clip is longer than base_video, trim it
    if final_clip_duration > base_video_duration + extend_by:
        final_clip = final_clip.subclip(0, base_video_duration + extend_by)
        
    # If final_clip is shorter than base_video, pad it with black frames
    elif final_clip_duration < base_video_duration:
        # Create a black clip with the remaining duration

        black_clip = (ColorClip((base_video.size), col=(0,0,0))
                      .set_duration(base_video_duration - final_clip_duration)
                      .set_fps(final_clip.fps)
                      .set_start(final_clip_duration)
                      .without_audio())

        # Combine the final and black clip using CompositeVideoClip
        final_clip = CompositeVideoClip([final_clip, black_clip])

    # Generate the final output video
    final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")


    # Close the video files
    reaction_video.close()
    base_video.close()


    output_video = VideoFileClip(output_file)
    print(f'DONE! Duration={output_video.duration} (compare to {base_video_duration})')


    # Cleanup temp files
    # for temp_file in temp_files:
    #     if os.path.isfile(temp_file):
    #         os.remove(temp_file)

    return output_file




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


