import os
import subprocess
import librosa
import numpy as np
from scipy.spatial.distance import euclidean

from dtw import dtw


import matplotlib.pyplot as plt
import seaborn as sns



def visualize_mfcc_and_dtw(base_audio: str, react_audio: str, dtw_path: np.array, hop_length: int):
    # Load the audio files
    y1, sr1 = librosa.load(base_audio, mono=True, sr=None)
    y2, sr2 = librosa.load(react_audio, mono=True, sr=None)

    # Compute MFCCs
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, hop_length=hop_length)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, hop_length=hop_length)

    # Transpose the MFCCs to have time on the x-axis (rows)
    mfcc1 = mfcc1.T
    mfcc2 = mfcc2.T

    # Create figure and axes
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the MFCCs
    sns.heatmap(mfcc1.T, ax=ax[0], cmap='viridis')
    ax[0].set_title('MFCCs for Base Audio')
    sns.heatmap(mfcc2.T, ax=ax[1], cmap='viridis')
    ax[1].set_title('MFCCs for Reaction Audio')

    # Create a new figure for the DTW path
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate the difference matrix
    diff_matrix = np.zeros_like(mfcc1)
    for i in range(len(dtw_path)):
        idx1, idx2 = dtw_path[i]
        diff_matrix[idx1] = np.abs(mfcc1[idx1] - mfcc2[idx2])

    # Convert the frame indices in dtw_path to time in seconds
    dtw_path_in_sec = np.zeros_like(dtw_path, dtype=float)
    dtw_path_in_sec[:, 0] = librosa.frames_to_time(dtw_path[:, 0], sr=sr1, hop_length=hop_length)
    dtw_path_in_sec[:, 1] = librosa.frames_to_time(dtw_path[:, 1], sr=sr2, hop_length=hop_length)

    # Plot the DTW path
    # ax.imshow(diff_matrix.T, aspect='auto', cmap='viridis', origin='lower')
    ax.plot(dtw_path_in_sec[:, 0], dtw_path_in_sec[:, 1], color='r')
    ax.set_title('DTW Path')

    plt.show()




# uses cross-correlation
from scipy import signal

def find_beginning(base_audio, react_audio, seconds=2):
    # Assuming y1 and y2 are your audio signals
    y1, sr1 = librosa.load(base_audio, mono=True, sr=None)  # Load base video audio
    y2, sr2 = librosa.load(react_audio, mono=True, sr=None)  # Load reaction video audio

    # Extract first n seconds of base audio
    start_segment = y1[:seconds*sr1]

    # Compute cross-correlation
    xcorr = signal.correlate(y2, start_segment)

    # Find the point of maximum correlation, this should be the start of the music in the reaction video
    start_point = xcorr.argmax()

    # Convert start_point from samples to seconds
    start_point_seconds = start_point / sr2


    print(f'The start point of the music in the reaction video is approximately at {start_point_seconds} seconds.')

    top_n = 10
    # Find the indices of the top N peaks
    top_peak_indices = signal.argrelextrema(xcorr, np.greater)[0]  # get indices of all peaks
    top_peak_indices_sorted = top_peak_indices[np.argsort(xcorr[top_peak_indices])][::-1]  # sort indices by peak height
    top_n_peak_indices = top_peak_indices_sorted[:top_n]  # get top N peak indices

    # Convert the indices to seconds
    start_points_in_seconds = top_n_peak_indices / sr1

    # Print the top N peaks along with their correlation scores
    for i in range(top_n):
        print(f"Peak {i+1}: Score={xcorr[top_n_peak_indices[i]]}, Start time={start_points_in_seconds[i]} seconds")


    return start_point_seconds



def extract_audio(video_file: str, output_dir: str, sample_rate: int = 44100) -> str:
    # Construct the output file path
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.wav")
    
    # If the file has already been extracted, return the existing path
    if os.path.exists(output_file):
        return output_file
    
    # Construct the ffmpeg command
    command = f'ffmpeg -i "{video_file}" -vn -acodec pcm_s16le -ar {sample_rate} -ac 2 "{output_file}"'
    print(command)
    # Execute the command
    subprocess.run(command, shell=True, check=True)
    
    return output_file


def get_sample_rate(audio_file: str) -> int:
    # Load the audio file with librosa
    _, sample_rate = librosa.load(audio_file, sr=None)
    return sample_rate


def get_frame_rate(video_file: str) -> float:
    cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1"
    args = cmd.split(" ") + [video_file]
    ffprobe_output = subprocess.check_output(args).decode('utf-8')
    # The output has the form 'num/den' so we compute the ratio to get the frame rate as a decimal number
    num, den = map(int, ffprobe_output.split('/'))
    return num / den






import tempfile
from typing import List, Tuple


def convert_frames_to_time(start:int, end:int, sample_rate: float, hop_length: int) -> Tuple[float, float]:
    s = librosa.frames_to_time(start, sr=sample_rate, hop_length=hop_length)
    e = librosa.frames_to_time(end, sr=sample_rate, hop_length=hop_length)
    return (s, e)

import numpy as np






def calculate_video_segments(dtw_indices: np.array, frame_rate: float, sample_rate: float, hop_length:int, PAUSE_THRESH: int=100, JUMP_THRESH: int=100, MATCH_THRESH: float=.5) -> List[Tuple[float, float]]:

    last_base = -1
    last_react = -1
    current_start = -1
    times_in_a_row = 0
    last_advanced = -1
    seen = {}

    candidates = []
    current_candidate = []

    for base_index, react_index in dtw_indices:


        if base_index in seen:

            times_in_a_row += 1
            if current_start > -1 and times_in_a_row > PAUSE_THRESH and current_start != last_advanced:
                candidates.append(current_candidate)
                
                print("SEGMENT END!", base_index, react_index, current_start, last_advanced, librosa.frames_to_time(current_start, sr=sample_rate, hop_length=hop_length), librosa.frames_to_time(last_advanced, sr=sample_rate, hop_length=hop_length))

                current_start = -1
                times_in_a_row = 0
                last_advanced = -1
                current_candidate = []

            continue

        times_in_a_row = 0
        last_advanced = react_index
        print("saw", base_index, react_index, librosa.frames_to_time(base_index, sr=sample_rate, hop_length=hop_length), librosa.frames_to_time(react_index, sr=sample_rate, hop_length=hop_length))
        seen[base_index] = True
        if base_index > last_base and current_start == -1:
            current_start = react_index
            current_candidate = []

        current_candidate.append( (base_index, react_index) )

        last_base = base_index
        last_react = react_index

    if current_start > -1 and current_start != last_react: 
        current_candidate.append((current_start, last_react))
        candidates.append(current_candidate)



    segments = []    
    for candidate in candidates: 
        last_react = 0
        last_base = 0 
        idx = 0
        first_react = -1
        first_base = -1

        for base_index, react_index in candidate:
            if first_react == -1:
                first_react = react_index
                first_base = base_index

            if idx == 0:
                last_react = react_index
                last_base = base_index
                idx += 1
                continue 

            if first_react > -1 and (react_index - last_react > JUMP_THRESH or (idx == len(candidate) - 1)):
                # split here
                if last_react != first_react:

                    # filter out segments where the reaction video has advanced far more than the base video, 
                    # or vice versa, which is a sign of some problem. 
                    match_quality = (last_react - first_react) / (last_base - first_base)

                    if True or match_quality > MATCH_THRESH and match_quality < 1 / (1 - MATCH_THRESH):
                        print("PROPOSED SEgEment", first_base, f"({librosa.frames_to_time(first_base, sr=sample_rate, hop_length=hop_length)})", first_react, f"({librosa.frames_to_time(first_react, sr=sample_rate, hop_length=hop_length)})"," => ", last_base, f"({librosa.frames_to_time(last_base, sr=sample_rate, hop_length=hop_length)})", last_react, f"({librosa.frames_to_time(last_react, sr=sample_rate, hop_length=hop_length)})", match_quality, MATCH_THRESH, 1 / MATCH_THRESH)                   
                        segments.append(convert_frames_to_time(first_react, last_react, sample_rate, hop_length))
                first_react = -1

            last_react = react_index
            last_base = base_index
            idx += 1

    print(segments)
    return segments



def trim_and_concat_video(video_file: str, video_segments: List[Tuple[float, float]], output_file: str, ext: str):
    temp_files = []

    for i, segment in enumerate(video_segments):
        if len(segment) > 2:
            if len(segment) > 4:
                start, end, _, _, filler = segment
            else: 
                start, end, filler = segment
        else: 
            start, end = segment
        if end <= start: 
            continue


        temp_file = os.path.join(tempfile.gettempdir(), f"temp_{i}{ext}")
        temp_files.append(temp_file)

        if filler: 
            # Creating a blank video segment
            blank_video = os.path.join(tempfile.gettempdir(), "blank_video.mp4")
            command = f"ffmpeg -t {end - start} -s 1920x1080 -f rawvideo -pix_fmt rgb24 -c:a aac -r 25 -i /dev/zero {temp_file}"
            print(command)
            subprocess.run(command, shell=True, check=True)

            # # Creating a silent audio segment
            # blank_audio = os.path.join(tempfile.gettempdir(), "silent_audio.wav")
            # command = f"ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -t {end - start} {blank_audio}"
            # print(command)
            # subprocess.run(command, shell=True, check=True)

            # #Combining the blank video and silent audio
            # command = f"ffmpeg -i {blank_video} -i {blank_audio} -c:v copy -c:a aac {temp_file}"
            # print(command)
            # subprocess.run(command, shell=True, check=True)

            # os.remove(blank_video)
            # os.remove(blank_audio)

        else:
            command = f'ffmpeg -y -i "{video_file}" -ss {start} -to {end} -c:v libx264 -c:a aac "{temp_file}"'
            print(command)
            subprocess.run(command, shell=True, check=True)


    concat_file = os.path.join(tempfile.gettempdir(), "concat.txt")
    with open(concat_file, 'w') as f:
        for temp_file in temp_files:
            f.write(f"file '{temp_file}'\n")
    
    command = f'ffmpeg -y -f concat -safe 0 -i "{concat_file}" -c copy "{output_file}"'
    print(command)
    subprocess.run(command, shell=True, check=True)

    # Cleanup temp files
    for temp_file in temp_files:
        if os.path.isfile(temp_file):
            os.remove(temp_file)
    if os.path.isfile(concat_file):
        os.remove(concat_file)

    return output_file




# def refine_inflection_points(segments, base_segments, base_audio, react_audio, window_size):
#     refined_segments = []
    
#     react_audio, sr_react = librosa.load(react_audio, mono=True, sr=None)
#     base_audio, sr_base = librosa.load(base_audio, mono=True, sr=None)

#     base_window_size = int(window_size * sr_base)
#     window_size = int(window_size * sr_react)

#     for i, segment in enumerate(segments):
#         start, end = segment
#         base_start = int((base_segments[i][0]) * sr_base)
#         base_end = int((base_segments[i][1]) * sr_base)

#         start *= sr_react
#         end *= sr_react

#         start = int(start)
#         end = int(end)


#         print(start, end, base_start, base_end)

#         # Handle the start of the segment
#         base_window_start  = base_audio[max(0, base_start-base_window_size//2):min(len(base_audio), base_start+base_window_size//2)]
#         react_window_start = react_audio[max(0, start-window_size//2):min(len(react_audio), start+window_size//2)]
#         cross_corr_start = correlate(base_window_start, react_window_start)
#         max_corr_start = np.argmax(cross_corr_start)
#         refined_start = (start - window_size//2 + max_corr_start) / sr_react
        
#         # Handle the end of the segment
#         base_window_end  = base_audio[max(0, base_end-base_window_size//2):min(len(base_audio), base_end+base_window_size//2)]
#         react_window_end = react_audio[max(0, end-window_size//2):min(len(react_audio), end+window_size//2)]
#         cross_corr_end = correlate(base_window_end, react_window_end)
#         max_corr_end = np.argmax(cross_corr_end)
#         refined_end = (end - window_size//2 + max_corr_end) / sr_react
        
#         refined_segments.append((refined_start, refined_end))
    
#     return refined_segments



def refine_inflection_points(segments, base_segments, base_audio_path, react_audio_path, window_size_seconds):
    # Load the audio files and get the sample rates
    y_base, sr_base = librosa.load(base_audio_path, mono=True, sr=None)
    y_react, sr_react = librosa.load(react_audio_path, mono=True, sr=None)
    
    # Convert window_size_seconds to samples for each audio
    window_size_base = int(window_size_seconds * sr_base)
    window_size_react = int(window_size_seconds * sr_react)

    # Initialize list to store refined segments
    refined_segments = []
    
    for i, (start, end) in enumerate(segments):
        # Convert start and end times to samples for each audio
        start_base = int(base_segments[i][0] * sr_base)
        end_base = int(base_segments[i][1] * sr_base)
        start_react = int(start * sr_react)
        end_react = int(end * sr_react)
        
        # Extract the base and react windows around the start and end of the segment
        base_start_window = y_base[max(0, start_base-window_size_base):min(len(y_base), start_base+window_size_base)]
        react_start_window = y_react[max(0, start_react-window_size_react):min(len(y_react), start_react+window_size_react)]
        
        base_end_window = y_base[max(0, end_base-window_size_base):min(len(y_base), end_base+window_size_base)]
        react_end_window = y_react[max(0, end_react-window_size_react):min(len(y_react), end_react+window_size_react)]

        # Compute MFCCs
        mfcc1 = librosa.feature.mfcc(y=base_start_window, sr=sr_base, hop_length=hop_length)
        mfcc2 = librosa.feature.mfcc(y=react_start_window, sr=sr_react, hop_length=hop_length)
        combined_features1 = zscore(mfcc1, axis=1)
        combined_features2 = zscore(mfcc2, axis=1)
        
        
        # Perform DTW on the windows
        alignment_start = dtw(combined_features1.T, combined_features2.T)


        # Compute MFCCs
        mfcc1 = librosa.feature.mfcc(y=base_end_window, sr=sr_base, hop_length=hop_length)
        mfcc2 = librosa.feature.mfcc(y=react_end_window, sr=sr_react, hop_length=hop_length)
        combined_features1 = zscore(mfcc1, axis=1)
        combined_features2 = zscore(mfcc2, axis=1)


        alignment_end = dtw(combined_features1.T, combined_features2.T)

        # Obtain the refined start and end points
        path_start = alignment_start.index2[0]
        path_end = alignment_end.index2[-1]

        # Convert back to time
        refined_start = path_start / sr_react
        refined_end = path_end / sr_react

        # Append the refined segment to the list
        refined_segments.append((refined_start, refined_end))

        print(start, refined_start, end, refined_end)

    return refined_segments





import soundfile as sf

def get_audio_duration(filename: str) -> float:
    # Open the file
    f = sf.SoundFile(filename)

    # Calculate the duration
    duration = len(f) / f.samplerate

    return duration


import glob

hop_length = 512

import os

def process_directory(directory: str, reactions_dir: str = 'reactions', output_dir: str = 'aligned', PAUSE_THRESH: int=100, JUMP_THRESH: int=100, MATCH_THRESH: float=.5, features: list=['mfcc', 'chroma_stft', 'spectral_contrast', 'tonnetz']):

    print("Processing directory", reactions_dir, "Outputting to", output_dir)
    reaction_dir = os.path.join(directory, reactions_dir)

    # Get the base video file
    base_video = (glob.glob(os.path.join(directory, f"{os.path.basename(directory)}.webm")) + glob.glob(os.path.join(directory, f"{os.path.basename(directory)}.mp4")))[0]
    
    # Check if base_video is in webm format, and if corresponding mp4 doesn't exist, convert it
    base_video_name, base_video_ext = os.path.splitext(base_video)
    if base_video_ext == '.webm':
        base_video_mp4 = base_video_name + '.mp4'
        if not os.path.exists(base_video_mp4):
            command = f'ffmpeg -y -i "{base_video}" "{base_video_mp4}"'
            subprocess.run(command, shell=True, check=True)
        base_video = base_video_mp4

    # Extract the base audio and get the sample rate
    base_audio = extract_audio(base_video, directory)
    base_sample_rate = get_sample_rate(base_audio)
    
    # Get the frame rate of the base video
    base_frame_rate = get_frame_rate(base_video)
    
    # Get all reaction video files
    react_videos = glob.glob(os.path.join(reaction_dir, "*.webm")) + glob.glob(os.path.join(reaction_dir, "*.mp4"))
    react_videos = [video for video in react_videos if video != base_video and "-"+output_dir not in video and "-trimmed" not in video]

    full_output_dir = os.path.join(directory, output_dir)
    if not os.path.exists(full_output_dir):
       # Create a new directory because it does not exist
       os.makedirs(full_output_dir)

    # Process each reaction video
    for react_video in react_videos:
        # Check if react_video is in webm format, and if corresponding mp4 doesn't exist, convert it
        react_video_name, react_video_ext = os.path.splitext(react_video)

        # Create the output video file name

        output_file = os.path.join(directory, output_dir, os.path.basename(react_video_name) + "-"+output_dir + react_video_ext)

        if os.path.exists(output_file): # skip if reaction already created
            continue

        if react_video_ext == '.webm':
            react_video_mp4 = react_video_name + '.mp4'
            if not os.path.exists(react_video_mp4):
                command = f'ffmpeg -y -i "{react_video}" -c:v libx264 -c:a aac "{react_video_mp4}"'
                subprocess.run(command, shell=True, check=True)
            react_video = react_video_mp4


        # Extract the reaction audio
        react_audio = extract_audio(react_video, reaction_dir)


        # if "-trimmed" not in react_video:
        #     beginning = find_beginning(base_audio, react_audio, seconds=2)
        #     if beginning > 4: # trim start so that we're closer to the beginning

        #         # Create the output file name
        #         trimmed_reaction = f"{react_video_name}-trimmed{react_video_ext}"

        #         if not os.path.exists(trimmed_reaction):
        #             # Start the trim at the nearest keyframe before start_time
        #             command = f'ffmpeg -y -i "{react_video}" -ss {beginning} -c copy "{trimmed_reaction}"'
        #             subprocess.run(command, shell=True, check=True)

        #         react_video = trimmed_reaction
        #         react_audio = extract_audio(react_video, reaction_dir)

        # Calculate the DTW alignment
        dtw_indices, dtw_indices_backward = calculate_dtw_forward_and_backward(base_audio, react_audio, hop_length, features)

        # Visualize the MFCCs and DTW path
        # visualize_mfcc_and_dtw(base_audio, react_audio, dtw_indices, hop_length)


        # Calculate the video segments
        # video_segments = calculate_video_segments(dtw_indices, base_frame_rate, base_sample_rate, hop_length, PAUSE_THRESH, JUMP_THRESH, MATCH_THRESH)
        
        video_segments, base_segments = find_inflection_points(dtw_indices, react_audio, base_audio, hop_length) 
        print("\nFOUND segments", video_segments)

        print("\nMissing base segments", find_unmatched_segments(base_segments, get_audio_duration(base_audio)))

        # refined_video_segments = refine_inflection_points(video_segments, base_segments, base_audio, react_audio, window_size_seconds=.5)


        visualize_dtw_path_and_segments(dtw_indices, dtw_indices_backward, base_audio, react_audio, hop_length, video_segments, base_segments)


        # Trim and align the reaction video
        trim_and_concat_video(react_video, refined_video_segments, output_file, react_video_ext)





# process_directory("Ren - The Hunger")


# process_directory("Ren - The Hunger", "reactions", "chroma+mfcc", 100, 1000, .1, features=['mfcc', 'chroma_stft'])
# process_directory("Ren - The Hunger", "chroma+mfcc", "chroma+mfcc-2pass", 10, 100, .5, features=['mfcc', 'chroma_stft'])


# process_directory("Ren - The Hunger", "reactions2", "chroma+mfcc", 100, 1000, .1, features=['mfcc', 'chroma_stft'])
# process_directory("Ren - The Hunger", "chroma+mfcc", "chroma+mfcc-2pass", 10, 100, .5, features=['mfcc', 'chroma_stft'])

# process_directory("Ren - The Hunger", "reactions2", "mfcc", 100, 1000, .1, features=['mfcc'])
# process_directory("Ren - The Hunger", "mfcc", "mfcc-2pass", 10, 100, .5, features=['mfcc'])




####################################
# Cross correlation algorithm
####################################




def find_chunk_potential_matches(chunk: List[Tuple[np.ndarray, int, int]], reaction_audio_data: np.ndarray, peak_tolerance: float) -> List[Tuple[int, int, int, int]]:
    base_start = chunk[1]
    base_end = chunk[2]
    start_index = chunk[3] # Start of the window in reaction_audio_data
    end_index = chunk[4] # End of the window in reaction_audio_data

    chunk = chunk[0]

    assert len(chunk) <= end_index - start_index
    assert start_index < end_index

    print("Finding matches for chunk", base_start, base_end, start_index, end_index)

    # Slice the reaction_audio_data to only include data within this window
    windowed_reaction_audio_data = reaction_audio_data[start_index:end_index]

    # Perform cross-correlation on this window
    correlation = correlate(windowed_reaction_audio_data, chunk)

    # Then find peaks as before
    peak_indices, _ = find_peaks(correlation, height=np.max(correlation)*peak_tolerance)

    # Adjust peak indices to be relative to the whole reaction_audio_data
    peak_indices += start_index

    chunk_size = len(chunk)

    # Add potential matches for this base chunk to the list
    chunk_matches = [(index, (index + chunk_size), base_start, base_end) for index in peak_indices]
    return chunk_matches

def find_potential_matches(base_audio_data, base_sample_rate, reaction_audio_data, reaction_sample_rate, chunk_duration: float = 1.0, peak_tolerance: float = 0.9) -> List[List[Tuple[int, int, int, int]]]:


    

    assert base_sample_rate == reaction_sample_rate, "Sample rates must match!"

    chunk_size = int(chunk_duration * base_sample_rate)  # chunk size in samples
    base_length = len(base_audio_data)
    reaction_length = len(reaction_audio_data)

    assert reaction_length >= base_length


    # Split base audio into chunks
    base_chunks = [ (base_audio_data[i:i+chunk_size], i, (i+chunk_size), i, reaction_length - (base_length - (i + chunk_size))) for i in range(0, base_length, chunk_size)]

    # Create a Pool of workers
    with Pool(cpu_count() // 2) as pool:
        func = partial(find_chunk_potential_matches, reaction_audio_data=reaction_audio_data, peak_tolerance=peak_tolerance)
        potential_matches = pool.map(func, base_chunks)

    return potential_matches


def derive_sequences(potential_matches: List[List[Tuple[int, int, int, int]]]) -> List[Tuple[int, int, int, int]]:
    subsequences = []
    last_end = 0.0  # end of last subsequence

    for chunk_matches in potential_matches:
        # Sort the potential matches for this chunk by start time
        sorted_matches = sorted(chunk_matches, key=lambda match: match[0])

        for match in sorted_matches:
            if match[0] >= last_end:
                # This match does not overlap with the previous subsequence, so add it to the list
                subsequences.append(match)
                last_end = match[1]
                break  # move on to the next base chunk

    # Compress continuous subsequences
    compressed_subsequences = []
    current_start, current_end, current_base_start, current_base_end = subsequences[0]

    for start, end, base_start, base_end in subsequences[1:]:
        if start == current_end:
            # This subsequence is continuous with the current one, extend it
            current_end = end
            current_base_end = base_end
        else:
            # This subsequence is not continuous, add the current one to the list and start a new one
            compressed_subsequences.append((current_start, current_end, current_base_start, current_base_end))
            current_start, current_end, current_base_start, current_base_end = start, end, base_start, base_end

    # Add the last subsequence
    compressed_subsequences.append((current_start, current_end, current_base_start, current_base_end))

    return compressed_subsequences



def pad_sequences(compressed_subsequences: List[Tuple[int, int, int, int]], reaction_duration: float, reaction_sample_rate: int, PAD: float) -> List[Tuple[int, int, int, int]]:
    print("")

    print("compressed", compressed_subsequences)
    print("")

    PAD = int(reaction_sample_rate * PAD)

    #######################
    # Post-process compressed_subsequences
    padded_subsequences = []
    if compressed_subsequences:
        # Add padding and make sure they are within the audio duration
        padded_start, padded_end = max(0, compressed_subsequences[0][0] - PAD), min(reaction_duration, compressed_subsequences[0][1] + PAD)

        for start, end, _, _ in compressed_subsequences[1:]:
            if start - padded_end <= 2 * PAD:
                # The next sequence is close enough to be combined with the current one
                padded_end = min(reaction_duration, end + PAD)  # Extend the end of the current sequence
            else:
                # The next sequence is too far, add the current one to the list and start a new one
                padded_subsequences.append((padded_start, padded_end))
                padded_start, padded_end = max(0, start - PAD), min(reaction_duration, end + PAD)

        # Add the last sequence to the list
        padded_subsequences.append((padded_start, padded_end))

    return padded_subsequences




if __name__ == '__main__':

    # process_directory("Ren - The Hunger", "crosses-knox", "crosses-chroma+mfcc", 20, 200, .4, features=['mfcc', 'chroma_stft'])
    # process_directory("Ren - The Hunger", "crosses-knox", "crosses-mfcc", 20, 200, .4, features=['mfcc'])
    # process_directory("Ren - The Hunger", "crosses-knox", "crosses-all", 20, 200, .4, features=['mfcc', 'chroma_stft', 'spectral_contrast', 'tonnetz'])
    # process_directory("Ren - The Hunger", "crosses-knox", "crosses-chroma", 20, 200, .4, features=['chroma_stft'])


    # process_directory("Ren - The Hunger", "crosses", "crosses-mfcc", 20, 200, .4, features=['mfcc'])


