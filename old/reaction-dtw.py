import os
import subprocess
import librosa
import numpy as np
from scipy.spatial.distance import euclidean

from dtw import dtw

import matplotlib.pyplot as plt
import glob

import tempfile
from typing import List, Tuple


from utilities import trim_and_concat_video, extract_audio, get_audio_duration


def visualize_dtw_path_and_segments(dtw_path, dtw_path_backwards, base_audio, react_audio, hop_length, segments, base_segments, refined_video_segments=None):
    plt.figure(figsize=(10, 10))
    
    # Load the audio files
    y1, sr1 = librosa.load(base_audio, mono=True, sr=None)
    y2, sr2 = librosa.load(react_audio, mono=True, sr=None)

    # Calculate the time for each audio
    time1 = np.arange(0, len(y1)) / sr1
    time2 = np.arange(0, len(y2)) / sr2
    
    
    # Plot the DTW path
    # plt.subplot(2, 1, 2)
    plt.plot(dtw_path[:, 1] * hop_length / sr2, dtw_path[:, 0] * hop_length / sr1, label="DTW Path", linewidth=2)

    plt.plot(dtw_path_backwards[:, 1] * hop_length / sr2, dtw_path_backwards[:, 0] * hop_length / sr1, label="DTW Path Backwards", linewidth=2, color='purple')


    # plt.plot(dtw_path_smoothed5 * hop_length / sr2, dtw_path[:, 0] * hop_length / sr1, label="DTW Path, smoothed5", linewidth=1)
    # plt.plot(dtw_path_smoothed50 * hop_length / sr2, dtw_path[:, 0] * hop_length / sr1, label="DTW Path, smoothed50", linewidth=1)
    # plt.plot(dtw_path_smoothed100 * hop_length / sr2, dtw_path[:, 0] * hop_length / sr1, label="DTW Path, smoothed100", linewidth=1)

    # Plot the segments

    print(segments)
    print("Base", base_segments)
    print("Refined", refined_video_segments)
    for i, segment in enumerate(segments):
        x1 = segment[0] #* hop_length / sr1
        x2 = segment[1] #* hop_length / sr1

        y1 = base_segments[i][0]
        y2 = base_segments[i][1]

        plt.plot( [x1, x2], [y1, y2]    , color='red', linestyle='solid')
        # plt.hlines(start_time, start_time, end_time, colors='red', linestyles='solid', label='Segment {}'.format(i+1), linewidth=2)

        # start_time = segment[0] #* hop_length / sr1
        # end_time = segment[1] #* hop_length / sr1
        # plt.plot(start, start, colors='red', linestyles='solid', label='Segment {}'.format(i+1), linewidth=1)
    
    plt.xlabel("Time in React Audio [s]")
    plt.ylabel("Time in Base Audio [s]")
    # plt.legend()

    plt.xticks(np.arange(min(time2), max(time2), 1))
    plt.yticks(np.arange(min(time1), max(time1), 1))

    plt.grid(True)

    plt.tight_layout()
    plt.show()


from scipy.stats import zscore

def calculate_dtw(base_audio: str, react_audio: str, hop_length: int, features: list, reversed: bool=False) -> np.array:
    # Load the audio files
    y1, sr1 = librosa.load(base_audio, mono=True, sr=None)
    y2, sr2 = librosa.load(react_audio, mono=True, sr=None)

    if (reversed):
        # Reverse the audios
        y1 = y1[::-1]
        y2 = y2[::-1]


    features1 = []
    features2 = []
    for feature in features:
        func = getattr(librosa.feature, feature)
        features1.append(func(y=y1, sr=sr1, hop_length=hop_length))
        features2.append(func(y=y2, sr=sr2, hop_length=hop_length))

    # # Compute MFCCs
    # mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, hop_length=hop_length)
    # mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, hop_length=hop_length)

    # # # Compute chroma feature
    # chroma1 = librosa.feature.chroma_stft(y=y1, sr=sr1, hop_length=hop_length)
    # chroma2 = librosa.feature.chroma_stft(y=y2, sr=sr2, hop_length=hop_length)

    # # # Compute spectral contrast
    # contrast1 = librosa.feature.spectral_contrast(y=y1, sr=sr1, hop_length=hop_length)
    # contrast2 = librosa.feature.spectral_contrast(y=y2, sr=sr2, hop_length=hop_length)

    # # Compute Tonnetz
    # tonnetz1 = librosa.feature.tonnetz(y=y1, sr=sr1)
    # tonnetz2 = librosa.feature.tonnetz(y=y2, sr=sr2)

    # # Compute Zero-Crossing Rate
    # zcr1 = librosa.feature.zero_crossing_rate(y=y1)
    # zcr2 = librosa.feature.zero_crossing_rate(y=y2)

    # # Compute Spectral Centroid
    # spec_centroid1 = librosa.feature.spectral_centroid(y=y1, sr=sr1)
    # spec_centroid2 = librosa.feature.spectral_centroid(y=y2, sr=sr2)

    # # Compute Spectral Rolloff
    # spec_rolloff1 = librosa.feature.spectral_rolloff(y=y1, sr=sr1)
    # spec_rolloff2 = librosa.feature.spectral_rolloff(y=y2, sr=sr2)

    # Combine the features
    if len(features) > 1:
        combined_features1 = np.concatenate(features1)
        combined_features2 = np.concatenate(features2)
    else:
        combined_features1 = features1[0]
        combined_features2 = features2[0]

    # Normalize the features
    combined_features1 = zscore(combined_features1, axis=1)
    combined_features2 = zscore(combined_features2, axis=1)

    # Check to make sure the number of combined features match
    if combined_features1.shape[0] != combined_features2.shape[0]:
        raise ValueError('The number of combined features in the base and reaction audios do not match')

    alignment = dtw(combined_features1.T, combined_features2.T, keep_internals=True)
    new_path = list(zip(alignment.index1, alignment.index2))

    return np.array(new_path)


def calculate_dtw_forward_and_backward(base_audio, react_audio, hop_length: int, features: list):
    # Calculate forward DTW
    dtw_forward = calculate_dtw(base_audio, react_audio, hop_length, features)


    # Calculate backward DTW
    dtw_backward = calculate_dtw(base_audio, react_audio, hop_length, features, reversed=True)

    # Reverse the backward DTW to align with the time direction of the forward DTW
    dtw_backward = dtw_backward[::-1]

    return dtw_forward, dtw_backward



def find_inflection_points(dtw_path, react_audio, base_audio, hop_length, n_seconds=.5, slope_threshold=.4, adjustment_threshold=.001):
    # Load the audio files
    _, sr_react = librosa.load(react_audio, mono=True, sr=None)
    _, sr_base = librosa.load(base_audio, mono=True, sr=None)

    # Create an unraveled DTW path
    react_time = np.arange(dtw_path[-1, 1] + 1) * hop_length / sr_react
    base_time = np.zeros_like(react_time)
    base_time[dtw_path[:, 1]] = dtw_path[:, 0] * hop_length / sr_base

    m_frames = int(sr_react / hop_length * n_seconds)

    diagonal_segments = []
    base_segments = []
    in_diagonal_segment = False
    diagonal_start = None
    diagonal_base_start = None


    for i in range(len(react_time)):
        # Determine the start and end of the window
        start = max(0, i - m_frames)
        end = min(len(react_time), i + m_frames)

        # Compute the slope of the segment
        x1 = react_time[start]
        x2 = react_time[end - 1]

        y1 = base_time[start]
        y2 = base_time[end - 1]

        slope = (y2 - y1) / (x2 - x1)


        if slope < slope_threshold:
            # If we're in a diagonal segment, this is the end
            if in_diagonal_segment:
                diagonal_end = react_time[i-1]
                diagonal_base_end = base_time[i-1]
                # seek backwards to find true end of the diagonal segment
                ii = i - 1
                current_val = base_time[i]
                while ii >= 0:
                  if abs(current_val - base_time[ii]) > adjustment_threshold:
                    diagonal_end = react_time[ii + 1]
                    diagonal_base_end = base_time[ii + 1]
                    break 
                  ii -= 1
                if diagonal_start < diagonal_end:                  
                    diagonal_segments.append((diagonal_start, diagonal_end))
                    base_segments.append((diagonal_base_start, diagonal_base_end))
                in_diagonal_segment = False
        else:
            # If we're not in a diagonal segment, this is the start
            if not in_diagonal_segment:
                diagonal_start = react_time[i]
                diagonal_base_start = base_time[i]
                # seek forwards to find true start of the diagonal segment
                ii = i + 1
                current_val = base_time[i]
                while ii < len(react_time):
                  if abs(current_val - base_time[ii]) > adjustment_threshold:
                    diagonal_start = react_time[ii - 1]
                    diagonal_base_start = base_time[ii - 1]
                    break 
                  ii += 1
                in_diagonal_segment = True

    # If we end in a diagonal segment, add it to the list
    if in_diagonal_segment:
        diagonal_end = react_time[-1]
        diagonal_base_end = base_time[-1]
        diagonal_segments.append((diagonal_start, diagonal_end))
        base_segments.append((diagonal_base_start, diagonal_base_end))

    return diagonal_segments, base_segments


def find_unmatched_segments(base_segments, base_audio_duration):
    base_segments.sort(key=lambda x: x[0])  # Sort segments by start time

    unmatched_segments = []

    # Add gap from start of audio to first segment, if present
    if base_segments[0][0] > 0:
        unmatched_segments.append((0, base_segments[0][0]))

    # Add gaps between segments
    for i in range(1, len(base_segments)):
        if base_segments[i-1][1] < base_segments[i][0]:  # Check if there's a gap between segments
            unmatched_segments.append((base_segments[i-1][1], base_segments[i][0]))

    # Add gap from last segment to end of audio, if present
    if base_segments[-1][1] < base_audio_duration:
        unmatched_segments.append((base_segments[-1][1], base_audio_duration))

    gaps = 0
    for start, end in unmatched_segments:
        gaps += end - start

    print(f"\nTotal gaps in base audio: {gaps}\n")

    return unmatched_segments



hop_length = 512
def process_directory(song: str, output_dir: str = 'aligned', features: list=['mfcc', 'chroma_stft', 'spectral_contrast', 'tonnetz']):

    directory = os.path.join('Media', song)
    reactions_dir = 'reactions'


    full_output_dir = os.path.join(directory, output_dir)
    if not os.path.exists(full_output_dir):
       # Create a new directory because it does not exist
       os.makedirs(full_output_dir)

    base_video, react_videos = prepare_reactions(directory)

    print("Processing directory", reactions_dir, "Outputting to", output_dir)
    reaction_dir = os.path.join(directory, reactions_dir)


    # Check if base_video is in webm format, and if corresponding mp4 doesn't exist, convert it
    base_video_name, base_video_ext = os.path.splitext(base_video)

    # Extract the base audio and get the sample rate
    song_audio_data, _, base_audio = extract_audio(base_video, directory)


    # Process each reaction video
    for react_video in react_videos:
        # Check if react_video is in webm format, and if corresponding mp4 doesn't exist, convert it
        react_video_name, react_video_ext = os.path.splitext(react_video)

        print(f"\n\nProcessing {directory} {react_video_name}")
        # Create the output video file name

        output_file = os.path.join(full_output_dir, os.path.basename(react_video_name) + "-"+output_dir + react_video_ext)

        if os.path.exists(output_file): # skip if reaction already created
            continue

        # Extract the reaction audio
        react_audio_data, _, react_audio = extract_audio(react_video, reaction_dir)

        # Calculate the DTW alignment
        dtw_indices, dtw_indices_backward = calculate_dtw_forward_and_backward(base_audio, react_audio, hop_length, features)
        
        video_segments, base_segments = find_inflection_points(dtw_indices, react_audio, base_audio, hop_length) 
        print("\nFOUND segments", video_segments)

        print("\nMissing base segments", find_unmatched_segments(base_segments, get_audio_duration(base_audio)))

        # refined_video_segments = refine_inflection_points(video_segments, base_segments, base_audio, react_audio, window_size_seconds=.5)

        visualize_dtw_path_and_segments(dtw_indices, dtw_indices_backward, base_audio, react_audio, hop_length, video_segments, base_segments)


        # Trim and align the reaction video
        # trim_and_concat_video(react_video, refined_video_segments, base_video, output_file, react_video_ext)



if __name__ == '__main__':
    songs = [ "Ren - Genesis" ] # [ "Ren - Fire","Ren - Genesis", "Ren - The Hunger"]

    for song in songs: 
        process_directory(song, "aligned", "crossed_and_warped", features=['mfcc', 'chroma_stft'])


