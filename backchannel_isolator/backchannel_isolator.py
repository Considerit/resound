import cv2
import numpy as np
import subprocess
import os
import scipy.signal
import librosa
import soundfile as sf

import matplotlib.pyplot as plt
from scipy import signal

from backchannel_isolator.track_separation import separate_vocals

from utilities.audio_processing import audio_percentile_loudness
from utilities import conversion_audio_sample_rate as sr
from utilities import conf

# I have two audio files. One file is a song. The other file is a reaction — a recording of 
# one or two people reacting to the song. The file contains a possibly distorted, though 
# generally aligned, playback of the song, as well as reactor(s) commenting on the song. 
# I would like you to write me a python function that tries to separate the song from the 
# commentary in the reaction audio. If it is too much to do in one pass, please suggest 
# a series of prompts I can use to arrive at the function. 

# One thing we have that typical approaches do not have access to is the base audio of the song 
# that we want to separate out. That’s a very valuable piece of data that source separation 
# doesn’t leverage. Neither does your code. Are there any techniques you can come up with 
# that can use the song’s audio to help in the separation task? 

# Because the reactors are re-encoding the audio (including the sound), as well as having a lot of 
# variation, there are pitch, volume, and other differences in the reaction audio’s version 
# of the song. Can you augment your code with a best-effort attempt to normalize some of these 
# differences between the song and reaction audio?

import cProfile
import pstats


profile_isolator = False
if profile_isolator:
    profiler = cProfile.Profile()



def construct_mask(diff, threshold):
    # Create a mask where the absolute difference is greater than the threshold
    mask = diff > threshold

    # Create a mask where the difference is greater than zero
    dilated_mask = diff > threshold


    # dilated_mask should be an expansion of mask. It should be True for all segments of positive values of diff 
    # that contain at least one sample where diff > threshold.
    i = 0
    while i < len(mask):
        if mask[i] and diff[i] > 1:
            # find start of positive segment
            start = i
            while i < len(mask) and diff[i] > threshold / 2:
                i += 1
            # mark segment in dilated_mask
            dilated_mask[start:i] = True
        else:
            dilated_mask[i] = False
            i += 1
    return dilated_mask

def create_mask_by_relative_perceptual_loudness_difference(song, reaction, threshold, silence_threshold=.001, plot=False, window=1000, std_dev=None):

    print('song percentile loudness')
    song_percentile_loudness = audio_percentile_loudness(song, loudness_window_size=100, percentile_window_size=window, std_dev_percentile=std_dev)
    print('reaction percentile loudness')
    reaction_percentile_loudness = audio_percentile_loudness(reaction, loudness_window_size=100, percentile_window_size=window, std_dev_percentile=std_dev)

    # Calculate the absolute difference between the percentile loudnesses
    diff = reaction_percentile_loudness - song_percentile_loudness
    # print("diff shape: ", diff.shape)

    avg_diff = np.average(diff)

    print(f"Average diff={avg_diff}")
    diff = diff - avg_diff

    dilated_mask = construct_mask(diff, threshold)



    if plot: 
        # Create a time array for plotting
        time_song = np.arange(song_percentile_loudness.shape[0]) / sr
        time_reaction = np.arange(reaction_percentile_loudness.shape[0]) / sr

        # Plot the percentile loudnesses
        plt.figure(figsize=(12, 8))

        # plt.subplot(3, 1, 1)
        # plt.plot(time_song, song_percentile_loudness, label='Song')
        # plt.plot(time_reaction, reaction_percentile_loudness, label='Reaction')
        # plt.legend()
        # plt.xlabel('Time (s)')
        # plt.ylabel('Percentile Loudness')
        # plt.title('Percentile Loudness of Song and Reaction')

        # Plot the absolute difference
        plt.subplot(2, 1, 1)
        plt.plot(time_reaction, diff, label='Absolute Difference')
        # plt.ylim(bottom=0)  # Constrain y-axis to non-negative values    
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Difference')
        plt.title('Absolute Difference in Percentile Loudness')

        # Plot the masks
        plt.subplot(2, 1, 2)
        plt.plot(time_reaction, dilated_mask, label='Dilated Mask')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Mask Values')
        plt.title('Masks of Commentator Segments')


        plt.tight_layout()
        plt.show()



    return dilated_mask, diff, song_percentile_loudness, reaction_percentile_loudness


# def create_mask_by_relative_pitch_difference(song, reaction, threshold, plot=False):
#     # Calculate the pitch of each audio
#     print("\tsong pitch")
#     song_pitch = calculate_pitch_level(song)
#     print("\treaction pitch")
#     reaction_pitch = calculate_pitch_level(reaction, sr)

#     # Calculate the percentile pitch of each audio
#     print("\tpercentile song pitch")
#     song_percentile_pitch = calculate_percentile_pitch(song_pitch)
#     print("\tpercentile reaction pitch")
#     reaction_percentile_pitch = calculate_percentile_pitch(reaction_pitch)

#     # Calculate the absolute difference between the percentile pitches
#     diff = song_percentile_pitch - reaction_percentile_pitch

#     dilated_mask = construct_mask(diff, threshold)


#     return dilated_mask, diff, song_percentile_pitch, reaction_percentile_pitch




def process_mask(mask, min_segment_length):
    # Number of frames corresponding to the minimum segment length
    min_segment_frames = int(min_segment_length * sr / conf.get('hop_length'))  

    # Initialize a list to hold the segments
    segments = []

    # Initialize variables to hold the current segment
    current_segment_start = None
    current_segment_length = 0

    if len(mask.shape) > 1: 
        mask = mask[0]

    # Iterate over the mask
    # for i in range(mask.shape[1]):
    for i,val in enumerate(mask): 
        # If the mask is True, add the frame to the current segment
        # if mask[0, i]:
        if val:
            if current_segment_start is None:
                current_segment_start = i
            current_segment_length += 1
        # If the mask is False and there's a current segment, end the current segment
        elif current_segment_start is not None:
            # If the current segment is long enough, add it to the list of segments
            if current_segment_length > min_segment_frames:
                segments.append([current_segment_start, current_segment_start + current_segment_length])
            # Reset the current segment
            current_segment_start = None
            current_segment_length = 0

    # If there's a current segment at the end of the mask, add it to the list of segments
    if current_segment_start:
        segments.append([current_segment_start, current_segment_start + current_segment_length])

    return segments

def merge_segments(segments, min_segment_length, max_gap_length):
    if len(segments) == 0:
        return []

    # Number of frames corresponding to the minimum segment length
    min_segment_frames = int(min_segment_length * sr / conf.get('hop_length'))

    # Number of frames corresponding to the maximum gap length
    max_gap_frames = int(max_gap_length * sr / conf.get('hop_length')) 

    # Initialize a list to hold the merged segments
    merged_segments = []

    # Start with the first segment
    current_segment = segments[0]

    # Iterate over the rest of the segments
    for next_segment in segments[1:]:
        # If the gap between the current segment and the next segment is smaller than the maximum, merge them
        if next_segment[0] - current_segment[-1] <= max_gap_frames:
            current_segment[1] = next_segment[1]
        # Otherwise, add the current segment to the list of merged segments and start a new current segment
        else:
            if current_segment[1] - current_segment[0] >= min_segment_frames:
                merged_segments.append(current_segment)
            current_segment = next_segment

    # Add the last current segment to the list of merged segments
    if current_segment[1] - current_segment[0] >= min_segment_frames:
        merged_segments.append(current_segment)


    return merged_segments

def pad_segments(segments, length, pad_beginning=0.1, pad_ending=0.1):
    # Convert padding length from seconds to samples
    pad_length_beginning = int(pad_beginning * sr)
    pad_length_end = int(pad_ending * sr)
        
    padded_segments = []
    # For each segment
    for segment in segments:
        # Pad the start and end of the segment
        start = max(0, segment[0] - pad_length_beginning)
        end = min(length, segment[1] + pad_length_end)
        padded_segments.append([start, end])

    
    return padded_segments

def apply_segments(audio, segments):
    # Initialize a new audio array with zeros
    suppressed_audio = np.zeros_like(audio)

    # For each segment, copy the corresponding audio from the original reaction
    for (start, end) in segments:
        suppressed_audio[start:end] = audio[start:end]

    return suppressed_audio



def mute_by_deviation(song_path, reaction_path, output_path, original_reaction):

    if profile_isolator:
        global profiler
        profiler.enable()


    min_segment_length = 0.01
    max_gap_frames = 0.5
    percentile_thresh = 90
    percent_volume_diff_thresh = 8

    # Load the song and reaction audio
    song, __ = librosa.load(song_path, sr=sr, mono=True)
    reaction, __ = librosa.load(reaction_path, sr=sr, mono=True)



    # assert sr_reaction == sr_song, f"Sample rates must be equal {sr_reaction} {sr_song}"
    # assert sr == sr_reaction, f"Sample rate must be equal to global value"


    # if mono:
    #     # Convert them to pseudo multi-channel (still mono but with an extra dimension to look like multi-channel)
    #     song = np.expand_dims(song, axis=0)
    #     reaction = np.expand_dims(reaction, axis=0)

    # Check the standard deviation of the audio data
    # assert song.shape == reaction.shape, f"Song and reaction must have the same number of channels (song = {song.shape}, reaction = {reaction.shape})"

    delay = 0
    extra_reaction = None # often a result of extend by
    if len(song) > len(reaction):
        song = song[:len(reaction)]
    else:
        original_reaction, __ = librosa.load(original_reaction, sr=sr, mono=True)
        extra_reaction = original_reaction[len(song):] # extract this from the original reaction audio, not the source separated content
        reaction = reaction[:len(song)]



        # pad_width = len(reaction) - len(song) - delay
        # song = np.pad(song, (delay, pad_width))


    # print("calculating short volume diff")
    # mask1, diff1 = create_mask_by_relative_perceptual_loudness_difference(song, reaction, percent_volume_diff_thresh)

    # print("calculating short volume diff")
    # mask12, diff12 = create_mask_by_relative_perceptual_loudness_difference(song, reaction, percent_volume_diff_thresh, std_dev=1000/4)

    print("calculating long volume diff")
    percep_mask = create_mask_by_relative_perceptual_loudness_difference(song, reaction, percent_volume_diff_thresh, window=1 * sr, plot=False)
    long_mask1, long_diff1, song_percentile_pitch, reaction_percentile_pitch = percep_mask
    
    # print("calculating long volume diff2")
    # percep_mask = create_mask_by_relative_perceptual_loudness_difference(song, reaction, percent_volume_diff_thresh, window=sr / 2)
    # long_mask12, long_diff12, song_percentile_pitch, reaction_percentile_pitch = percep_mask

    # print("calculating long volume diff")
    # percep_mask = create_mask_by_relative_perceptual_loudness_difference(song, reaction, percent_volume_diff_thresh, window=4 * sr, plot=False)
    # long_mask2, long_diff2, song_percentile_pitch, reaction_percentile_pitch = percep_mask



    # print('calculating pitch diff')
    # mask2, diff2, song_percentile_pitch, reaction_percentile_pitch = create_mask_by_relative_pitch_difference(song, reaction, threshold=50, plot=False)


    # print('calculating diffs and masks')

    # combined_diff = diff1 * diff2
    # combined_diff = 100 * combined_diff / np.max(combined_diff)

    # mask = combined_diff > 15
    # dilated_mask = combined_diff > 15

    # confirmed_diff = diff1
    # confirmed_mask = (diff1 > percent_volume_diff_thresh) & (diff2 > 5)
    # confirmed_dilated_mask = (diff1 > percent_volume_diff_thresh) & (diff2 > 5)

    # long_confirmed_diff = (diff1 + long_diff1 + diff12 + long_diff12) / 4
    # long_confirmed_mask = long_confirmed_diff > percent_volume_diff_thresh
    # long_confirmed_dilated_mask = long_confirmed_diff > percent_volume_diff_thresh

    # print('dialating masks')

    # # dilated_mask should be an expansion of mask. It should be True for all segments of positive values of diff 
    # # that contain at least one sample where diff > threshold.
    # i = 0
    # while i < len(mask):
    #     if mask[i] and combined_diff[i] > 1:
    #         # find start of positive segment
    #         start = i
    #         while i < len(mask) and combined_diff[i] > 1:
    #             i += 1
    #         # mark segment in dilated_mask
    #         dilated_mask[start:i] = True
    #     else:
    #         dilated_mask[i] = False
    #         i += 1

    # print('\tdialated volume')

    # # dilated_mask should be an expansion of mask. It should be True for all segments of positive values of diff 
    # # that contain at least one sample where diff > threshold.
    # i = 0
    # while i < len(confirmed_mask):
    #     if confirmed_mask[i] and confirmed_diff[i] > 1:
    #         # find start of positive segment
    #         start = i
    #         while i < len(confirmed_mask) and confirmed_diff[i] > 1:
    #             i += 1
    #         # mark segment in dilated_mask
    #         confirmed_dilated_mask[start:i] = True
    #     else:
    #         confirmed_dilated_mask[i] = False
    #         i += 1

    # print('\tdialated pitch+volume')

    # # dilated_mask should be an expansion of mask. It should be True for all segments of positive values of diff 
    # # that contain at least one sample where diff > threshold.
    # i = 0
    # while i < len(long_mask1):
    #     print(f"\ti={i} {len(long_mask1)}", end='\r')
    #     if long_mask1[i] and long_confirmed_diff[i] > 1:
    #         # find start of positive segment
    #         start = i
    #         while i < len(long_mask1) and long_confirmed_diff[i] > 1:
    #             i += 1
    #             print(f"\ti={i} {len(long_mask1)}", end='\r')
    #         # mark segment in dilated_mask
    #         long_confirmed_dilated_mask[start:i] = True
    #     else:
    #         long_confirmed_dilated_mask[i] = False
    #         i += 1

    # print('\tdialated long_vol+volume')


    if False: 
        print('plotting')

        # Create a time array for plotting
        time_song = np.arange(song_percentile_pitch.shape[0]) / sr
        time_reaction = np.arange(reaction_percentile_pitch.shape[0]) / sr

        # Plot the percentile pitches
        plt.figure(figsize=(12, 10))  # Increase figure size for 6 subplots

        plots = 6 
        # # Plot the absolute difference for volume
        # plt.subplot(plots, 2, 1)
        # plt.plot(time_reaction, diff1, label='Absolute Difference, volume')
        # plt.legend()
        # plt.xlabel('Time (s)')
        # plt.ylabel('Absolute Difference')
        # plt.title('Absolute Difference in Percentile Volume')

        # # Plot the volume mask
        # plt.subplot(plots, 2, 2)
        # plt.plot(time_reaction, mask1, label='Dilated Mask, volume')
        # plt.legend()
        # plt.xlabel('Time (s)')
        # plt.ylabel('Mask Values')
        # plt.title('Masks of Commentator Segments (Volume)')

        # # Plot the absolute difference for pitch
        # plt.subplot(plots, 2, 3)
        # plt.plot(time_reaction, diff2, label='Absolute Difference, pitch')
        # plt.legend()
        # plt.ylabel('Absolute Difference')

        # # Plot the pitch mask
        # plt.subplot(plots, 2, 4)
        # plt.plot(time_reaction, mask2, label='Dilated Mask, pitch')
        # plt.legend()
        # plt.ylabel('Mask Values')

        # Plot the absolute difference for joint
        plt.subplot(plots, 2, 1)
        plt.plot(time_reaction, long_diff12, label='Absolute Difference, long vol 12')
        plt.legend()
        plt.ylabel('Absolute Difference')

        # Plot the joint mask
        plt.subplot(plots, 2, 2)
        plt.plot(time_reaction, long_mask12, label='Dilated Mask, long vol 12')
        plt.legend()
        plt.ylabel('Mask Values')

        # Plot the absolute difference for joint
        plt.subplot(plots, 2, 3)
        plt.plot(time_reaction, long_diff1, label='Absolute Difference, long vol 1')
        plt.legend()
        plt.ylabel('Absolute Difference')

        # Plot the joint mask
        plt.subplot(plots, 2, 4)
        plt.plot(time_reaction, long_mask1, label='Dilated Mask, long vol 1')
        plt.legend()
        plt.ylabel('Mask Values')

        # Plot the absolute difference for joint
        plt.subplot(plots, 2, 5)
        plt.plot(time_reaction, long_diff2, label='Absolute Difference, long vol 2')
        plt.legend()
        plt.ylabel('Absolute Difference')

        # Plot the joint mask
        plt.subplot(plots, 2, 6)
        plt.plot(time_reaction, long_mask2, label='Dilated Mask, long vol 2')
        plt.legend()
        plt.ylabel('Mask Values')

        # # # Plot the absolute difference for joint
        # # plt.subplot(plots, 2, 5)
        # # plt.plot(time_reaction, combined_diff, label='Absolute Difference, joint')
        # # plt.legend()
        # # plt.ylabel('Absolute Difference')

        # # # Plot the joint mask
        # # plt.subplot(plots, 2, 6)
        # # plt.plot(time_reaction, dilated_mask, label='Dilated Mask, joint')
        # # plt.legend()
        # # plt.ylabel('Mask Values')


        # # Plot the absolute difference for joint
        # plt.subplot(plots, 2, 7)
        # plt.plot(time_reaction, confirmed_diff, label='Absolute Difference, confirmed')
        # plt.legend()
        # plt.ylabel('Absolute Difference')

        # # Plot the joint mask
        # plt.subplot(plots, 2, 8)
        # plt.plot(time_reaction, confirmed_dilated_mask, label='Dilated Mask, confirmed')
        # plt.legend()
        # plt.ylabel('Mask Values')


        # # Plot the absolute difference for joint
        # plt.subplot(plots, 2, 9)
        # plt.plot(time_reaction, long_confirmed_diff, label='Absolute Difference, long confirmed')
        # plt.legend()
        # plt.ylabel('Absolute Difference')

        # # Plot the joint mask
        # plt.subplot(plots, 2, 10)
        # plt.plot(time_reaction, long_confirmed_dilated_mask, label='Dilated Mask, long confirmed')
        # plt.legend()
        # plt.ylabel('Mask Values')



        plt.tight_layout()
        plt.show()


    # mask = confirmed_dilated_mask
    mask = long_mask1

    segments = process_mask(mask, min_segment_length / 1000)

    padded_segments = pad_segments(segments, len(reaction), pad_beginning=0.5, pad_ending=0.1)
    merged_segments = merge_segments(padded_segments, min_segment_length, max_gap_frames)

    suppressed_reaction = apply_segments(reaction, merged_segments)


    if extra_reaction is not None:

        suppressed_reaction = np.concatenate((suppressed_reaction, extra_reaction))

    # if mono:
    #     # Squeeze the 2D arrays to 1D
    #     suppressed_reaction = np.squeeze(suppressed_reaction)

    
    sf.write(output_path, suppressed_reaction.T, sr)

    # suppressed_reaction = post_process_audio(suppressed_reaction, original_sr_reaction)
    # # Now you can save suppressed_reaction into a file
    # output_path = os.path.splitext(reaction_path)[0] + "_commentary.wav"
    # sf.write(output_path, suppressed_reaction.T, original_sr_reaction)  # Transpose the output because soundfile expects shape (n_samples, n_channels)

    if profile_isolator:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
        stats.print_stats()
        profiler.enable()

    return output_path




def process_reactor_audio(reaction, extended_by=0):
    output_dir = conf.get('temp_directory')
    reaction_audio = reaction.get('aligned_audio_path')
    base_audio = conf.get('base_audio_path')

    reaction_vocals_path, song_vocals_path = separate_vocals(output_dir, base_audio, reaction_audio, post_process=True)
    output_path = os.path.splitext(reaction_vocals_path)[0] + "_isolated_commentary.wav"

    if not os.path.exists(output_path):

        if extended_by > 0:
            reaction_audio_name, ext = os.path.splitext(reaction_audio)
            new_reaction_audio = f"{reaction_audio_name}-truncated{ext}"
            full_reaction_audio, _ = librosa.load(reaction_audio, sr=sr)
            extended_audio = full_reaction_audio[-sr * extended_by:]  # Last extended_by seconds of audio
            truncated_reaction_audio = full_reaction_audio[:-sr * extended_by]  # Except for last extended_by seconds
            sf.write(new_reaction_audio, truncated_reaction_audio.T, sr)  # Write truncated_reaction_audio to a new file
            reaction_audio = new_reaction_audio

        print(f"Separating commentary from {reaction_audio} to {output_path}")
        mute_by_deviation(song_vocals_path, reaction_vocals_path, output_path, reaction_audio)

        if extended_by > 0:
            output_audio, _ = librosa.load(output_path, sr=sr)
            new_output_audio = np.concatenate((output_audio, extended_audio))  # Append truncated_reaction_audio to the end
            sf.write(output_path, new_output_audio.T, sr)  # Write new_output_audio to a new file

    return output_path


