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

from utilities.audio_processing import audio_percentile_loudness, convert_to_mono
from utilities import conversion_audio_sample_rate as sr
from utilities import conf

# I have two audio tracks. One track is a song. The other track is a reaction — a recording of 
# one or two people reacting to the song. The reaction track contains a possibly distorted, though 
# fairly nicely aligned, playback of the song, as well as reactor(s) commenting occassionally on top 
# of the song. Because the reactors are re-encoding the audio (including the sound), as well 
# as having a lot of variation, there are pitch, volume, and other differences in the reaction 
# audio’s version of the song. 

# The only thing I really care about in the reaction audio are the sounds the reactor makes while the 
# base song is playing. This is the backchannel -- the sounds that a listener makes without taking the 
# speaking baton (laughs, signs, shouts, brief comments, etc).

# Because I'm compiling these reaction tracks together, I have a way to isolate the backchannel sounds 
# in the reaction audio from the audio of the song itself in the reaction audio. Subtracting out the 
# base audio from the aligned reaction didn't work, but then I realized I could simply try to detect 
# the portions of the reaction audio where the reactor was making noise, without trying to eliminate 
# the base song audio. Reactors don't make noise that often relative to the length of the song, so 
# I could just mute all the segments of the reaction audio where we didn't detect an active backchannel. 
# Then when we stitch all the reaction together, only a few reactors will tend to be active at the 
# same time. This has proven tractible. 

# I invented a new measure to help find active backchannels: Relative Volume Difference. Consider an audio track. 
# Calculate the max perceptual volume in the track. For each audio sample in the track calculate the percentage 
# of the max observed volume, and store it in a vector. Now we have a normalized measure to compare against other 
# tracks! If we compare the percentile volume vectors for the base audio track and an aligned reaction track 
# where the reactor doesn't make any noise, the vectors should be about the same. If we subtract one from the 
# other, the resulting vector should be close to zero throughout. However, because volume is a measure of energy, 
# a reactor adds energy to the audio signal whenever they make noise. The insight is thus that the backchannel 
# is active when the percentile volume of the reaction audio is significantly higher than the percentile 
# volume of the base song audio. So we can find the backchannel by subtracting the percentile volume vector of 
# the song audio from the percentile volume vector of the reaction audio, and declaring the back channel 
# active when the value is above some threshold (with some smoothing as well of course).

# This all works fairly well, but there are still significant false positives and negatives. 
# I'd like to improve on these issues. Before we delve into those issues though, I'd like to ask you about
# you:
# (1) What do you think of this approach? What would you change or add? 
# (2) How would you approach the issue of backchannel isolation given a reaction track and song track? 
# (3) Has relative volume difference already been invented, and if so, by what name does it go by? 
# (4) Are there other metrics that could also leverage the relationship between the reaction and 
#     song track, and serve as confirming metrics? 



# In particular, I'd like to 
# address false positives in the identification of a meaningful backchannel activation. 




import cProfile
import pstats


profile_isolator = False
if profile_isolator:
    profiler = cProfile.Profile()


def plot_masks(song_percentile_loudness, reaction_percentile_loudness, diff, mask):
    print('plotting')

    # Create a time array for plotting
    time_song = np.arange(song_percentile_loudness.shape[0]) / sr
    time_reaction = np.arange(reaction_percentile_loudness.shape[0]) / sr

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


    # Plot the absolute difference for joint
    plt.subplot(plots, 2, 3)
    plt.plot(time_reaction, diff, label='Absolute Difference, long vol 1')
    plt.legend()
    plt.ylabel('Absolute Difference')

    # Plot the joint mask
    plt.subplot(plots, 2, 4)
    plt.plot(time_reaction, mask, label='Dilated Mask, long vol 1')
    plt.legend()
    plt.ylabel('Mask Values')

    plt.tight_layout()
    plt.show()    



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

def apply_segments(audio, segments, audibility_threshold=0.01, suppresion_period=2): 
    # Initialize a new audio array with zeros
    suppressed_audio = np.zeros_like(audio)
    suppresion_period *= sr

    # For each segment, check if it contains an audible sample and if so, copy it
    for (start, end) in segments:
        segment_audio = audio[start:end]
        
        # Find the indices where the audio is above the threshold
        audible_indices = np.where(np.abs(segment_audio) > audibility_threshold)[0]
        
        # Check for inaudible stretches of more than 1 second
        inaudible_stretches = np.diff(audible_indices) > suppresion_period
        
        # If there are any inaudible stretches of more than 1 second
        if np.any(inaudible_stretches):
            # Start copying from the beginning of the segment
            chunk_start = 0
            
            # Iterate through the inaudible stretches
            for i in np.where(inaudible_stretches)[0]:
                chunk_end = audible_indices[i]
                suppressed_audio[start + chunk_start:start + chunk_end] = segment_audio[chunk_start:chunk_end]
                print(f"Selected segment: {(start + chunk_start) / sr} to {(start + chunk_end) / sr}")
                
                # Skip 1 second after the inaudible stretch
                chunk_start = audible_indices[i+1]

            # Handle the segment after the last inaudible stretch
            chunk_end = len(segment_audio)
            suppressed_audio[start + chunk_start:start + chunk_end] = segment_audio[chunk_start:chunk_end]
            print(f"Selected segment: {(start + chunk_start) / sr} to {(start + chunk_end) / sr}")

        else:
            # If there are no inaudible stretches of more than 1 second, just copy the entire segment
            suppressed_audio[start:end] = segment_audio
            print(f"Selected entire segment from {start / sr} to {end / sr}")

    return suppressed_audio



def mute_by_deviation(reaction, song_path, reaction_path, output_path):

    if profile_isolator:
        global profiler
        profiler.enable()


    min_segment_length = 0.01
    max_gap_frames = 0.5
    percent_volume_diff_thresh = 5

    audio_data, __ = sf.read(song_path)
    song = convert_to_mono(audio_data)

    audio_data, __ = sf.read(reaction_path)
    reaction_data = convert_to_mono(audio_data)

    delay = 0
    extra_reaction = None # often a result of extend by
    if len(song) > len(reaction_data):
        song = song[:len(reaction_data)]
    else:
        conf.get('load_aligned_reaction_data')(reaction.get('channel'))

        original_reaction = reaction.get('aligned_reaction_data')

        extra_reaction = original_reaction[len(song):] # extract this from the original reaction audio, not the source separated content
        reaction_data = reaction_data[:len(song)]

    # print("calculating short volume diff")
    # mask1, diff1 = create_mask_by_relative_perceptual_loudness_difference(song, reaction_data, percent_volume_diff_thresh)

    print("calculating long volume diff")
    percep_mask = create_mask_by_relative_perceptual_loudness_difference(song, reaction_data, percent_volume_diff_thresh, window=1 * sr, plot=False)
    long_mask1, long_diff1, song_percentile_loudness, reaction_percentile_loudness = percep_mask
    

    # print('calculating diffs and masks')

    # long_confirmed_diff = (diff1 + long_diff1) / 4
    # long_confirmed_mask = long_confirmed_diff > percent_volume_diff_thresh
    # long_confirmed_dilated_mask = long_confirmed_diff > percent_volume_diff_thresh


    if False: 
        plot_masks(song_percentile_loudness, reaction_percentile_loudness, long_diff1, long_mask1)



    mask = long_mask1

    segments = process_mask(mask, min_segment_length / 1000)

    confirmed_segments = confirm_via_correlation(reaction, segments)

    padded_segments = pad_segments(confirmed_segments, len(reaction_data), pad_beginning=0.75, pad_ending=0.25)
    merged_segments = merge_segments(padded_segments, min_segment_length, max_gap_frames)

    suppressed_reaction = apply_segments(reaction_data, merged_segments)


    if extra_reaction is not None:
        suppressed_reaction = np.concatenate((suppressed_reaction, extra_reaction))
    
    sf.write(output_path, suppressed_reaction, sr)

    if profile_isolator:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
        stats.print_stats()
        profiler.enable()

    return output_path



from scipy.signal import correlate

def confirm_via_correlation(reaction, segments, threshold=.9):
    from aligner.scoring_and_similarity import get_segment_mfcc_cosine_similarity_score
    from aligner.find_segment_start import correct_peak_index

    aligned_reaction_audio = reaction.get('aligned_reaction_data')
    aligned_reaction_mfcc = librosa.feature.mfcc(y=aligned_reaction_audio, sr=sr, n_mfcc=conf.get('n_mfcc'), hop_length=conf.get('hop_length'))

    song_audio = conf.get('song_audio_data')

    confirmed_segments = []
    for segment in segments:
        reaction_start, reaction_end = segment

        song_search_start = max(0, reaction_start - 5 * sr)
        song_search_end = min(len(song_audio) - 1, reaction_start + 5 * sr)

        song_segment = song_audio[song_search_start:song_search_end]
        reaction_segment = aligned_reaction_audio[reaction_start:reaction_end]

        correlation = correlate(song_segment, reaction_segment)
        song_start = song_search_start + correct_peak_index(np.argmax(correlation), len(reaction_segment))
        song_end = song_start + len(reaction_segment)

        full_segment = (reaction_start, reaction_end, song_start, song_end)
        similarity = get_segment_mfcc_cosine_similarity_score(reaction, full_segment, reaction_audio_mfcc=aligned_reaction_mfcc)

        if similarity > threshold:
            print(f'REMOVE! Candidate backchannel {reaction_start/sr:.1f}, {reaction_end/sr:.1f}<==>{song_start/sr:.1f}, {song_end/sr:.1f} failed confirmation. Cosine similarity={similarity}')
        else: 
            print(f'KEEP! Candidate backchannel {reaction_start/sr:.1f}, {reaction_end/sr:.1f}<==>{song_start/sr:.1f}, {song_end/sr:.1f} passed confirmation. Cosine similarity={similarity}')

            confirmed_segments.append(segment)

    return confirmed_segments



def isolate_reactor_backchannel(reaction, extended_by=0):

    conf.get('load_base_video')()
    
    output_dir = conf.get('temp_directory')
    reaction_audio = reaction.get('aligned_audio_path')
    base_audio = conf.get('base_audio_path')

    vocal_path_filename = 'vocals-post-high-passed.wav'
    base_dir = os.path.splitext(base_audio)[0].split('/')[-1]

    song_separation_path  = os.path.join(output_dir, base_dir)
    react_separation_path = os.path.join(output_dir, os.path.splitext(reaction_audio)[0].split('/')[-1] )

    reaction_vocals_path = os.path.join(react_separation_path, vocal_path_filename)
    song_vocals_path =     os.path.join(song_separation_path, vocal_path_filename)

    backchannel_filename = f"{reaction.get('channel')}_isolated_backchannel.wav"
    backchannel_path = os.path.join(output_dir, backchannel_filename)

    song_length = len(conf.get('song_audio_data')) / sr + 1

    if not os.path.exists( reaction_vocals_path ):
        separate_vocals(react_separation_path, reaction_audio, vocal_path_filename, duration=song_length)

    if not os.path.exists( song_vocals_path ):
        separate_vocals(song_separation_path, base_audio, vocal_path_filename, duration=song_length)

    if not os.path.exists(backchannel_path):

        print(f"Separating commentary from {reaction_audio} to {backchannel_path}")
        mute_by_deviation(reaction, song_vocals_path, reaction_vocals_path, backchannel_path)

    return backchannel_path
















