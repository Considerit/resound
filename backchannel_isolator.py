import cv2
import numpy as np
import subprocess
import os
import numpy as np
import scipy.signal
import librosa
import soundfile as sf

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import signal




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






def calculate_mfcc(audio, sr):
    # Calculate the MFCCs for the audio
    mfcc = librosa.feature.mfcc(audio, sr)
    return mfcc


def calculate_difference(mfcc1, mfcc2):
    # Calculate the difference between the two MFCCs
    diff = np.abs(mfcc1 - mfcc2)
    return diff






##########################
# We have been trying to solve the following problem:

# I'm trying to isolate the sections of a reaction audio in which one or more commentators are making noise. 
# We also have the original song, and they are *mostly* aligned in time.  

# In my preprocessing, I’m separating audio by source into a vocal track, and ignoring the accompaniment. 
# This gets rid of most of the music from consideration, leaving us with vocals. 

# The first step in my algorithm for identifying parts of a reaction audio where commentators are speaking 
# is to identify the parts of the reaction audio that is most different from the base song (using MFCCs 
#     to get at perceptual differences). 

# To do this, I'm creating a mask that is true where the absolute difference between the two tracks is 
# greater than some threshold.

# Setting this threshold in a principled way is tricky. Our first attempt looks like: 

# def create_mask(diff, percentile):
#     # Calculate the nth percentile of the absolute differences
#     threshold = np.percentile(np.abs(diff), percentile)
#     # Create a mask where the absolute difference is greater than the threshold
#     mask = np.abs(diff) > threshold
#     return mask

# However, because I have separated out the vocals, much of both of the tracks are nearly silent. 
# This distorts the percentile approach to setting a threshold. 

# So what we've been working on has been trying to ignore the near-silent sections of the audio when 
# setting the threshold. We've been first calculating frame energy to identify near-silent parts of 
# the audio, and then trying to use that (unsucessfully) filter the MFCC diff used to compute a 
# threshold for masking the reaction audio.  

# We've been running into lots of errors though in the dimensions of the various arrays involved, 
# and in interpreting the silence threshold as a decibal=>energy mapping. I've included our latest 
# code below. Please fix it, or identify a different method for achieving the goal outlined above. Thanks!


def calculate_frame_energy(audio, frame_length, hop_length):
    # Frame the audio signal
    frames = librosa.util.frame(audio, frame_length, hop_length)
    # Calculate the energy of each frame
    energy = np.sum(frames**2, axis=0)
    return energy

def create_mask_by_mfcc_difference(song, sr_song, reaction, sr_reaction, percentile, silence_threshold=.001):


    mfcc_song = calculate_mfcc(song, sr_song)
    mfcc_reaction = calculate_mfcc(reaction, sr_reaction)

    if mfcc_song.shape != mfcc_reaction.shape:
        print("The shapes of the MFCCs do not match!")
        return
        

    print("mfcc_song shape: ", mfcc_song.shape)
    print("mfcc_reaction shape: ", mfcc_reaction.shape)

    diff = calculate_difference(mfcc_song, mfcc_reaction)


    # Calculate the frame energy
    frame_length = 2048  # You may need to adjust this
    hop_length = 512  # and this
    frame_energy = calculate_frame_energy(reaction, frame_length, hop_length)

    # Create a mask where the frame energy is greater than the silence threshold
    not_silent = frame_energy > silence_threshold

    # Reduce the dimensionality of diff by calculating the Euclidean norm along the MFCC axis
    diff_norm = np.linalg.norm(diff, axis=0)

    # Check if not_silent contains any True values
    if not np.any(not_silent):
        print("All frames are silent. Please check your silence threshold.")
        return None

    # Reshape not_silent to match diff_norm
    not_silent_frames = np.reshape(not_silent, (1, len(not_silent)))

    # Repeat the not_silent_frames to match the rows in diff
    not_silent = np.repeat(not_silent_frames, diff.shape[0], axis=0)

    # Adjust the shape of not_silent to match the size of diff
    if diff.shape[1] > not_silent.shape[1]:
        padding = diff.shape[1] - not_silent.shape[1]
        not_silent = np.pad(not_silent, ((0, 0), (0, padding)), constant_values=False)
        
    # Mask diff with not_silent
    diff_masked = diff.copy()
    diff_masked[~not_silent] = 0

    # Calculate the percentile of the absolute differences where not silent
    threshold = np.percentile(np.abs(diff_masked), percentile)

    # Create a mask where the absolute difference is greater than the threshold
    mask = np.abs(diff) > threshold

    return mask[0]






# def calculate_perceptual_loudness(audio):
#     # Calculate the STFT of the audio
#     stft = librosa.stft(audio)

#     # Calculate the RMS energy, which we'll use as our loudness measure
#     rms = librosa.feature.rms(S=np.abs(stft))[0]

#     print("lengths should be equal", len(audio), len(rms))

#     return rms

def calculate_perceptual_loudness(audio, frame_length=2048, hop_length=512):
    # Calculate the RMS energy over frames of audio
    rms = np.array([np.sqrt(np.mean(np.square(audio[i:i+frame_length]))) for i in range(0, len(audio), hop_length)])
    
    # Interpolate the rms to match the length of audio
    rms_interp = np.interp(np.arange(len(audio)), np.arange(0, len(audio), hop_length), rms)
    
    return rms_interp




def calculate_loudness(audio, window_size=1000):
    loudness = calculate_perceptual_loudness(audio)

    # Define a Gaussian window
    window = signal.windows.gaussian(window_size, std=window_size/10)
    
    # Normalize the window to have sum 1
    window /= window.sum()
    
    loudness = np.convolve(loudness, window, mode='same')  # Convolve audio with window

    return loudness



def calculate_percentile_loudness(loudness, window_size=1000):
    # Find the maximum loudness value
    max_loudness = np.max(loudness)
    
    # Calculate the percentage of the max for each loudness value
    percent_of_max_loudness = (loudness / max_loudness) * 100
    

    # Define a Gaussian window
    window = signal.windows.gaussian(window_size, std=window_size/10)
    
    # Normalize the window to have sum 1
    window /= window.sum()
    
    percent_of_max_loudness = np.convolve(percent_of_max_loudness, window, mode='same')  # Convolve audio with window


    return percent_of_max_loudness


def create_mask_by_volume_difference(song, sr_song, reaction, sr_reaction, threshold, silence_threshold=.001, plot=False):
    # Calculate the loudness of each audio
    song_loudness = calculate_loudness(song)
    reaction_loudness = calculate_loudness(reaction)

    # Calculate the percentile loudness of each audio
    song_percentile_loudness = calculate_percentile_loudness(song_loudness)
    reaction_percentile_loudness = calculate_percentile_loudness(reaction_loudness)

    # Calculate the absolute difference between the percentile loudnesses
    diff = reaction_percentile_loudness - song_percentile_loudness
    print("diff shape: ", diff.shape)

    # Create a mask where the absolute difference is greater than the threshold
    mask = diff > threshold

    # Create a mask where the difference is greater than zero
    dilated_mask = diff > threshold

    # dilated_mask should be an expansion of mask. It should be True for all segments of positive values of diff 
    # that contain at least one sample where diff > threshold.
    i = 0
    while i < len(mask):
        if mask[i]:
            # find start of positive segment
            start = i
            while i < len(mask) and diff[i] > 1:
                i += 1
            # mark segment in dilated_mask
            dilated_mask[start:i] = True
        else:
            dilated_mask[i] = False
            i += 1


    if plot: 
        # Create a time array for plotting
        time_song = np.arange(song_percentile_loudness.shape[0]) / sr_song
        time_reaction = np.arange(reaction_percentile_loudness.shape[0]) / sr_reaction

        # Plot the percentile loudnesses
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(time_song, song_percentile_loudness, label='Song')
        plt.plot(time_reaction, reaction_percentile_loudness, label='Reaction')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Percentile Loudness')
        plt.title('Percentile Loudness of Song and Reaction')

        # Plot the absolute difference
        plt.subplot(3, 1, 2)
        plt.plot(time_reaction, diff, label='Absolute Difference')
        plt.ylim(bottom=0)  # Constrain y-axis to non-negative values    
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Difference')
        plt.title('Absolute Difference in Percentile Loudness')

        # Plot the masks
        plt.subplot(3, 1, 3)
        plt.plot(time_reaction, mask, label='Mask')
        plt.plot(time_reaction, dilated_mask, label='Dilated Mask')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Mask Values')
        plt.title('Masks of Commentator Segments')


        plt.tight_layout()
        plt.show()



    return dilated_mask














def process_mask(mask, min_segment_length, sr):
    # Number of frames corresponding to the minimum segment length
    min_segment_frames = int(min_segment_length * sr / 512)  # 512 is the default hop length in librosa's mfcc function

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

def merge_segments(segments, min_segment_length, max_gap_length, sr):
    # Number of frames corresponding to the minimum segment length
    min_segment_frames = int(min_segment_length * sr / 512)  # 512 is the default hop length in librosa's mfcc function

    # Number of frames corresponding to the maximum gap length
    max_gap_frames = int(max_gap_length * sr / 512)  # 512 is the default hop length in librosa's mfcc function

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

def filter_segments_by_volume(audio, sr, segments, volume_percentile=50):
    # Calculate the STFT of the audio
    stft = librosa.stft(audio)

    # Calculate the RMS energy (approximate volume) of the audio
    rms = librosa.feature.rms(S=np.abs(stft))

    # Convert RMS energy to 1-D array for simplicity
    rms = rms[0, :]

    # Find the maximum RMS value across the entire track
    max_volume = np.max(rms)

    # Determine the volume threshold
    volume_threshold = volume_percentile / 100.0 * max_volume

    # Filter segments where the max volume is greater than the threshold
    filtered_segments = [segment for segment in segments if np.max(rms[segment[0]:segment[1]]) > volume_threshold]

    return filtered_segments


def pad_segments(segments, sr, length, pad_length=0.1):
    # Convert padding length from seconds to samples
    pad_length_samples = int(pad_length * sr)
        
    padded_segments = []
    # For each segment
    for segment in segments:
        # Pad the start and end of the segment
        start = max(0, segment[0] - pad_length_samples)
        end = min(length, segment[1] + pad_length_samples)
        padded_segments.append([start, end])

    
    return padded_segments


def apply_segments(audio, segments):
    # Initialize a new audio array with zeros
    suppressed_audio = np.zeros_like(audio)

    # For each segment, copy the corresponding audio from the original reaction
    for (start, end) in segments:
        suppressed_audio[start:end] = audio[start:end]

    return suppressed_audio





def mute_by_deviation(song_path, reaction_path, output_path):
    min_segment_length = 0.05
    max_gap_frames = 0.15
    percentile_thresh = 90
    percent_volume_diff_thresh = 10
    do_matching = False

    # Load the song and reaction audio
    song, sr_song = librosa.load(song_path, sr=None, mono=True)

    # Load reaction audio
    reaction, sr_reaction = librosa.load(reaction_path, sr=None, mono=True)

    assert sr_reaction == sr_song, f"Sample rates must be equal {sr_reaction} {sr_song}"



    if do_matching:
        reaction = match_audio(reaction, song, sr_reaction)


    # if mono:
    #     # Convert them to pseudo multi-channel (still mono but with an extra dimension to look like multi-channel)
    #     song = np.expand_dims(song, axis=0)
    #     reaction = np.expand_dims(reaction, axis=0)

    # Check the standard deviation of the audio data
    # assert song.shape == reaction.shape, f"Song and reaction must have the same number of channels (song = {song.shape}, reaction = {reaction.shape})"

    delay = 0
    if len(song) > len(reaction):
        song = song[:len(reaction)]
    else:
        pad_width = len(reaction) - len(song) - delay
        song = np.pad(song, (delay, pad_width))


    mask = create_mask_by_volume_difference(song, sr_song, reaction, sr_reaction, percent_volume_diff_thresh)
    segments = process_mask(mask, min_segment_length / 1000, sr_reaction)

    # merged_segments = merge_segments(segments, min_segment_length, max_gap_frames, sr_reaction)  # Maximum gap of 0.1 seconds
    # merged_segments = filter_segments_by_volume(reaction, sr_reaction, merged_segments)

    padded_segments = pad_segments(segments, sr_reaction, len(reaction), pad_length=0.15)
    merged_segments = merge_segments(padded_segments, min_segment_length, max_gap_frames, sr_reaction)



    suppressed_reaction = apply_segments(reaction, merged_segments)


    # if mono:
    #     # Squeeze the 2D arrays to 1D
    #     suppressed_reaction = np.squeeze(suppressed_reaction)

    
    sf.write(output_path, suppressed_reaction.T, sr_reaction)

    # suppressed_reaction = post_process_audio(suppressed_reaction, original_sr_reaction)
    # # Now you can save suppressed_reaction into a file
    # output_path = os.path.splitext(reaction_path)[0] + "_commentary.wav"
    # sf.write(output_path, suppressed_reaction.T, original_sr_reaction)  # Transpose the output because soundfile expects shape (n_samples, n_channels)

    return output_path



def process_reactor_audio(reaction_audio, base_audio):
    print(f"Separating commentary from {reaction_audio}")

    # if not "Pegasus" in reaction_audio:
    #     print("Skipping...")
    #     return

    # straight_mute_path = mute_by_deviation(base_audio, reaction_audio)


    reaction_vocals_path, song_vocals_path = separate_vocals(base_audio, reaction_audio, post_process=True)
    output_path = os.path.splitext(reaction_vocals_path)[0] + f"_isolated_commentary.wav"
    if not os.path.exists(output_path):
        mute_by_deviation(song_vocals_path, reaction_vocals_path, output_path)

    return output_path




def match_audio(reaction_audio, song_audio, sr):
    # reaction_audio = equalize_spectra(song_audio, reaction_audio, sr_reaction)

    # # Compute dB scale for volume normalization
    # db_song = librosa.amplitude_to_db(np.abs(librosa.stft(song_audio)))
    # db_reaction = librosa.amplitude_to_db(np.abs(librosa.stft(reaction_audio)))
    
    # # Normalize volume
    # reaction_audio *= np.median(db_song) / np.median(db_reaction)

    segment_length = 10 * sr

    reaction_audio = adaptive_normalize_volume(song_audio, reaction_audio, segment_length)

    # reaction_audio = adaptive_pitch_matching(song_audio, reaction_audio, segment_length, sr) 

    # reaction_audio = spectral_subtraction_with_stft(reaction_audio, song_audio)


    # # Compute pitch for pitch normalization
    # song_padded_stft = np.abs(librosa.stft(song_audio))
    # reaction_stft = np.abs(librosa.stft(reaction_audio))
    
    # pitches_song, _ = librosa.piptrack(S=song_padded_stft, sr=sr)
    # pitches_reaction, _ = librosa.piptrack(S=reaction_stft, sr=sr)
    
    # # Compute median pitch of each track
    # pitch_song = np.median(pitches_song[pitches_song > 0])
    # pitch_reaction = np.median(pitches_reaction[pitches_reaction > 0])

    # # Normalize pitch
    # reaction_audio = librosa.effects.pitch_shift(reaction_audio, sr=sr, n_steps=(pitch_song - pitch_reaction))


    return reaction_audio




def subtract_song(song_path, reaction_path):

    mono = True
    # Load the song and reaction audio
    song, original_sr_song = librosa.load(song_path, sr=None, mono=mono)

    # Load reaction audio
    reaction, original_sr_reaction = librosa.load(reaction_path, sr=None, mono=mono)

    assert original_sr_reaction == original_sr_song, "Sample rates must be equal"

    target_sr = original_sr_reaction # * 4

    # Load the song and reaction audio
    song, sr_song = librosa.load(song_path, sr=target_sr, mono=mono)

    # Load reaction audio
    reaction, sr_reaction = librosa.load(reaction_path, sr=target_sr, mono=mono) 


    if mono:
        # Convert them to pseudo multi-channel (still mono but with an extra dimension to look like multi-channel)
        song = np.expand_dims(song, axis=0)
        reaction = np.expand_dims(reaction, axis=0)

    # Check the standard deviation of the audio data
    assert song.shape[0] == reaction.shape[0], f"Song and reaction must have the same number of channels (song = {song.shape[0]}, reaction = {reaction.shape[0]})"

    delay = 0
    if len(song[0]) > len(reaction[0]):
        song_padded = song[:,:len(reaction[0])]
    else:
        song_padded = np.pad(song, ((0, 0), (delay, len(reaction[0]) - len(song[0]) - delay)))


    commentary = np.zeros_like(reaction)


    def find_shift_energy(shift, song_audio, reaction_audio):
        # Shift the song
        if shift < 0:
            song_shifted = np.concatenate([song_audio[abs(shift):], np.zeros(abs(shift))])
        else:
            song_shifted = np.concatenate([np.zeros((shift,)), song_audio[:-shift]])

        # Ensure song_shifted and reaction_audio are the same length
        if len(song_shifted) < len(reaction_audio):
            song_shifted = np.concatenate([song_shifted, np.zeros(len(reaction_audio) - len(song_shifted))])
        elif len(song_shifted) > len(reaction_audio):
            song_shifted = song_shifted[:len(reaction_audio)]

        # Subtract the song from the reaction audio
        commentary_test = reaction_audio - song_shifted

        # Compute the energy excluding the first and last 3 seconds
        energy = np.sum(commentary_test[int(3*sr_reaction):-int(3*sr_reaction)]**2)

        # Update the minimum energy and best shift
        print(f"Considering shift {shift} [{energy}  {energy / min_energy}]", end='\r')

        return energy


    for channel in range(song.shape[0]):


        # Preprocessing steps
        reaction[channel] = match_audio(reaction[channel], song_padded[channel], sr_reaction)

        # The range of shifts to test
        no_shift = reaction[channel] - song_padded[channel]
        min_energy = np.sum(no_shift[int(3*sr_reaction):-int(3*sr_reaction)]**2)
        best_shift = 0

        

        for shift_range in [range(0,-100,-1), range(0,100,1)]:
            last_shift = float('inf')
            for shift in shift_range:
                energy = find_shift_energy(shift, song_padded[channel], reaction[channel])
                if energy < min_energy:
                    min_energy = energy
                    best_shift = shift
                # if last_shift < energy and (abs(shift) > 120 or min_energy * 4 < energy):
                #     break

                last_shift = energy


        print(f"\nBest shift is {best_shift} [{min_energy}] \n")

        if best_shift < 0:
            song_padded[channel] = np.concatenate([song_padded[channel, abs(best_shift):], np.zeros(abs(best_shift))])
        else:
            song_padded[channel] = np.concatenate([song_padded[channel, best_shift:], np.zeros(best_shift)])

        # Ensure same length before equalization
        if len(song_padded[channel]) > len(reaction[channel]):
            reaction[channel] = np.pad(reaction[channel], (0, len(song_padded[channel]) - len(reaction[channel])))
        else:
            song_padded[channel] = np.pad(song_padded[channel], (0, len(reaction[channel]) - len(song_padded[channel])))

        
        
        # Subtract the song from the reaction audio
        commentary[channel] = reaction[channel] - song_padded[channel]

        check_negative_version(reaction[channel], commentary[channel], song_padded[channel])



    if mono:
        # Squeeze the 2D arrays to 1D
        commentary = np.squeeze(commentary)

    commentary = librosa.resample(commentary, sr_reaction, original_sr_reaction)

    output_path = os.path.splitext(reaction_path)[0] + "_after_subtraction.wav"
    sf.write(output_path, commentary.T, original_sr_reaction)

    # commentary = post_process_audio(commentary, original_sr_reaction)
    # # Now you can save commentary into a file
    # output_path = os.path.splitext(reaction_path)[0] + "_commentary.wav"
    # sf.write(output_path, commentary.T, original_sr_reaction)  # Transpose the output because soundfile expects shape (n_samples, n_channels)

    return output_path







def check_negative_version(reaction_audio, resulting_audio, song_audio):
    # calculate correlations
    corr_resulting = np.corrcoef(reaction_audio, resulting_audio)[0,1]
    corr_inverted_song = np.corrcoef(reaction_audio, -song_audio)[0,1]
    
    # check which correlation is higher
    if corr_resulting > corr_inverted_song:
        print('The resulting audio seems not to be a negative version of the original.')
    else:
        print('The resulting audio may be a negative version of the original.')
    
    print('Correlation between reaction and resulting audio:', corr_resulting)
    print('Correlation between reaction and inverted song:', corr_inverted_song)



def adaptive_normalize_volume(song, reaction, segment_length):
    # Make sure that the song and reaction are numpy arrays
    assert isinstance(song, np.ndarray) and isinstance(reaction, np.ndarray)
    # Make sure the segment length is a positive integer
    assert isinstance(segment_length, int) and segment_length > 0

    # Calculate the number of segments
    num_segments = len(song) // segment_length

    # Reshape the song and reaction into a 2D array of segments
    song_segments = song[:num_segments*segment_length].reshape(-1, segment_length)
    reaction_segments = reaction[:num_segments*segment_length].reshape(-1, segment_length)

    # Calculate the RMS volume of each segment
    song_volumes = np.sqrt(np.mean(song_segments**2, axis=1))
    reaction_volumes = np.sqrt(np.mean(reaction_segments**2, axis=1))

    # Avoid divide-by-zero errors
    reaction_volumes = np.where(reaction_volumes == 0, 1, reaction_volumes)

    # Calculate the volume scale factors
    scale_factors = song_volumes / reaction_volumes

    # Limit scale factors to be at least 1
    scale_factors = np.maximum(scale_factors, 1.0)

    # Apply the scale factors to the reaction segments
    reaction_segments = reaction_segments * scale_factors[:, np.newaxis]

    # Flatten the segments back into a 1D array
    normalized_reaction = reaction_segments.flatten()

    # If the original reaction was longer than num_segments*segment_length, append the leftover samples without scaling
    if len(reaction) > len(normalized_reaction):
        normalized_reaction = np.append(normalized_reaction, reaction[len(normalized_reaction):])

    return normalized_reaction

def adaptive_pitch_matching(song, reaction, segment_length, sr):
    # Make sure that the song and reaction are numpy arrays
    assert isinstance(song, np.ndarray) and isinstance(reaction, np.ndarray)
    # Make sure the segment length is a positive integer
    assert isinstance(segment_length, int) and segment_length > 0

    # Calculate the number of segments
    num_segments = len(song) // segment_length

    # Reshape the song and reaction into a 2D array of segments
    song_segments = song[:num_segments*segment_length].reshape(-1, segment_length)
    reaction_segments = reaction[:num_segments*segment_length].reshape(-1, segment_length)

    # For each segment
    for i in range(num_segments):
        # Compute the median pitch of each segment
        pitch_song = np.median(librosa.piptrack(song_segments[i], sr=sr)[0])
        pitch_reaction = np.median(librosa.piptrack(reaction_segments[i], sr=sr)[0])

        # Apply the pitch shift to the reaction segment
        reaction_segments[i] = librosa.effects.pitch_shift(reaction_segments[i], sr=sr, n_steps=(pitch_song - pitch_reaction))

    # Flatten the segments back into a 1D array
    pitch_matched_reaction = reaction_segments.flatten()

    # If the original reaction was longer than num_segments*segment_length, append the leftover samples without shifting
    if len(reaction) > len(pitch_matched_reaction):
        pitch_matched_reaction = np.append(pitch_matched_reaction, reaction[len(pitch_matched_reaction):])

    return pitch_matched_reaction


# Can you suggest or write a python function that could post-process what is returned the subtract_song function 
# you just wrote? This post-processing function would try to clean up some of the non-speech artifacts 
# created by the subtraction.



from scipy.signal import butter, lfilter

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def post_process_audio_with_highpass_filter(audio, sr, cutoff=100):
    return highpass_filter(audio, cutoff, sr)



# import numpy as np
from scipy.fftpack import fft, ifft


def equalize_spectra(audio1, audio2, sr):
    # Compute the Fourier transform of both signals
    fft_audio1 = fft(audio1)
    fft_audio2 = fft(audio2)

    # Calculate the spectral division
    mask = np.abs(fft_audio1) / np.abs(fft_audio2 + 1e-10)  # The small number prevents division by zero

    # Apply the spectral mask to the second signal
    fft_audio2_eq = fft_audio2 * mask

    # Compute the inverse Fourier transform to get the equalized signal in the time domain
    audio2_eq = np.real(ifft(fft_audio2_eq))

    return audio2_eq


import scipy.signal

def spectral_subtraction_with_stft(audio1, audio2, n_fft=2048, hop_length=512):
    # Compute the STFT of both signals
    stft_audio1 = librosa.stft(audio1, n_fft=n_fft, hop_length=hop_length)
    stft_audio2 = librosa.stft(audio2, n_fft=n_fft, hop_length=hop_length)

    # Calculate the power spectra
    power_audio1 = np.abs(stft_audio1)**2
    power_audio2 = np.abs(stft_audio2)**2

    # Perform power spectral subtraction
    power_diff = power_audio1 - power_audio2
    power_diff = np.maximum(power_diff, 0)  # Ensure the result is nonnegative

    # Compute the phase of the original signal
    phase_audio1 = np.angle(stft_audio1)

    # Construct the modified STFT by combining the square root of the subtracted power spectrum with the original phase
    stft_diff = np.sqrt(power_diff) * np.exp(1j * phase_audio1)

    # Compute the inverse STFT to get the resulting signal in the time domain
    audio_diff = librosa.istft(stft_diff, hop_length=hop_length)

    return audio_diff




import noisereduce as nr

def post_process_audio(commentary, sr):

    # mono = commentary.ndim == 1

    # # If the audio is mono (1D array), add an extra dimension to make it look like single-channel multi-channel
    # if mono:
    #     commentary = np.expand_dims(commentary, axis=0)

    # processed_commentary = np.zeros_like(commentary)
    
    # for channel in range(commentary.shape[0]):
    #     # Reduce noise
    #     reduced_noise = nr.reduce_noise(commentary[channel], sr=sr)
        
    #     # Apply a highpass filter to remove low-frequency non-speech components
    #     sos = scipy.signal.butter(10, 100, 'hp', fs=sr, output='sos')
    #     filtered = scipy.signal.sosfilt(sos, reduced_noise)
        
    #     processed_commentary[channel] = filtered

    # if mono:
    #     processed_commentary = processed_commentary[0]

    # return processed_commentary

    # Reduce noise
    reduced_noise = nr.reduce_noise(commentary, sr=sr)
    
    # Apply a highpass filter to remove low-frequency non-speech components
    sos = scipy.signal.butter(10, 100, 'hp', fs=sr, output='sos')
    filtered = scipy.signal.sosfilt(sos, reduced_noise)
    
    return filtered



def pre_process_audio(commentary, sr):
    processed_commentary = np.zeros_like(commentary)
    
    for channel in range(commentary.shape[0]):
        # Reduce noise
        reduced_noise = commentary[channel]
        # reduced_noise = nr.reduce_noise(commentary[channel], sr=sr)
        
        # Apply a highpass filter to remove low-frequency non-speech components
        sos = scipy.signal.butter(10, 100, 'hp', fs=sr, output='sos')
        reduced_noise = scipy.signal.sosfilt(sos, reduced_noise)


        # Raise the volume
        # Compute dB scale for volume normalization
        db_reaction = librosa.amplitude_to_db(np.abs(librosa.stft(reaction[channel])))

        # Normalize volume
        song_padded[channel] *= 1.1 * np.median(db_reaction) / np.median(db_song)

        
        processed_commentary[channel] = reduced_noise #filtered

    return processed_commentary



from spleeter.separator import Separator
import numpy as np
from scipy.spatial import distance
from scipy.signal import correlate
import librosa
from scipy.signal import fftconvolve

def separate_vocals(song_path, reaction_path, post_process=False):
    # Create a separator with 2 stems (vocals and accompaniment)
    separator = Separator('spleeter:2stems')

    print(f"reaction {reaction_path}")

    song_sep = os.path.join(os.path.dirname(reaction_path), 'song_output')
    song_separation_path = os.path.join(song_sep, os.path.splitext(song_path)[0].split('/')[-1] )

    reaction_sep = os.path.join(os.path.dirname(reaction_path), 'reaction_output')
    react_separation_path = os.path.join(reaction_sep, os.path.splitext(reaction_path)[0].split('/')[-1] )

    print(f"reaction: {reaction_sep}")

    # Perform source separation on song and reaction audio
    if not os.path.exists(song_separation_path):
        song_sources = separator.separate_to_file(song_path, song_sep)
    if not os.path.exists(react_separation_path):
        reaction_sources = separator.separate_to_file(reaction_path, reaction_sep)

    # Load the separated tracks
    song_vocals_path = os.path.join(song_separation_path, 'vocals.wav')
    reaction_vocals_path = os.path.join(react_separation_path, 'vocals.wav')


    if post_process: 

        song_vocals_high_passed_path = os.path.join(song_separation_path, 'vocals-post-high-passed.wav')
        song_vocals, sr_song = librosa.load( song_vocals_path, sr=None, mono=True )
        if not os.path.exists(song_vocals_high_passed_path):
            song_vocals = post_process_audio_with_highpass_filter(song_vocals, sr_song)  
            song_vocals = post_process_audio(song_vocals, sr_song)
            sf.write(song_vocals_high_passed_path, song_vocals.T, sr_song)
        song_vocals_path = song_vocals_high_passed_path

        reaction_vocals_high_passed_path = os.path.join(react_separation_path, 'vocals-post-high-passed.wav')
        reaction_vocals, sr_reaction = librosa.load( reaction_vocals_path, sr=None, mono=True )
        
        if not os.path.exists(reaction_vocals_high_passed_path):
            reaction_vocals = post_process_audio_with_highpass_filter(reaction_vocals, sr_reaction)  
            reaction_vocals = post_process_audio(reaction_vocals, sr_reaction)
            sf.write(reaction_vocals_high_passed_path, reaction_vocals.T, sr_reaction)
        reaction_vocals_path = reaction_vocals_high_passed_path

        assert sr_reaction == sr_song, f"Sample rates must be equal {sr_reaction} {sr_song}"

        # print(f"{reaction_vocals.shape} {len(reaction_vocals)}    {song_vocals.shape}   {len(song_vocals)}")
        # reaction_vocals = match_audio(reaction_vocals, song_vocals, sr_reaction)
        # reaction_vocals_path = os.path.join(react_separation_path, 'vocals-post-high-passed-volume-matched.wav')
        # sf.write(reaction_vocals_path, reaction_vocals.T, sr_reaction)


    return (reaction_vocals_path, song_vocals_path)

