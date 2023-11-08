from scipy import signal
from scipy.signal import correlate
from scipy.interpolate import interp1d

import numpy as np
from utilities import conversion_audio_sample_rate as sr
import librosa
import math
import os

def find_best_alignment(song_data, reaction_data, ref_tuple, search_window):
    song_sr, reaction_sr = ref_tuple
    song_sr = int(song_sr * sr)
    reaction_sr = int(reaction_sr * sr)

    # Ensure the segments do not exceed the bounds of the data
    song_start = max(0, song_sr - search_window)
    song_end = min(len(song_data), song_sr + search_window)

    reaction_start = max(0, reaction_sr - (song_sr - song_start))
    reaction_end = min(len(reaction_data), reaction_sr + (song_end - song_sr))

    # Extract segments around the reference points
    song_segment = song_data[song_start:song_end]
    reaction_segment = reaction_data[reaction_start:reaction_end]
    
    # Check for length discrepancies and adjust
    if len(song_segment) < len(reaction_segment):
        reaction_segment = reaction_segment[:len(song_segment)]
    elif len(reaction_segment) < len(song_segment):
        song_segment = song_segment[:len(reaction_segment)]

    # Compute cross-correlation
    cross_corr = correlate(song_segment, reaction_segment, mode="valid")

    # Find the offset for the best alignment
    offset = np.argmax(cross_corr) - len(reaction_segment) + 1

    # Adjust reference sample rates based on the found offset
    adjusted_reaction_sr = reaction_sr + offset

    return song_sr, adjusted_reaction_sr

def find_normalization_factor(song_data, reaction_data, ref_tuple, search_window=None):
    print(f"\tNormalizing to {ref_tuple[0]} <===> {ref_tuple[1]}")
    if search_window is None:
        search_window = int(sr / 2)
    
    # Find the best alignment
    song_sr, reaction_sr = find_best_alignment(song_data, reaction_data, ref_tuple, search_window)

    # Ensure the segments do not exceed the bounds of the data
    song_start = max(0, song_sr - search_window)
    song_end = min(len(song_data), song_sr + search_window)

    reaction_start = max(0, reaction_sr - (song_sr - song_start))
    reaction_end = min(len(reaction_data), reaction_sr + (song_end - song_sr))

    # Compute the sum of absolute amplitudes in the search window
    sum_amplitude_song = np.sum(np.abs(song_data[song_start:song_end]))
    sum_amplitude_reaction = np.sum(np.abs(reaction_data[reaction_start:reaction_end]))

    # Avoid division by zero
    if sum_amplitude_reaction == 0:
        raise ValueError("The sum of absolute amplitudes in the reaction data segment is zero. Cannot normalize.")

    # Compute the normalization factor
    normalization_factor = sum_amplitude_song / sum_amplitude_reaction

    return normalization_factor


def normalize_audio_manually(song_data, reaction_data, ref_tuple, search_window=None):

    normalization_factor = find_normalization_factor(song_data, reaction_data, ref_tuple, search_window)
    normalization_factor = max(.9, normalization_factor)
    normalized_reaction_data = reaction_data * normalization_factor
    print(f"\t\tDone! normalization_factor={normalization_factor}")

    return song_data, normalized_reaction_data








def get_highest_amplitude_segments(audio, segment_length, top_n):
    from scipy.signal import find_peaks


    print(f"Finding the top {top_n} amplitude segments")


    if top_n < 20:
        dist = 10*sr
    elif top_n < 50: 
        dist = 5*sr
    else: 
        dist = sr


    window_length = int(segment_length * sr)
    step_size = int(.5 * sr)

    amplitude_sums = np.array([np.sum(np.abs(audio[i:i+window_length])) for i in range(0, len(audio) - window_length, step_size)])
    
    # We're using negative amplitude sums to find valleys (lowest amplitude segments)
    # Since we want the segments of highest amplitude, we search for the valleys of the negative signal
    peaks, _ = find_peaks(-amplitude_sums, distance=dist // step_size)

    # Sort the peaks based on the amplitude and take the top_n segments
    top_indices = peaks[np.argsort(amplitude_sums[peaks])][-top_n:] * step_size

    return top_indices

def best_matching_segment(song_segment, reaction_audio, song_mfcc_segment, reaction_mfcc, hop_length):
    from scipy.signal import correlate, find_peaks
    from aligner.scoring_and_similarity import mfcc_cosine_similarity
    # Use cross-correlation to find potential alignment

    cross_corr = correlate(reaction_audio, song_segment)
    potential_match_start = np.argmax(cross_corr)
    potential_match_start = max(0, potential_match_start - (len(song_segment) - 1))
    
    potential_match_start_mfcc = int(potential_match_start / hop_length)

    return potential_match_start, mfcc_cosine_similarity(song_mfcc_segment, reaction_mfcc[:, potential_match_start_mfcc:potential_match_start_mfcc+song_mfcc_segment.shape[1]])


import statistics
def normalize_reaction_audio(reaction, song_audio, reaction_audio, song_mfcc, reaction_mfcc, segment_length=1, top_n=10, shot_in_the_dark=False, good_match_threshold=.9):
    from utilities import conf


    print('\tNormalizing audio')
    print('\t\tGetting highest amplitude segments')

    hop_length = conf.get('hop_length')

    top_song_indices = get_highest_amplitude_segments(song_audio, segment_length, top_n)
    window_length = int(segment_length * sr)
    
    best_matching_indices = []

    for idx,index in enumerate(top_song_indices):
        song_segment = song_audio[index:index+window_length]
        song_mfcc_segment = song_mfcc[:, int(index/hop_length):int((index+window_length)/hop_length)]

        reaction_start = max(index, reaction.get('start_reaction_search_at', 0))
        reaction_end = reaction.get('end_reaction_search_at', len(reaction_audio))


        reaction_audio_segment = reaction_audio[reaction_start:reaction_end]
        reaction_mfcc_segment = reaction_mfcc[:, int(reaction_start/hop_length):int(reaction_end/hop_length)]
        
        match, score = best_matching_segment(song_segment, reaction_audio_segment, song_mfcc_segment, reaction_mfcc_segment, hop_length)
        if not math.isnan(score):
            best_matching_indices.append( ((index, reaction_start + match), score)   )  
            print(f'\t\t{idx}: {score}  {index / sr} <=> {(reaction_start + match) / sr}')

            # if score > good_match_threshold and top_n > 50:
            #     break




    top_matches = [b for b in best_matching_indices if b[1] > good_match_threshold]

    if len(top_matches) > 0: 
        top_matches.sort(key=lambda x: x[1], reverse=True)

        normalization_factors = []
        for match in top_matches:
            ref_tuple, score = match

            normalization_factor = find_normalization_factor(song_audio, reaction_audio, (ref_tuple[0] / sr, ref_tuple[1] / sr))
            normalization_factors.append(normalization_factor)

        def avg(arr):
            return sum(arr) / len(arr)

        normalization_factor = avg([avg(normalization_factors), statistics.median(normalization_factors)]) # midmean
        normalization_factor = max(.9, min(4, normalization_factor))

        print(f"\t\tDone! normalization_factor={normalization_factor}", normalization_factors)

        return normalization_factor


    else: 
        if top_n < 100: 
            next_top_n = top_n * 5
            return normalize_reaction_audio(reaction, song_audio, reaction_audio, song_mfcc, reaction_mfcc, segment_length=1, top_n=next_top_n, shot_in_the_dark=shot_in_the_dark, good_match_threshold=good_match_threshold)
        
        elif not shot_in_the_dark: 
            normalized_reaction_data = reaction_audio * 5 # Almost always failure to find an alignment point is when the reaction audio is way low.
                                                          # We'll try to give it a boost and then try again.

            normalized_reaction_mfcc = librosa.feature.mfcc(y=normalized_reaction_data, sr=sr, n_mfcc=conf.get('n_mfcc'), hop_length=conf.get('hop_length'))

            return normalize_reaction_audio(reaction, song_audio, normalized_reaction_data, song_mfcc, normalized_reaction_mfcc, segment_length=1, top_n=10, shot_in_the_dark=True, good_match_threshold=.85)

        else:
            print("Failed to find a normalization point", shot_in_the_dark)
            return 1


    











def compute_waveform(signal, window_size=4410, hop_size=2205):
    """
    Compute the waveform (envelope) of the signal using RMS over windows.
    
    :param signal: Input audio signal
    :param window_size: Size of the window for RMS computation
    :param hop_size: Hop size between windows
    :return: Waveform of the signal
    """
    num_windows = int((len(signal) - window_size) / hop_size) + 1
    waveform = np.zeros(num_windows)
    
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        window = signal[start:end]
        waveform[i] = np.sqrt(np.mean(window**2))
    
    return waveform

def normalize_waveform(waveform, reference_waveform):
    """
    Normalize the amplitude of a waveform based on a reference waveform.
    
    :param waveform: The waveform to be normalized
    :param reference_waveform: The waveform to normalize against
    :return: Normalized waveform
    """
    scale_factor = np.max(reference_waveform) / np.max(waveform)
    return waveform * scale_factor

def get_normalized_waveform(reaction_signal, song_signal=None, window_size=4410, hop_size=2205):
    """
    Get the normalized waveform of a reaction signal based on a song signal.
    If no song signal is provided, returns the unnormalized waveform of the reaction signal.
    
    :param reaction_signal: Reaction audio signal
    :param song_signal: Song audio signal (Optional)
    :param window_size: Size of the window for RMS computation
    :param hop_size: Hop size between windows
    :return: (Normalized) waveform of the reaction signal
    """
    reaction_waveform = compute_waveform(reaction_signal, window_size, hop_size)
    
    if song_signal is not None:
        song_waveform = compute_waveform(song_signal, window_size, hop_size)
        return normalize_waveform(reaction_waveform, song_waveform)
    else:
        return reaction_waveform






def pitch_contour(y, sr, fmin=50, fmax=4000, hop_length=512):
    """
    Extracts the pitch contour of audio data using the Yin algorithm.
    
    Parameters:
    - y: Audio time series.
    - sr: Sampling rate.
    - fmin: Minimum frequency to consider for pitch (in Hz).
    - fmax: Maximum frequency to consider for pitch (in Hz).

    Returns:
    - pitches: Array of pitch values in Hz.
    """

    # Compute pitch using Yin algorithm
    pitches, _ = librosa.piptrack(y=y, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length)

    # Extract the maximum pitch for each frame
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = pitches[:, t].argmax()
        pitch_values.append(pitches[index, t])

    pitch_matrix = np.array(pitch_values).reshape(1, -1)
    
    return pitch_matrix



def spectral_flux(y, sr, hop_length=512):
    """
    Computes the spectral flux of audio data.
    
    Parameters:
    - y: Audio time series.
    - sr: Sampling rate.

    Returns:
    - flux: A list containing spectral flux values for each time frame.
    """

    # Compute magnitude spectrogram
    S = np.abs(librosa.stft(y, hop_length))

    # Compute the spectral flux
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))


    flux = flux.reshape(1, -1)
    
    return flux




def root_mean_square_energy(y, sr, frame_length=2048, hop_length=512):
    """
    Computes the Root Mean Square Energy (RMSE) of audio data.
    
    Parameters:
    - y: Audio time series.
    - sr: Sampling rate.
    - frame_length: Number of samples in each frame for analysis.
    - hop_length: Number of samples between successive frames.

    Returns:
    - rmse_values: An array containing the RMSE values for each frame.
    """

    rmse_values = librosa.feature.rms(y, frame_length=frame_length, hop_length=hop_length, center=True)

    rmse_values = np.array(rmse_values[0]).reshape(1, -1)
    
    return rmse_values




def continuous_wavelet_transform(y, sr, wavelet='cmor', max_scale=128, scale_step=1):
    import pywt

    """
    Computes the Continuous Wavelet Transform (CWT) of audio data.
    
    Parameters:
    - y: Audio time series.
    - sr: Sampling rate.
    - wavelet: Type of wavelet to use. 'cmor' is the Complex Morlet wavelet, suitable for audio.
    - max_scale: Maximum scale (or frequency) to compute the CWT for.
    - scale_step: Step size between scales.

    Returns:
    - cwt_result: 2D array containing the CWT coefficients. 
                  Shape is (number of scales, length of y).
    """

    scales = np.arange(1, max_scale + 1, scale_step)
    cwt_result, _ = pywt.cwt(y, scales, wavelet, sampling_period=1/sr)

    return cwt_result








def convert_to_mono(soundfile_audio):
    # Check the number of dimensions
    if len(soundfile_audio.shape) == 1:
        return soundfile_audio  # It's already mono
    elif soundfile_audio.shape[1] > 1:
        return np.mean(soundfile_audio, axis=1)  # Convert stereo to mono
    else:
        return soundfile_audio  # It's mono with a shape like (n, 1), just return it as-is



def calculate_perceptual_loudness(audio, frame_length=2048, hop_length=None):
    from utilities import conf

    if hop_length is None:
        hop_length = conf.get('hop_length')
        
    # Calculate the RMS energy over frames of audio
    rms = np.array([np.sqrt(np.mean(np.square(audio[i:i+frame_length]))) for i in range(0, len(audio), hop_length)])
    
    # Interpolate the rms to match the length of audio
    rms_interp = np.interp(np.arange(len(audio)), np.arange(0, len(audio), hop_length), rms)
    
    return rms_interp


def calculate_loudness(audio, window_size=100, hop_length=None):
    loudness = calculate_perceptual_loudness(audio, hop_length=hop_length)

    # Define a Gaussian window
    window = signal.windows.gaussian(window_size, std=window_size/10)
    
    # Normalize the window to have sum 1
    window /= window.sum()
    
    loudness = np.convolve(loudness, window, mode='same')  # Convolve audio with window

    return loudness



def calculate_percentile_loudness(loudness, window_size=1000, std_dev=None):
    if std_dev is None:
        std_dev = window_size / 3

    # Find the maximum loudness value and compute the percent_of_max_loudness directly
    factor = 100.0 / np.max(loudness)
    percent_of_max_loudness = loudness * factor
    
    # Define a Gaussian window and normalize it
    window = signal.windows.gaussian(window_size, std=std_dev)
    window /= window.sum()

    # Convolve audio with window using fftconvolve
    percent_of_max_loudness = signal.fftconvolve(percent_of_max_loudness, window, mode='same')

    return percent_of_max_loudness


def audio_percentile_loudness(audio, loudness_window_size=100, percentile_window_size=1000, std_dev_percentile=None, hop_length=1 ):

    loudness = calculate_loudness(audio, window_size=loudness_window_size, hop_length=hop_length)

    percentile_loudness = calculate_percentile_loudness(loudness, window_size=percentile_window_size, std_dev=std_dev_percentile)

    return percentile_loudness







# def calculate_pitch(audio, sr, frame_length=2048, hop_length=512):
#     # Calculate the pitch over frames of audio
#     pitch = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
#                         sr=sr, frame_length=frame_length, hop_length=hop_length)
    
#     # Interpolate the pitch to match the length of audio
#     # Create a time array for the pitch data, scaled to match the audio length
#     time_pitch = np.linspace(0, len(audio), num=len(pitch))
#     pitch_interp = np.interp(np.arange(len(audio)), time_pitch, pitch)
    
#     return pitch_interp



# def calculate_pitch_level(audio, sr, window_size=1000):
#     pitch = calculate_pitch(audio, sr)

#     # Define a Gaussian window
#     window = signal.windows.gaussian(window_size, std=window_size/5)
    
#     # Normalize the window to have sum 1
#     window /= window.sum()
    
#     pitch = np.convolve(pitch, window, mode='same')  # Convolve audio with window

#     return pitch

# def calculate_percentile_pitch(pitch, window_size=5000):
#     # Find the maximum pitch value
#     max_pitch = np.max(pitch)
    
#     # Calculate the percentage of the max for each pitch value
#     percent_of_max_pitch = (pitch / max_pitch) * 100
    
#     # Define a Gaussian window
#     window = signal.windows.gaussian(window_size, std=window_size/10)
    
#     # Normalize the window to have sum 1
#     window /= window.sum()
    
#     percent_of_max_pitch = np.convolve(percent_of_max_pitch, window, mode='same')  # Convolve audio with window

#     return percent_of_max_pitch






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




def match_audio(reaction_audio, song_audio):
    # reaction_audio = equalize_spectra(song_audio, reaction_audio)

    # # Compute dB scale for volume normalization
    # db_song = librosa.amplitude_to_db(np.abs(librosa.stft(song_audio)))
    # db_reaction = librosa.amplitude_to_db(np.abs(librosa.stft(reaction_audio)))
    
    # # Normalize volume
    # reaction_audio *= np.median(db_song) / np.median(db_reaction)

    segment_length = 10 * sr

    reaction_audio = adaptive_normalize_volume(song_audio, reaction_audio, segment_length)

    # reaction_audio = adaptive_pitch_matching(song_audio, reaction_audio, segment_length) 

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

def adaptive_pitch_matching(song, reaction, segment_length):
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


# import numpy as np
from scipy.fftpack import fft, ifft


def equalize_spectra(audio1, audio2):
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




