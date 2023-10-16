from scipy import signal
import numpy as np
from utilities import conversion_audio_sample_rate as sr
from utilities import conf
import librosa


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



def calculate_percentile_loudness(loudness, window_size=1000, std_dev=None, hop_length=1):
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

    if hop_length > 1:
        # Sub-sample the percent_of_max_loudness array using the hop_length
        percent_of_max_loudness = percent_of_max_loudness[::hop_length]

    return percent_of_max_loudness


def calculate_percentile_loudness2(loudness, window_size=1000, std_dev=None, hop_length=1):
    if std_dev is None:
        std_dev = window_size / 3

    # Find the maximum loudness value
    max_loudness = np.max(loudness)
    
    # Calculate the percentage of the max for each loudness value
    percent_of_max_loudness = (loudness / max_loudness) * 100
    
    # Define a Gaussian window
    window = signal.windows.gaussian(window_size, std=std_dev)
    
    # Normalize the window to have sum 1
    window /= window.sum()
    
    percent_of_max_loudness = np.convolve(percent_of_max_loudness, window, mode='same')  # Convolve audio with window

    if hop_length > 1:
        # Sub-sample the percent_of_max_loudness array using the hop_length
        percent_of_max_loudness = percent_of_max_loudness[::hop_length]

    return percent_of_max_loudness


def audio_percentile_loudness(audio, loudness_window_size=100, percentile_window_size=1000, std_dev_percentile=None, hop_length=1 ):

    loudness = calculate_loudness(audio, window_size=loudness_window_size, hop_length=hop_length)

    percentile_loudness = calculate_percentile_loudness(loudness, window_size=percentile_window_size, std_dev=std_dev_percentile, hop_length=hop_length)

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




