from pydub import AudioSegment
import subprocess
import numpy as np

import os

import matplotlib.pyplot as plt

from utilities import conf, conversion_audio_sample_rate as sr

from backchannel_isolator.track_separation import (
    separate_vocals_for_song,
    separate_vocals_for_reaction,
)


def is_base_silent(start, end, threshold_db=-20):
    song_data = conf.get("song_audio_data")[start:end]
    return is_silent(song_data, threshold_db=threshold_db)


quiet_parts = {}


def get_quiet_parts_of_song(
    min_silent_duration=2, loudness_threshold=0.5, window_size_seconds=1, visualize=False
):
    global quiet_parts

    song_key = conf.get("song_key")
    if song_key not in quiet_parts:
        (
            silence_binary_mask,
            __,
            mean_loudness,
            loudness,
            loudness_threshold,
        ) = get_silence_binary_mask_for_song(
            min_silent_duration=min_silent_duration,
            loudness_threshold=loudness_threshold,
            window_size_seconds=window_size_seconds,
        )

        # Find contiguous silent sections
        quiet_sections = []
        start = None

        for i, is_silent in enumerate(silence_binary_mask):
            if is_silent and start is None:
                start = i * window_size_seconds  # Start of silent section
            elif not is_silent and start is not None:
                end = i * window_size_seconds  # End of silent section
                quiet_sections.append((start, end))
                start = None

        # If the song ends in silence, we need to capture the final section
        if start is not None:
            quiet_sections.append((start, len(silence_binary_mask) * window_size_seconds))

        quiet_parts[song_key] = quiet_sections

    quiet_sections = quiet_parts[song_key]

    if visualize:
        time = np.arange(len(loudness)) * window_size_seconds

        plt.figure(figsize=(10, 6))

        # Plot loudness
        plt.plot(time, loudness, label="Loudness")

        # Plot mean loudness as dashed line
        plt.axhline(mean_loudness, color="red", linestyle="--", label="Mean Loudness")

        # Plot loudness threshold as dashed line
        plt.axhline(
            loudness_threshold * mean_loudness,
            color="green",
            linestyle="--",
            label="Loudness Threshold",
        )

        # Plot silence binary mask
        plt.plot(time, silence_binary_mask * np.max(loudness), label="Silence Mask", alpha=0.5)

        plt.xlabel("Time (seconds)")
        plt.ylabel("Loudness")
        plt.title("Loudness and Silence Detection")
        plt.legend()
        plt.show()

    return quiet_sections


# Find silent regions (longer than 3 seconds and quieter than x% of mean loudness)
def get_silence_binary_mask_for_song(
    min_silent_duration=2,
    loudness_threshold=0.5,
    window_size_seconds=1,
    use_accompaniment_only=False,
):
    if use_accompaniment_only:
        accompaniment, vocals, __ = separate_vocals_for_song()
        audio, __ = sf.read(accompaniment)

    else:
        audio = conf.get("song_audio_data")

    loudness = compute_rms_loudness_chunked(audio, sr, window_size_seconds)
    mean_loudness = np.mean(loudness)

    silence_binary_mask = find_silence_binary_mask(
        loudness,
        mean_loudness,
        sr,
        min_silent_duration=min_silent_duration,
        window_size_seconds=window_size_seconds,
        threshold=loudness_threshold,
    )

    return (
        silence_binary_mask,
        window_size_seconds,
        mean_loudness,
        loudness,
        loudness_threshold,
    )


# Find silent regions (longer than 3 seconds and quieter than x% of mean loudness)
def get_silence_binary_mask_for_reaction(
    reaction,
    min_silent_duration=3,
    loudness_threshold=1,
    window_size_seconds=1,
    use_accompaniment_only=True,
):
    if use_accompaniment_only:
        accompaniment, vocals, __ = separate_vocals_for_reaction(reaction)
        reaction_audio, __ = sf.read(accompaniment)

    else:
        reaction_audio = reaction.get("reaction_audio_data")

    loudness = compute_rms_loudness_chunked(reaction_audio, sr, window_size_seconds)
    mean_loudness = np.mean(loudness)

    silence_binary_mask = find_silence_binary_mask(
        loudness,
        mean_loudness,
        sr,
        min_silent_duration=min_silent_duration,
        window_size_seconds=window_size_seconds,
        threshold=loudness_threshold,
    )

    return (
        silence_binary_mask,
        window_size_seconds,
        mean_loudness,
        loudness,
        loudness_threshold,
    )


def compute_rms_loudness_chunked(audio, sr, window_size_seconds):
    """
    Compute the RMS loudness of the audio signal in chunks to reduce memory usage.
    audio: The audio data (1D array).
    sr: Sample rate of the audio.
    window_size_seconds: The size of each chunk in seconds.
    """
    window_size = int(sr * window_size_seconds)  # Convert seconds to samples
    loudness = []

    # Process the audio in chunks
    for start in range(0, len(audio), window_size):
        end = start + window_size
        chunk = audio[start:end]
        if len(chunk) > 0:
            rms_value = np.sqrt(np.mean(chunk**2))
            loudness.append(rms_value)

    return np.array(loudness)


def find_silence_binary_mask(
    loudness,
    mean_loudness,
    sr,
    window_size_seconds,
    min_silent_duration,
    threshold,
):
    """Find silent regions where loudness is less than threshold * mean loudness."""
    silence_threshold = threshold * mean_loudness
    silence_mask = (loudness < silence_threshold).astype(int)

    # Convert min_silent_duration from seconds to samples (based on window size)
    min_silent_samples = int(min_silent_duration / window_size_seconds)  # Adjust for window size

    silence_binary_mask = np.zeros_like(silence_mask)

    start = 0
    while start < len(silence_mask):
        if silence_mask[start] == 1:
            end = start
            while end < len(silence_mask) and silence_mask[end] == 1:
                end += 1
            if (end - start) >= min_silent_samples:
                silence_binary_mask[start:end] = 1
            start = end
        else:
            start += 1

    return silence_binary_mask


def plot_dB_over_time(data, window_duration=3.0):
    from utilities import conversion_audio_sample_rate as sr

    """
    Plots the dB level of an audio file over time using a sliding window.

    Parameters:
    - filename: Path to the audio file.
    - window_duration: Duration of the sliding window in seconds.
    """

    # Number of samples in each window
    window_size = int(window_duration * sr)

    # Calculate the RMS energy for each window
    rms_values = []
    for start in range(0, len(data) - window_size + 1, window_size):
        window = data[start : start + window_size]
        rms_energy = np.sqrt(np.mean(window**2))
        rms_values.append(rms_energy)

    # Convert RMS energy values to dB
    rms_db = 20 * np.log10(rms_values)

    # Plotting
    times = np.arange(0, len(data) / sr, window_duration)[: len(rms_db)]
    plt.figure(figsize=(10, 5))
    plt.plot(times, rms_db, "-o")
    plt.title("dB Level Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def is_normalized(audio_data):
    """
    Determines if a given audio data array is normalized.

    Parameters:
    - audio_data: Numpy array containing the audio samples.

    Returns:
    - True if the audio data is normalized, False otherwise.
    """

    return np.all(audio_data >= -1.0) and np.all(audio_data <= 1.0)


def is_silent(data, threshold_db=-50):
    from utilities.audio_processing import convert_to_mono

    """
    Determines if a given audio segment is predominantly silent.
    
    Parameters:
    - data: the soundfile audio data.
    - threshold_db: The RMS energy threshold below which the segment is considered silent, in decibels.
    
    Returns:
    - True if the segment is silent, False otherwise.
    """

    data = convert_to_mono(data)

    assert is_normalized(data)

    # Calculate the RMS energy of the segment
    rms_energy = np.sqrt(np.mean(data**2))

    # Convert the RMS energy to dB
    rms_db = 20 * np.log10(rms_energy)

    # Check if the RMS energy in dB is below the threshold
    return rms_db < threshold_db


def detect_leading_silence(sound, silence_threshold=-20.0, chunk_size=10):
    """
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    """
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms : trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def get_edge_silence(video_file, trim_end=True):
    _, video_ext = os.path.splitext(video_file)

    sound = AudioSegment.from_file(video_file, format=video_ext[1:])

    start_trim = detect_leading_silence(sound, silence_threshold=-30.0)
    if trim_end:
        end_trim = detect_leading_silence(sound.reverse(), silence_threshold=-30.0)
    else:
        end_trim = 0

    start = start_trim
    end = len(sound) - end_trim

    start /= 1000
    end /= 1000

    print(f"Edge silence for {video_file}: ({start_trim}, {end_trim})")

    return (start, end)


def trim_silence(video_file, output_file):
    # Step 1: Detect silence
    start, end = get_edge_silence(video_file)

    # Step 2: Remove silence

    if start_trim > 0 or end_trim > 10:
        cmd = ["ffmpeg", "-i", video_file, "-ss", f"{start}", "-to", f"{end}", output_file]
        print(cmd)
        subprocess.run(cmd)

        print(f"Trimmed video saved as {output_file}")

        subprocess.run(["mv", output_file, video_file])


def trim_silence_from_all(
    directory: str, reactions_dir: str = "reactions", output_dir: str = "aligned"
):
    reaction_dir = os.path.join(directory, output_dir)

    base_video = glob.glob(os.path.join(directory, f"{os.path.basename(directory)}.mp4"))
    if len(base_video) > 0:
        base_video = base_video[0]
        base_video_name, base_video_ext = os.path.splitext(base_video)

        trim_silence(base_video, f"{base_video_name}-trimmed{base_video_ext}")

    # react_videos = glob.glob(os.path.join(reaction_dir, "*.mp4"))

    # for react_video in react_videos:
    #     react_video_name, react_video_ext = os.path.splitext(react_video)
    #     trim_silence(react_video, f"{react_video_name}-trimmed{react_video_ext}")


if __name__ == "__main__":
    songs = ["Ren - Fire", "Ren - Genesis", "Ren - The Hunger"]

    for song in songs:
        trim_silence_from_all(song, "reactions", "aligned")
