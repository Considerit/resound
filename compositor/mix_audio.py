import numpy as np
import subprocess
import os
import math
import soundfile as sf

from moviepy.editor import AudioFileClip, VideoFileClip

from utilities import conf, conversion_audio_sample_rate as sr, save_object_to_file

from utilities.audio_processing import calculate_perceptual_loudness
from utilities import print_profiling

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pyloudnorm as pyln


def write_audio_to_file(reaction, audio, name):
    channel = reaction.get("channel")
    path = os.path.join(conf.get("temp_directory"), f"{channel}-int-{name}.wav")
    sf.write(path, audio, sr)


def get_peak_normalized_base_audio_and_rms(base_video=None):
    knobs = conf.get("audio_mixing", {})

    if base_video is None:
        base_video = VideoFileClip(conf.get("base_video_path"))

    normalize_base_with_headroom = knobs.get(
        "normalize_base_with_headroom", 0.2
    )  # base volume set to 1 - headroom

    base_video.audio.fps = sr
    base_audio_as_array = base_video.audio.to_soundarray()

    base_audio_as_array = peak_normalize_with_headroom(
        base_audio_as_array, headroom=normalize_base_with_headroom
    )
    base_audio_rms = rms_level_excluding_silence(base_audio_as_array)

    return base_audio_as_array, base_audio_rms


def mix_audio(base_video, output_size):
    # Mixing knobs
    knobs = conf.get("audio_mixing", {})
    normalize_base_with_headroom = knobs.get(
        "normalize_base_with_headroom", 0.2
    )  # base volume set to 1 - headroom
    foreground_scaler = knobs.get("foreground_scaler", 1)
    background_scaler = knobs.get("background_scaler", 1)
    sum_reaction_audio_threshold = knobs.get("sum_reaction_audio_threshold", 0.99)
    base_excess_limitation = knobs.get("base_excess_limitation", 0.75)
    stereo_pan_max = knobs.get("stereo_pan_max", 0.75)

    base_audio_as_array, base_audio_rms = get_peak_normalized_base_audio_and_rms(
        base_video
    )

    for channel, reaction in conf.get("reactions").items():
        print(f"\t\tLoading and volume matching audio for {channel}")
        reactors = reaction.get("reactors")
        if reactors is None:
            continue

        audio_path = reaction.get("backchannel_audio")
        isolated_backchannel = AudioFileClip(audio_path).to_soundarray()

        reaction["mixed_audio"] = isolated_backchannel

    (
        audio_scaling_factors,
        audible_segments,
    ) = foreground_background_backchannel_segments(
        base_audio_as_array,
        foreground_scaler,
        background_scaler,
        conf.get("disable_backchannel_backgrounding"),
    )

    all_reactor_audios = []

    print("\tMixing all audio clips")

    for channel, reaction in conf.get("reactions").items():
        print(f"\t\tMixing audio for {channel}")

        audio_scaling_factor = audio_scaling_factors[channel]
        audio_scaling_factor_reshaped = audio_scaling_factor[:, np.newaxis]

        reaction["mixed_audio"] *= audio_scaling_factor_reshaped
        # write_audio_to_file(reaction, reaction['mixed_audio'], 'adjusted_for_foreground')

        position = reaction.get("reactors")[0]["position"]

        # Pan the reactor's audio based on grid assignment
        pan_position = (2 * position[0] / output_size[0]) - 1
        reaction["mixed_audio"] = pan_audio_stereo(
            reaction["mixed_audio"], pan_position, max_pan=stereo_pan_max
        )
        # write_audio_to_file(reaction, reaction['mixed_audio'], 'after_panning')

    if len(conf.get("reactions").items()) > 0:
        # Apply dynamic limiting on the collected reactor audios
        dynamic_limit_without_combining(
            base_audio_as_array, base_excess_limitation, sum_reaction_audio_threshold
        )

    return base_audio_as_array, audible_segments


def find_contiguous_backchannel_segments(audio, max_silence_duration_samples=1.5):
    # Identify contiguous backchannel segments in each reaction audio. A segment
    # can have at most 2 seconds of consecutive silence while still being treated
    # as contiguous. audio is the result of a moviepy audioclip to_soundarray call.
    # Silence is represented by the value of 0 (it was previously processed as such).

    # Define the max silence duration allowed within a contiguous segment (in samples)
    max_silence_duration_samples *= sr

    # Create a boolean array: True for non-silent samples, False for silent samples
    non_silent_samples = np.any(audio != 0, axis=1)

    # Find the indices where the boolean array changes from True to False or vice versa
    changes = np.diff(non_silent_samples).nonzero()[0] + 1

    # Include the start and end of the array if they are non-silent
    if non_silent_samples[0]:
        changes = np.insert(changes, 0, 0)
    if non_silent_samples[-1]:
        changes = np.append(changes, len(non_silent_samples))

    # Pair the change points to form segments of non-silence
    segments = [(changes[i], changes[i + 1]) for i in range(0, len(changes), 2)]

    # Merge segments that are separated by a short duration of silence
    contiguous_segments = []
    for start, end in segments:
        if (
            contiguous_segments
            and start - contiguous_segments[-1][1] <= max_silence_duration_samples
        ):
            contiguous_segments[-1] = [contiguous_segments[-1][0], end]
        else:
            contiguous_segments.append([start, end])

    return contiguous_segments


# Mixing all of the backchannel together equally can cause a cacophony during the parts of the song when
# there are many active backchannels.

# Instead, I'd like to designate each contiguous segment of backchannel audio as either foreground
# (which will eventually be assigned e.g. 80% of the reactor volume for its duration) or as background
# (which will eventually be mixed equally at 20% of allocated reactor volume).


def foreground_background_backchannel_segments(
    base_audio_as_array,
    foreground_scaler,
    background_scaler,
    disable_backchannel_backgrounding,
):
    channel_segments = []
    scaling_factors = {}
    audible_segments = {}

    print("\tDetermining foreground and background")
    foregrounded_backchannel_segments = []
    backgrounded_backchannel_segments = []

    meter = pyln.Meter(sr)  # create BS.1770 meter

    for channel, reaction in conf.get("reactions").items():
        print(f"\t\tForeground/background audio for {channel}")

        foregrounded_backchannel = reaction.get("foregrounded_backchannel", [])
        backgrounded_backchannel = reaction.get("backgrounded_backchannel", [])

        # 1) Identify contiguous backchannel channel_segments in each reaction audio. A segment
        #      can have at most 2 seconds of consecutive silence while still being treated
        #      as contiguous.
        reaction_audio = reaction.get("mixed_audio")
        contiguous_backchannel_segments = audible_segments[
            channel
        ] = find_contiguous_backchannel_segments(reaction_audio)

        scaling_factors[channel] = np.zeros(len(reaction_audio))
        for seg in contiguous_backchannel_segments:
            start, end = seg
            # score = seg[0][1] - seg[0][0]
            # score = np.sum(calculate_perceptual_loudness(reaction_audio[start:end]))

            loudness = meter.integrated_loudness(reaction_audio[start:end])
            perc_of_song = 100 * (end - start) / base_audio_as_array.shape[0]

            segment_length = (end - start) / sr

            # best segments are 3-7 secs, long segments are usually not great
            if segment_length < 1:
                score_multiplier = 0.75
            if segment_length < 3:
                score_multiplier = 1.25
            elif segment_length < 7:
                score_multiplier = 2.5
            elif segment_length < 10:
                score_multiplier = 2
            else:
                score_multiplier = 1

            score = score_multiplier * (loudness + 30)

            # print(channel, score, loudness, perc_of_song)
            channel_seg = [seg, channel, score]
            channel_segments.append(channel_seg)

            # 2) Initialize each backchannel segment as foreground. We'll track this by creating
            #        a numpy array that will represent a scaling factor at each sample in the
            #        audio array. Foreground = foreground_scaler. Background = background_scaler. An audio mask I guess.

            scaling_factors[channel][start:end] = np.ones(end - start)

            for fb in foregrounded_backchannel:
                if start <= fb * sr <= end:
                    foregrounded_backchannel_segments.append(channel_seg)

            for fb in backgrounded_backchannel:
                if start <= fb * sr <= end or fb == -1:
                    backgrounded_backchannel_segments.append(channel_seg)

    # 3) Find the timestamp where there are the most foreground backchannels
    #     - Pick one of the backchannels to be the foreground
    #       ...probably based on total relative perceptible volume over relative perceptible
    #          volume of the base song for the duration of the backchannel segment
    #       - Mark the others at that timestamp as in the background
    #     - Loop until there aren't any sections with multiple foreground channels

    spacing = 1 * sr

    def construct_temporal_storage_for_backchannel_segments(foregrounded):
        duration = len(conf.get("song_audio_data"))
        bins = []
        for i in range(0, duration, spacing):
            center = i
            my_bin = []
            for chan in foregrounded:
                if segments_overlap(
                    chan[0], (center - spacing / 2, center + spacing / 2)
                ):
                    my_bin.append(chan)
            bins.append(my_bin)
        return bins

    def remove_from_foreground(bins, to_remove):
        starts = [chan[0][0] for chan in to_remove]
        ends = [chan[0][1] for chan in to_remove]

        earliest = min(starts)
        latest = max(ends)

        to_remove_keys = {}
        for chan in to_remove:
            key = str((chan[0], chan[1]))
            to_remove_keys[key] = True

        for i, my_bin in enumerate(bins):
            center = i * spacing

            if not (earliest / spacing - 1 <= i <= latest / spacing + 1):
                continue

            len_before = len(my_bin)
            bins[i] = [
                chan for chan in my_bin if str((chan[0], chan[1])) not in to_remove_keys
            ]

    def segments_overlap(seg1, seg2):
        # Check if two segments overlap
        start1, end1 = seg1
        start2, end2 = seg2

        return max(start1, start2) < min(end1, end2)

    def find_most_overlapping_backchannel_segments(bins):
        biggest_overlap = []

        for i, my_bin in enumerate(bins):
            # Find the largest list of channel segments in my_bin that overlap in time.
            # Each channel segment is a tuple whose first element is a (start, end) tuple
            # that can be used to determine overlap.

            # A list to store (overlap count, list of segments) tuples
            overlap_list = []

            for segment in my_bin:
                # Initialize overlap count for this segment
                overlap_count = 0
                overlapping_segments = []

                # Check overlap with other segments
                for other_segment in my_bin:
                    if segments_overlap(segment[0], other_segment[0]):
                        overlap_count += 1
                        overlapping_segments.append(other_segment)

                # Store the count and the segments that overlap
                overlap_list.append((overlap_count, overlapping_segments))

            # Find the entry with the maximum overlap count
            if len(overlap_list) > 0:
                max_overlap = max(overlap_list, key=lambda x: x[0])
                if max_overlap[0] > len(biggest_overlap):
                    biggest_overlap = max_overlap[1]

        return biggest_overlap

    def move_to_background(active_channel_segments, featured):
        max_score = featured[2]

        foregrounded = []
        total_score = 0
        for chan in active_channel_segments:
            if chan == featured:
                continue

            total_score += chan[2]
            foregrounded.append(chan)

        split_between = len(foregrounded)
        for chan in foregrounded:
            channel = chan[1]
            start, end = chan[0]
            score = chan[2]

            co_listener_scaler = background_scaler / max(2, split_between)
            quality_scaler = background_scaler * score / total_score
            scaler = min(1, max(co_listener_scaler, quality_scaler))
            # scaler = 0
            scaling_factors[channel][start:end] = np.minimum(
                scaler, scaling_factors[channel][start:end]
            )

        return foregrounded

    if not disable_backchannel_backgrounding:
        print("\t\tIteratively backgrounding")

        for featured in foregrounded_backchannel_segments:
            seg, channel, score = featured

            active_channel_segments = [
                s for s in channel_segments if segments_overlap(s[0], seg)
            ]

            print(
                f"\t\tFeatured backchannel for {channel} at {(seg[1]-seg[0]) / 2 / sr}. Backgrounding {len(active_channel_segments) - 1} other backchannels"
            )

            removed = move_to_background(active_channel_segments, featured)
            for chan in removed:
                channel_segments.remove(chan)

        bins = construct_temporal_storage_for_backchannel_segments(channel_segments)

        i = 0
        while True:
            print_profiling()

            active_channel_segments = find_most_overlapping_backchannel_segments(bins)

            if len(active_channel_segments) <= 1:
                break

            highest_scoring_segment = None
            for chan in active_channel_segments:
                if (
                    not highest_scoring_segment or chan[2] > highest_scoring_segment[2]
                ) and chan not in backgrounded_backchannel_segments:
                    highest_scoring_segment = chan

            to_remove = move_to_background(
                active_channel_segments, highest_scoring_segment
            )
            remove_from_foreground(bins, to_remove)

            i += 1

    # Any audio past the end of the base audio should be set to one (or how about 1 / num_reactors?)
    for channel, reaction in conf.get("reactions").items():
        audio_mask = scaling_factors[channel]
        diff = len(audio_mask) - len(base_audio_as_array)
        if diff > 0:
            audio_mask[-diff:] = np.full(
                diff, 1 / len(conf.get("reactions"))
            )  # np.ones(diff)

    plot_scaling_factors(scaling_factors)

    for channel, audible in audible_segments.items():
        for segment in audible:
            avg_factor = np.mean(scaling_factors[channel][segment[0] : segment[1]])
            segment.append(avg_factor)

            print("\t\t\t", channel, int(segment[0]), int(segment[1]), avg_factor)
            assert len(segment) == 3

    return scaling_factors, audible_segments


# Scaling_factors is a dict with channel as keys, and the value is an array over time with each entry between zero and one.
# In the plot, the x-axis should be time (in seconds). The y-axis will be categorical,
# with a row for each channel (height=10). The channel name should be written on the right side axis (not in a legend).
# Each row should be composed of line segments with slope 0 spanning contiguous sections of the same non-zero values.
# The color of each line segment should correspond to its spectrum value between 0 to 1, with red=0 and green=1.

# Scaling_factors is a dict. A channel name is the key (and there are potentially hundreds).
# The value is an array over time with each entry set to a number between zero and one.
# plot_scaling_factors visualizes this data as a heat map. The x-axis is the time (in seconds).
# The y axis is each channel, with about 15 pixels per channel. The channel name is on the
# right side axis (not in a legend). The color should reflect the value at that point in time
# on a spectrum from white (=0) to green (=1). The array is a mask of an audio array, and thus
# has sr samples per second. This is much higher resolution than we need. So the heat map
# is based on the data at a granularity that examines every hop_length (=5096) samples.
from matplotlib.colors import LinearSegmentedColormap


def plot_scaling_factors(scaling_factors, hop_length=5096, show=False):
    # Determine the number of channels and the maximum length of the arrays
    n_channels = len(scaling_factors)
    if n_channels == 0:
        return

    max_length = max(len(v) for v in scaling_factors.values())

    channels = list(scaling_factors.keys())
    channels.sort()

    # Resample and pad the data for each channel
    resampled_data = []
    for channel in channels:
        channel_data = scaling_factors[channel]
        # Resample the data
        resampled = channel_data[::hop_length]
        # Calculate the number of padding needed, ensuring it's not negative
        pad_width = max(0, (max_length - 1) // hop_length + 1 - len(resampled))
        # Pad the resampled data
        padded = np.pad(resampled, (0, pad_width), "constant", constant_values=0)
        resampled_data.append(padded)

    # Create a 2D array for the heatmap data
    heatmap_data = np.array(resampled_data)

    # Create a custom colormap that goes from white to black
    white_black_cmap = LinearSegmentedColormap.from_list(
        "white_black", ["white", "black"]
    )

    font_size = 8
    space_per_label = 1.5 * font_size
    total_label_space = space_per_label * n_channels
    total_height_inches = total_label_space / 72

    # Adjust figure size and aspect ratio here
    fig, ax = plt.subplots(figsize=(10, total_height_inches))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.99, bottom=0.05)

    cax = ax.imshow(
        heatmap_data, aspect="auto", cmap=white_black_cmap, interpolation="nearest"
    )

    # Set the channel names on the y-axis
    ax.set_yticks(np.arange(n_channels))
    ax.set_yticklabels(channels, fontsize=font_size)
    ax.yaxis.tick_right()  # Move y-axis labels to the right

    # Set the time (x-axis) labels for every 30 seconds
    max_time = max_length / sr
    tick_interval = 30  # Interval in seconds for the major ticks
    num_ticks = max_time // tick_interval  # Number of ticks to display

    # Generate tick positions and labels
    ticks = np.arange(0, num_ticks + 1) * tick_interval * sr / hop_length
    tick_labels = [f"{int(t)}s" for t in np.arange(0, num_ticks + 1) * tick_interval]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    # Set axis labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")

    # Add a color bar
    # plt.colorbar(cax, ax=ax)

    filename = os.path.join(
        conf.get("temp_directory"), f"backchannel-foreground-backchannel-audio.png"
    )
    plt.savefig(filename, dpi=500)

    if show:
        plt.show()

    plt.close()


def rms_level_excluding_silence(audio_array, threshold=0.01):
    """Compute the RMS level, excluding silence."""
    non_silent_samples = audio_array[np.abs(audio_array) > threshold]

    if len(non_silent_samples) == 0:
        return 1

    rms = np.sqrt(np.mean(non_silent_samples**2))

    if math.isnan(rms):
        print("NAN!", np.mean(non_silent_samples**2), non_silent_samples)

    return rms


LUFS_normalization = -16


def adjust_gain_for_loudness_match(audio_array, channel):
    meter = pyln.Meter(sr)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio_array)

    # loudness normalize audio to LUFS_normalization dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(
        audio_array, loudness, LUFS_normalization
    )

    print(f"Loudness of {channel}={loudness}, adjusting to {LUFS_normalization}db LUFS")
    return loudness_normalized_audio


def adjust_gain_for_rms_match(audio_array, target_rms, channel):
    """Adjust the gain of audio_array to match the target RMS."""
    current_rms = rms_level_excluding_silence(audio_array)
    print(
        f"MATCHING RMS of {channel}={current_rms} to base RMS={target_rms}. Adjustment is {target_rms / current_rms} louder"
    )
    return audio_array * (target_rms / current_rms)


def peak_normalize_with_headroom(audio_array, headroom=0.025):
    """Peak normalize the audio with a given headroom."""
    peak = np.max(np.abs(audio_array))
    scale_factor = (1 - headroom) / peak
    return audio_array * scale_factor


def dynamic_limit_without_combining(
    base_audio, base_excess_limitation, sum_reaction_audio_threshold=0.99
):
    """Dynamically limit the reactor audios in chunks to avoid exceeding the threshold."""
    chunk_size = int(sr * 0.5)  # Using half-second chunks

    all_reaction_audio = [r.get("mixed_audio") for r in conf.get("reactions").values()]

    # Find out the max length amongst all audios
    max_len = max(
        len(base_audio),
        max([len(reactor_audio) for reactor_audio in all_reaction_audio]),
    )

    # Find the peak volume of the base (song) audio
    base_peak_volume = np.max(np.abs(base_audio))

    # For padding purposes
    def pad_audio_chunk(chunk, size):
        if len(chunk) < size:
            padding = np.zeros((size - len(chunk), chunk.shape[1]))
            return np.vstack((chunk, padding))
        return chunk

    for i in range(0, max_len, chunk_size):
        chunk_end = min(i + chunk_size, max_len)

        # Fetching base audio chunk and padding if necessary
        base_chunk = base_audio[i:chunk_end]
        base_chunk = pad_audio_chunk(base_chunk, chunk_size)

        combined_chunk = base_chunk.copy()

        # Fetching reactor audios chunk-by-chunk and summing them up
        for reactor_audio in all_reaction_audio:
            reactor_chunk = (
                reactor_audio[i:chunk_end]
                if i < len(reactor_audio)
                else np.zeros_like(base_chunk)
            )
            reactor_chunk = pad_audio_chunk(reactor_chunk, chunk_size)
            combined_chunk += reactor_chunk

        # Calculate the maximum allowable volume for the reactor audio in this chunk
        if chunk_end <= len(base_audio):
            max_allowable_volume = (
                np.max(np.abs(base_chunk)) + base_excess_limitation * base_peak_volume
            )
        else:
            max_allowable_volume = 1

        # Determine scaling factor based on the new constraint and existing threshold
        scaling_factor = min(
            1,
            sum_reaction_audio_threshold / np.max(np.abs(combined_chunk)),
            max_allowable_volume / np.max(np.abs(combined_chunk)),
        )

        # Apply scaling factor back to the original chunks of reactor audios
        for reactor_audio in all_reaction_audio:
            if i < len(reactor_audio):
                reactor_chunk = reactor_audio[i:chunk_end]
                reactor_audio[i:chunk_end] = reactor_chunk * scaling_factor


def pan_audio_stereo(audio_array, pan_position, max_pan=0.7):
    """
    Pan the audio in the stereo field with variable constraints.
    -1.0 is fully left
     1.0 is fully right
     0.0 is centered
    The 'max_pan' argument defines the maximum panning as a ratio, where
    1.0 is full panning and 0.0 is no panning (always centered).
    For example, a 'max_pan' of 0.7 means the audio will not pan more than a 70/30 split.
    """
    # Ensure the maximum pan ratio is within 0.0 to 1.0
    max_pan = np.clip(max_pan, 0.0, 1.0)

    # Calculate the minimum pan ratio
    min_pan = 1.0 - max_pan

    # Ensure stereo audio
    if len(audio_array.shape) == 1:
        audio_array = np.array([audio_array, audio_array]).T

    # Limit the pan_position to the range that enforces the desired max/min split
    limited_pan_position = (max_pan - min_pan) * pan_position

    # Calculate gain for each channel
    left_gain = np.clip(1 - limited_pan_position, min_pan, max_pan)
    right_gain = np.clip(1 + limited_pan_position, min_pan, max_pan)

    # Apply the gain
    audio_array[:, 0] *= left_gain
    audio_array[:, 1] *= right_gain

    return audio_array
