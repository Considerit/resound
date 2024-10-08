import math, random
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import correlate, find_peaks
from aligner.scoring_and_similarity import (
    mse_mfcc_similarity,
    mfcc_cosine_similarity,
    raw_cosine_similarity,
)


from utilities import conf, on_press_key
from utilities import conversion_audio_sample_rate as sr

from utilities import save_object_to_file, read_object_from_file


def unique(list1):
    present = {}
    new_list = []
    for l in list1:
        if not l[0] in present:
            new_list.append(l)
            present[l[0]] = True
    return new_list


def correct_peak_index(peak_index, chunk_len):
    # return max(0,peak_index)
    return max(0, peak_index - (chunk_len - 1))


seg_start_cache = {}
full_search_cache = {}


def initialize_segment_start_cache():
    global seg_start_cache

    seg_start_cache.clear()
    # seg_start_cache.update(cache)
    full_search_cache.clear()


# For debugging:
# If ground truth is defined, we try to filter down to only the paths
# that include the ground truth. It isn't perfect, but can really help
# to figure out if the system at least generates the ground truth path.
# Even if it doesn't select it.


# TODO:
#   - ditch the open chunk / closed chunk and use song / reaction


def find_segment_starts(
    reaction,
    open_chunk,
    closed_chunk,
    current_chunk_size,
    peak_tolerance,
    open_start,
    closed_start,
    distance,
    upper_bound=None,
    hop_length=1,
    signal="standard",
):
    global full_search_start_cache
    global seg_start_cache

    key = f"{conf.get('song_key')} {signal} {open_start} {closed_start} {len(open_chunk)} {len(closed_chunk)} {upper_bound} {peak_tolerance} {hop_length}"

    multi_dimensional = not isinstance(open_chunk, (list, tuple)) and np.ndim(open_chunk) > 1

    if upper_bound is not None:
        # print(f"BOUNDING {closed_start / sr}! {upper_bound / sr} => {(upper_bound + closed_start + current_chunk_size)/sr}")

        # if not multi_dimensional:
        #     open_chunk = open_chunk[   :int((upper_bound + closed_start - open_start + current_chunk_size ) / hop_length)]
        # else:
        #     open_chunk = open_chunk[:, :int((upper_bound + closed_start - open_start + current_chunk_size ) / hop_length)]
        if not multi_dimensional:
            open_chunk = open_chunk[: int((upper_bound + closed_start - open_start) / hop_length)]
        else:
            open_chunk = open_chunk[
                :, : int((upper_bound + closed_start - open_start) / hop_length)
            ]

    if not multi_dimensional:
        if len(open_chunk) == 0 or len(closed_chunk) == 0:
            print(
                "\nError! Open chunk and/or closed chunk are zero length",
                len(open_chunk),
                len(closed_chunk),
                upper_bound + closed_start,
                open_start,
                current_chunk_size,
            )
            return None

        correlation = correlate(open_chunk, closed_chunk)
    else:
        if open_chunk.shape[1] == 0 or closed_chunk.shape[1] == 0:
            print(
                "\nError! Open chunk and/or closed chunk are zero length",
                open_chunk.shape,
                closed_chunk.shape,
                upper_bound + closed_start,
                open_start,
                current_chunk_size,
            )
            return None

        correlations = [
            correlate(open_chunk[i, :], closed_chunk[i, :]) for i in range(open_chunk.shape[0])
        ]
        correlation = np.mean(correlations, axis=0)
        correlation = correlation - np.mean(correlation[100 : len(correlation) - 100])

    cross_max = np.max(correlation)

    # Find peaks
    peak_indices, _ = find_peaks(
        correlation,
        height=cross_max * (peak_tolerance + (1 - peak_tolerance) * 0.25),
        distance=distance / hop_length,
    )
    peak_indices2, _ = find_peaks(
        correlation,
        height=cross_max * peak_tolerance,
        distance=3 * distance / hop_length,
    )
    peak_indices = np.unique(np.concatenate((peak_indices, peak_indices2)))

    if hop_length == 1:
        peak_indices = [
            [pi, correct_peak_index(pi, current_chunk_size)] for pi in peak_indices.tolist()
        ]
    else:
        peak_indices = [
            [pi * hop_length, correct_peak_index(pi * hop_length, current_chunk_size)]
            for pi in peak_indices.tolist()
        ]

    if len(peak_indices) == 0:
        # print(f"No peaks found for {closed_start} [{closed_start / sr} sec] / {open_start} [{open_start / sr} sec] {np.max(correlation)}")
        return None

    # if upper_bound is not None:
    #     for c in peak_indices:
    #         print("\t", (c[1] + open_start - closed_start)/sr, " < "  , upper_bound / sr, "?")
    #         assert(c[1] + open_start - closed_start < upper_bound)

    return (peak_indices, correlation, key)


# peak_indices is a dictionary, with indicies for each signal. Each signal has peak indicies and the overall correlation
def score_start_candidates(
    reaction,
    signals,
    open_chunk,
    closed_chunk,
    open_chunk_mfcc,
    closed_chunk_mfcc,
    peak_indices,
    current_chunk_size,
    peak_tolerance,
    open_start,
    closed_start,
):
    assert (
        len(closed_chunk) == current_chunk_size,
        len(closed_chunk),
        current_chunk_size,
    )

    max_mfcc_mse_score = 0
    max_mfcc_cosine_score = 0
    max_raw_cosine_score = 0
    max_composite_score = 0

    scores_at_index = {}
    max_scores = {}

    hop_length = conf.get("hop_length")

    for signal, (candidate_indicies, __, __) in peak_indices.items():
        max_correlation_score = 0

        for unadjusted_candidate_index, candidate_index in candidate_indicies:
            open_chunk_here = open_chunk[candidate_index : candidate_index + current_chunk_size]

            if len(open_chunk_here) != current_chunk_size:
                # print(f"Skipping because we couldn't make a chunk of size {current_chunk_size} [{current_chunk_size / sr}] starting at {candidate_index} [{candidate_index / sr}] from chunk of length {len(open_chunk_here) / sr} (open chunk len={len(open_chunk)})")
                continue

            if candidate_index not in scores_at_index:
                open_chunk_here_mfcc = open_chunk_mfcc[
                    :,
                    round(candidate_index / hop_length) : round(
                        (candidate_index + current_chunk_size) / hop_length
                    ),
                ]
                mfcc_mse_score = mse_mfcc_similarity(open_chunk_here_mfcc, closed_chunk_mfcc)
                mfcc_cosine_score = mfcc_cosine_similarity(open_chunk_here_mfcc, closed_chunk_mfcc)
                raw_cosine_score = raw_cosine_similarity(
                    open_chunk[candidate_index : candidate_index + current_chunk_size],
                    closed_chunk,
                )
                composite_score = raw_cosine_score * mfcc_cosine_score * mfcc_mse_score

                scores_at_index[candidate_index] = (
                    mfcc_cosine_score,
                    mfcc_mse_score,
                    composite_score,
                    raw_cosine_score,
                    {},
                )

            (
                mfcc_cosine_score,
                mfcc_mse_score,
                composite_score,
                raw_cosine_score,
                signal_scores,
            ) = scores_at_index[candidate_index]

            for ssignal, (__, correlation, __) in peak_indices.items():
                hop_length = signals[ssignal][0]
                my_idx = math.floor(unadjusted_candidate_index / hop_length)
                if my_idx < correlation.shape[0]:
                    signal_scores[ssignal] = correlation[my_idx]
                else:
                    print("WEIRD! correlation index is out of bounds", ssignal, my_idx)

            correlation_score = signal_scores[signal]

            if correlation_score > max_correlation_score:
                max_correlation_score = correlation_score

            if mfcc_mse_score > max_mfcc_mse_score:
                max_mfcc_mse_score = mfcc_mse_score

            if mfcc_cosine_score > max_mfcc_cosine_score:
                max_mfcc_cosine_score = mfcc_cosine_score

            if raw_cosine_score > max_raw_cosine_score:
                max_raw_cosine_score = raw_cosine_score

            if composite_score > max_composite_score:
                max_composite_score = composite_score

        max_scores[signal] = max_correlation_score

    thresholds = {
        "mfcc_mse_score": peak_tolerance + (1 - peak_tolerance) * 0.25,
        "mfcc_cosine_score": peak_tolerance + (1 - peak_tolerance) * 0.25,
        "composite_score": peak_tolerance + (1 - peak_tolerance) * 0.25,
        "standard": peak_tolerance + (1 - peak_tolerance) * 0.75,
        "standard mfcc": 100,
        "accompaniment": 1,
        "pitch contour on vocals": 100,
    }
    default_threshold = 0.99

    candidates = []
    for candidate_index, (
        mfcc_cosine_score,
        mfcc_mse_score,
        composite_score,
        raw_cosine_score,
        signal_scores,
    ) in scores_at_index.items():
        passes = False

        passes = passes or mfcc_mse_score >= max_mfcc_mse_score * thresholds.get(
            "mfcc_mse_score", default_threshold
        )
        passes = passes or mfcc_cosine_score >= max_mfcc_cosine_score * thresholds.get(
            "mfcc_cosine_score", default_threshold
        )
        passes = passes or composite_score >= max_composite_score * thresholds.get(
            "composite_score", default_threshold
        )

        for signal, correlation_score in signal_scores.items():
            threshold = thresholds.get(signal, default_threshold)
            max_correlation_score = max_scores[signal]
            passes = passes or correlation_score >= max_correlation_score * threshold

        if passes:
            try:
                candidates.append(
                    (
                        candidate_index,
                        mfcc_cosine_score * signal_scores.get("standard", 1),
                    )
                )
            except Exception as e:
                print("huh?", signal_scores.keys())
                raise (e)

    # Helps us examine how the system is perceiving candidate starting locations
    plot_candidate_starting_locations(
        reaction,
        peak_indices,
        scores_at_index,
        candidates,
        thresholds,
        max_scores,
        open_start,
        closed_start,
    )

    candidates.sort(key=lambda x: x[1], reverse=True)

    return [c[0] for c in candidates]


# This function still needs to be refactored
def plot_candidate_starting_locations(
    reaction,
    peak_indices,
    scores_at_index,
    candidates,
    thresholds,
    max_scores,
    open_start,
    closed_start,
):
    has_point_of_interest = False

    points_of_interest = [
        # (abs(closed_start - 146.9 * sr) < 2 * sr and open_start < 600 * sr),
        # (abs(closed_start - 146.9 * sr) < 2 * sr and open_start < 600 * sr),
        # (abs(closed_start - 149 * sr) < 1 * sr and open_start > 620 * sr and open_start < 650 * sr),
        # (abs(closed_start - 33.6 * sr) < 2 * sr) # and open_start < 400 * sr),
        # (abs(closed_start - 35 * sr) < 2 * sr and open_start < 105 * sr),  # ian diazapam; should generate 127
        # (abs(closed_start - 42 * sr) < 2 * sr and open_start < 138 * sr),  # ian diazapam; should generate 220
        # (abs(closed_start - 60 * sr) < 3 * sr and open_start < 240 * sr),  # ian diazapam; should generate 278
        # (abs(closed_start - 72 * sr) < 3 * sr and open_start < 295 * sr),  # ian diazapam; should generate 326
        # (abs(closed_start - 89 * sr) < 4 * sr and open_start < 348 * sr),  # ian diazapam; should generate 403
        # (abs(closed_start - 109 * sr) < 4 * sr and open_start < 428 * sr),  # ian diazapam; should generate 481
        # (abs(closed_start - 148 * sr) < 5 * sr and open_start < 525 * sr),  # ian diazapam; should generate 548
        # (abs(closed_start - 164 * sr) < 5 * sr and open_start < 568 * sr),  # ian diazapam; should generate 597
        # (abs(closed_start - 169 * sr) < 6 * sr and open_start < 607 * sr),  # ian diazapam; should generate 621
        # (abs(closed_start - 194 * sr) < 6 * sr and open_start < 651 * sr),  # ian diazapam; should generate 661
        # (abs(closed_start - 146.9 * sr) < 2 * sr and open_start < 624 * sr), # dicodec genesis; should generate 633.6
        # (abs(closed_start - 148 * sr) < 2 * sr and open_start < 634 * sr), # dicodec genesis; should generate 649
        # (abs(closed_start - 145 * sr) < 2 * sr and open_start < 284 * sr), # thatsnotactingeither suicide; should generate 450
        # (abs(closed_start - 1.2 * sr) < .15 * sr and open_start < 53 * sr), # dan wheeler crutch; should generate continuity 52.4
        # (abs(closed_start - 50 * sr) < .15 * sr and open_start < 135 * sr), # dan wheeler crutch; should generate continuity 52.4
    ]
    for pi in points_of_interest:
        has_point_of_interest = has_point_of_interest or pi

    GENERATE_FOR_MAKING_OF = False

    if not has_point_of_interest and not GENERATE_FOR_MAKING_OF:
        return

    hop_length = conf.get("hop_length")

    metrics = ("mfcc_mse_score", "mfcc_cosine_score", "composite_score")
    signals = peak_indices.keys()
    num_metrics = len(metrics)
    num_signals = len(signals)

    if GENERATE_FOR_MAKING_OF:
        num_metrics = 0
        num_signals = 1

    screen_width = get_screen_width()
    screen_height = get_screen_height()

    if screen_width:
        width_per_plot = 0.9 * screen_width / max(num_signals, num_metrics)
    else:
        width_per_plot = 5  # Default if unable to determine screen width

    total_width = 20  # max(num_signals, num_metrics) * width_per_plot

    total_height = 12  # 0.9 * screen_height / max(num_signals, num_metrics)

    fig = plt.figure(figsize=(total_width, total_height))
    plt.style.use("dark_background")

    fig.patch.set_facecolor((0, 0, 0))
    fig.tight_layout()

    fig.suptitle(f"For song start @ {closed_start / sr}")

    all_candidate_indicies = scores_at_index.keys()

    # Iterate through signals and plot them
    for idx, (signal, (candidate_indicies, correlation, __)) in enumerate(peak_indices.items()):
        if GENERATE_FOR_MAKING_OF and idx > 0:
            continue

        ax = plt.subplot(
            2 if not GENERATE_FOR_MAKING_OF else 4,
            max(num_signals, num_metrics),
            idx + 1,
        )
        ax.set_facecolor((0, 0, 0))

        if not GENERATE_FOR_MAKING_OF:
            ax.set_title(signal)

        if "mfcc" in signal:
            x_values = np.arange(len(correlation)) * hop_length / sr + open_start / sr

        else:
            x_values = [(x + open_start) / sr for x in range(len(correlation))]

        if not GENERATE_FOR_MAKING_OF:
            plt.scatter(
                [c / sr + open_start / sr + 3 for c in all_candidate_indicies],
                [scores_at_index[c][4][signal] for c in all_candidate_indicies],
                color="blue",
            )
            plt.scatter(
                [c[1] / sr + open_start / sr + 3 for c in candidate_indicies],
                [scores_at_index[c[1]][4][signal] for c in candidate_indicies],
                color="purple",
            )

            plt.scatter(
                [c[0] / sr + open_start / sr + 3 for c in candidates],
                [scores_at_index[c[0]][4][signal] for c in candidates],
                color="green",
            )

        for c in candidates:
            print(
                "candidate: ",
                c[0] / sr + open_start / sr,
                scores_at_index[c[0]][4][signal],
            )
        plt.plot(
            x_values,
            correlation,
            linewidth=0.25,
            color=(97 / 255, 126 / 255, 252 / 255),
        )

        if not GENERATE_FOR_MAKING_OF:
            plt.axhline(y=thresholds[signal], color="b", linestyle="--")

        plt.yticks([])

        major_ticks = plt.gca().get_xticks()
        x_labels = [seconds_to_timestamp(x) for x in major_ticks]
        plt.xticks(ticks=major_ticks, labels=x_labels)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.xlabel("Reaction Time (s)", fontsize="x-large", labelpad=24)
        plt.ylabel("Correlation", fontsize="x-large", labelpad=24)
        if not GENERATE_FOR_MAKING_OF:
            plt.grid(True, color=(0.35, 0.35, 0.35))  # Adds grid lines

        plt.xlim(left=open_start / sr, right=x_values[-1])

    # if full_search:
    #     plt.subplot(2, 3, 2)
    #     plt.title("Aggregate MFCC Correlation")
    #     x_values = np.arange(len(normalized_aggregate_mfcc_correlation)) * hop_length / sr + open_start / sr
    #     plt.plot(x_values, normalized_aggregate_mfcc_correlation)
    #     plt.scatter(x_values[peak_mfcc_indices], [normalized_aggregate_mfcc_correlation[p] for p in peak_mfcc_indices], color='red')
    #     plt.scatter(x_values[adjusted_mfcc_peaks_for_plot], [normalized_aggregate_mfcc_correlation[p] for p in adjusted_mfcc_peaks_for_plot], color='green')

    # Iterate through metrics and plot them

    if not GENERATE_FOR_MAKING_OF:
        for idx, metric in enumerate(metrics):
            ax = plt.subplot(
                2,
                max(num_signals, num_metrics),
                max(num_signals, num_metrics) + idx + 1,
            )

            ax.set_facecolor((0, 0, 0))

            ax.set_title(metric)

            plt.scatter(
                [c / sr + open_start / sr for c in all_candidate_indicies],
                [scores_at_index[c][idx] for c in all_candidate_indicies],
                color="red",
            )

            plt.scatter(
                [c[0] / sr + open_start / sr for c in candidates],
                [scores_at_index[c[0]][idx] for c in candidates],
                color="green",
            )

            plt.xlim(left=open_start / sr, right=x_values[-1])
            plt.ylim(bottom=0)

            plt.xlabel("Time (s)")
            plt.grid(True)  # Adds grid lines
    else:
        # reaction audio waveform
        ax = plt.subplot(
            4,
            1,
            2,
        )
        x_values = [(x) / sr for x in range(len(reaction.get("reaction_audio_data")))]

        plt.yticks([])
        plt.plot(x_values, reaction.get("reaction_audio_data"), linewidth=0.025)

        major_ticks = plt.gca().get_xticks()
        x_labels = [seconds_to_timestamp(x) for x in major_ticks]
        plt.xticks(ticks=major_ticks, labels=x_labels)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        reaction_time = x_values[-1]
        plt.xlim(left=0 / sr, right=reaction_time)

        # song audio waveform
        ax = plt.subplot(
            4,
            1,
            3,
        )

        x_values = [(x) / sr for x in range(conf.get("song_length"))]

        plt.yticks([])
        plt.plot(
            x_values,
            conf.get("song_audio_data"),
            linewidth=0.025,
            color=(166 / 255, 142 / 255, 213 / 255),
        )

        major_ticks = plt.gca().get_xticks()
        x_labels = [seconds_to_timestamp(x) for x in major_ticks]
        plt.xticks(ticks=major_ticks, labels=x_labels)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.xlim(left=0 / sr, right=reaction_time)
        plt.ylim(top=1.75, bottom=-1.75)

        # 3-min chunk song audio waveform
        ax = plt.subplot(
            4,
            1,
            4,
        )

        song_audio = conf.get("song_audio_data")
        chunk_audio = song_audio[sr : 4 * sr]
        x_values = [(x) / sr for x in range(len(chunk_audio))]

        plt.yticks([])
        plt.plot(
            x_values,
            chunk_audio,
            linewidth=0.025,
            color=(166 / 255, 142 / 255, 213 / 255),
        )

        major_ticks = plt.gca().get_xticks()
        x_labels = [seconds_to_timestamp(x) for x in major_ticks]
        plt.xticks(ticks=major_ticks, labels=x_labels)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.xlim(left=-1, right=4)
        plt.ylim(top=0.5, bottom=-0.5)

        plt.savefig("cross-correlation.pdf")

    plt.show()


def seconds_to_timestamp(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    if remaining_seconds == 0:
        padded_secs = "00"
    else:
        padded_secs = f"{remaining_seconds:02}"

    return f"{int(minutes)}:{padded_secs}".split(".")[0]


from screeninfo import get_monitors


def get_screen_width():
    monitors = get_monitors()
    if monitors:
        return monitors[0].width
    return None


def get_screen_height():
    monitors = get_monitors()
    if monitors:
        return monitors[0].height
    return None
