import os, math
from utilities import conversion_audio_sample_rate as sr
from utilities import conf

from scipy.signal import correlate
import numpy as np
import matplotlib.pyplot as plt

from aligner.scoring_and_similarity import (
    get_segment_mfcc_cosine_similarity_score,
)

from aligner.align_by_audio.find_segment_end import find_segment_end


def find_best_intercept(
    reaction,
    intercepts,
    base_start,
    base_end,
    include_cross_correlation=False,
    reaction_start=None,
    reaction_end=None,
    print_intercepts=False,
    visualize=False,
):
    unique_intercepts = {}  # y-intercepts as in y=ax+b, solving for b when a=1

    if include_cross_correlation:
        # intercept_range = (min(intercepts), max(intercepts))

        song_data = conf.get("song_audio_data")
        reaction_data = reaction.get("reaction_audio_data")
        song_segment = song_data[base_start:base_end]

        width = base_end - base_start

        search_start = max(0, reaction_start - width)
        search_end = min(len(reaction_data), reaction_end + width)
        reaction_segment = reaction_data[search_start:search_end]

        # Perform the correlation
        cross_corr = correlate(reaction_segment, song_segment, mode="valid")

        search_window = int(sr / 2)

        # Calculate the start and end index in cross_corr corresponding to reaction_start and reaction_end
        corr_search_start = min(reaction_start, width) - search_window
        corr_search_end = corr_search_start + 2 * search_window

        # Ensure the bounds are within the length of the cross_corr array
        corr_search_start = max(0, corr_search_start)
        corr_search_end = min(len(cross_corr) - 1, corr_search_end)

        # print(f"{corr_search_start/sr}-{corr_search_end/sr} (prev={(corr_search_start + (base_end - base_start) + search_window)/sr}) ({len(cross_corr)/sr})")

        # Find the maximum correlation value within the constrained window
        if corr_search_end >= corr_search_start:
            windowed_corr_max_index = (
                np.argmax(cross_corr[corr_search_start:corr_search_end]) + corr_search_start
            )

            # Calculate the starting index of the best match within the original reaction_data
            corr_reaction_start = search_start + windowed_corr_max_index

            # Ensure that corr_reaction_start is within the desired range
            # Since we are bounding our argmax, this should inherently be the case.
            corr_intercept = corr_reaction_start - base_start

            # print('corr', base_start / sr, base_end / sr, reaction_start/sr, reaction_end/sr, corr_reaction_start / sr, corr_intercept / sr, offset/sr, reaction_start <= corr_reaction_start <= reaction_end)
            assert (
                reaction_start <= corr_reaction_start <= reaction_end,
                f"{reaction_start/sr} <= {corr_reaction_start/sr} <= {reaction_end/sr}",
            )

            intercepts.append(corr_intercept)

    for b in intercepts:
        # if (
        #     not include_cross_correlation
        #     or reaction_start - 0.05 * sr <= b + base_start <= reaction_end + 0.05 * sr
        # ):

        unique_intercepts[b] = True
    #     # else:
    #     #     print('intr', base_start / sr, base_end / sr, reaction_start/sr, reaction_end/sr, (b + base_start) / sr, b/sr, reaction_start <= b + base_start <= reaction_end)

    #     #     print("FILTERED!")

    best_line_def = None
    best_line_def_score = -1
    best_intercept = None
    for intercept in unique_intercepts.keys():
        intercept = int(intercept)

        candidate_line_def = [
            intercept + base_start,
            intercept + base_end,
            base_start,
            base_end,
        ]

        score = get_segment_mfcc_cosine_similarity_score(reaction, candidate_line_def)
        if include_cross_correlation and print_intercepts:
            if intercept == corr_intercept:
                print(
                    "*SCOR",
                    score,
                    base_start / sr,
                    base_end / sr,
                    reaction_start / sr,
                    reaction_end / sr,
                    (intercept + base_start) / sr,
                    intercept / sr,
                )
            else:
                print(
                    " SCOR",
                    score,
                    base_start / sr,
                    base_end / sr,
                    reaction_start / sr,
                    reaction_end / sr,
                    (intercept + base_start) / sr,
                    intercept / sr,
                )

        if score > best_line_def_score:
            best_line_def_score = score
            best_line_def = candidate_line_def
            best_intercept = intercept

    if visualize:
        scores = []
        for intercept in unique_intercepts.keys():
            candidate_line_def = [
                intercept + base_start,
                intercept + base_end,
                base_start,
                base_end,
            ]
            score = get_segment_mfcc_cosine_similarity_score(reaction, candidate_line_def)
            scores.append(score)

        plot_scores_of_intercepts(
            list(unique_intercepts.keys()),
            scores,
            corr_intercept if include_cross_correlation else None,
            base_start,
            base_end,
        )

    # print(f"\t{base_start / sr} - {base_end/sr}", best_intercept, len(intercepts) )

    return int(best_intercept)


def plot_scores_of_intercepts(intercepts, scores, corr_intercept, base_start, base_end):
    plt.figure(figsize=(10, 6))

    for intercept, score in zip(intercepts, scores):
        if intercept == corr_intercept:
            plt.scatter(
                intercept, score, c="red", marker="*", s=100
            )  # Mark the correlation-derived intercept in red
        else:
            plt.scatter(intercept, score, c="blue", marker="o")

    plt.title(f"Scores of Intercepts (base_start: {base_start/sr}, base_end: {base_end/sr})")
    plt.xlabel("Intercepts")
    plt.ylabel("Scores")
    plt.grid(True)
    plt.show()


def sharpen_segment_intercept(reaction, segment, padding=None, apply_changes=True):
    ###################################
    # if we have an imprecise segment composed of strokes that weren't exactly aligned,
    # we'll want to find the best line through them
    # if segment.get('imprecise', False):
    # print("FINDING BEST INTERCEPT IN SHARPEN SEGMENTS")

    reaction_len = len(reaction.get("reaction_audio_data"))

    if padding is None:
        padding = int(sr / 2)
    int_reaction_start = max(segment["end_points"][0] - padding, 0)
    int_reaction_end = min(segment["end_points"][1] + padding, reaction_len)

    stroke_intercepts = {s[0] - s[2]: True for s in segment["strokes"]}

    intercept = find_best_intercept(
        reaction,
        list(stroke_intercepts.keys()),
        segment["end_points"][2],
        segment["end_points"][3],
        include_cross_correlation=True,
        reaction_start=int_reaction_start,
        reaction_end=int_reaction_end,
    )

    base_start, base_end = segment["end_points"][2:4]

    new_end_points = [
        intercept + base_start,
        intercept + base_end,
        base_start,
        base_end,
    ]

    if apply_changes:
        segment["old_end_points"] = segment["end_points"]

        segment["end_points"] = new_end_points
    else:
        return intercept, new_end_points


def sharpen_segment_endpoints(reaction, segment, step, padding):
    reaction_start, reaction_end, base_start, base_end = segment["end_points"]

    #########################################
    # Now we're going to try to sharpen up the endpoint

    if conf.get("song_length") - base_end > 4 * sr:  # don't sharpen endpoints when at the end
        heat = [0 for i in range(0, int((base_end - base_start) / step))]

        for stroke in segment["strokes"]:
            (__, __, stroke_start, stroke_end) = stroke
            position = stroke_start
            while position + step < stroke_end:
                idx = int((position - base_start) / step)
                heat[idx] += 1
                position += step

        highest_idx = -1
        highest_val = -1
        for idx, val in enumerate(heat):
            if val >= highest_val:
                highest_idx = idx
                highest_val = val

        # now we're going to use find_segment_end starting from the last local maximum
        sharpen_start = max(0, highest_idx * step - padding)

        current_start = base_start + sharpen_start
        degraded_reaction_start = reaction_start + sharpen_start

        end_segment, _, _ = find_segment_end(
            reaction, current_start, degraded_reaction_start, 0, padding
        )

        if end_segment is not None:
            new_reaction_end = end_segment[1]
            new_base_end = end_segment[3]

            if new_base_end < base_start + highest_idx * step:
                new_reaction_end = reaction_start + highest_idx * step
                new_base_end = base_start + highest_idx * step

            # cap reduction to chunk_size
            new_base_end = max(int(base_end - segment.get("chunk_size")), new_base_end)
            new_reaction_end = max(int(reaction_end - segment.get("chunk_size")), new_reaction_end)

            if new_base_end < base_end:
                segment["end_points"] = [
                    reaction_start,
                    new_reaction_end,
                    base_start,
                    new_base_end,
                ]
                segment["key"] = str(segment["end_points"])
        else:
            print("AGG! Could not find segment end")

    ####################################################
    # Now we're going to try to sharpen up the beginning
    increment = int(step / 100)
    beginning_sharpen_threshold = 0.9

    candidate_base_start = base_start
    candidate_reaction_start = reaction_start

    first_score = None
    while candidate_base_start >= 0 and candidate_base_start >= base_start - step:
        candidate_segment = (
            candidate_reaction_start,
            candidate_reaction_start + step,
            candidate_base_start,
            candidate_base_start + step,
        )

        section_score = get_segment_mfcc_cosine_similarity_score(reaction, candidate_segment)

        if first_score is None:
            first_score = section_score
        elif section_score < beginning_sharpen_threshold * first_score:
            break

        candidate_base_start -= increment
        candidate_reaction_start -= increment

    shift = base_start - (candidate_base_start + increment)
    if shift > 0:
        segment["end_points"][0] -= shift
        segment["end_points"][1] -= shift
        segment["end_points"][2] -= shift
        segment["end_points"][3] -= shift
        # print(f"Shifted by {shift / sr} to {segment['end_points'][2] / sr}")
