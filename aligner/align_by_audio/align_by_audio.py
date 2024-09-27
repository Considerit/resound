import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math

# from silence import is_silent

from utilities import conversion_audio_sample_rate as sr
from utilities import conf, print_profiling, save_object_to_file, read_object_from_file
from aligner.align_by_audio.bounds import get_bound, in_bounds, create_reaction_alignment_bounds
from aligner.align_by_audio.find_segment_start import (
    find_segment_starts,
    score_start_candidates,
    correct_peak_index,
    seconds_to_timestamp,
    initialize_segment_start_cache,
)
from aligner.align_by_audio.find_segment_end import find_segment_end, initialize_segment_end_cache
from aligner.scoring_and_similarity import (
    find_best_path,
    path_score,
    print_path,
    initialize_path_score,
    initialize_segment_tracking,
)

from aligner.path_finder import construct_all_paths, initialize_paint_caches
from aligner.path_refiner import sharpen_path_boundaries
from aligner.segment_pruner import prune_unreachable_segments, prune_poor_segments
from aligner.segment_consolidizer import consolidate_segments, bridge_gaps
from aligner.segment_refiner import (
    sharpen_segment_intercept,
    sharpen_segment_endpoints,
    find_best_intercept,
)
from aligner.align_by_audio.pruning_search import initialize_path_pruning

from aligner.visualize_alignment import (
    splay_paint,
    compile_images_to_video,
    GENERATE_FULL_ALIGNMENT_VIDEO,
)


attempts_progression = {
    "chunk_size": [3, 3, 5, 5, 8, 8, 10, 10, 12, 12, 3, 5, 8, 10, 12],
    "allowed_spacing": [3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 12, 12, 12, 12, 12],
}


def initialize_caches():
    initialize_segment_end_cache()
    initialize_path_score()
    initialize_segment_tracking()
    initialize_paint_caches()
    initialize_path_pruning()
    initialize_segment_start_cache()


def paint_paths(
    reaction,
    seed_segments=None,
    peak_tolerance=0.4,
    allowed_spacing=None,
    allowed_spacing_on_ends=10 * sr,
    attempts=0,
):
    initialize_caches()

    chunk_size = get_chunk_size(reaction, attempts=attempts)
    if allowed_spacing is None:
        allowed_spacing = attempts_progression["allowed_spacing"][attempts]
        # print(f"Attempts={attempts}   {attempts % 3}    {allowed_spacing}")
        allowed_spacing *= sr

    allowed_spacing_on_ends = max(allowed_spacing, allowed_spacing_on_ends)

    print(
        f"\n###############################\n# {conf.get('song_key')} / {reaction.get('channel')}"
    )
    print(f"Painting path with chunk size {chunk_size / sr} and {allowed_spacing / sr} spacing")

    step = int(0.5 * sr)

    start = time.perf_counter()

    print("FIND SEGMENTS")

    segments = find_segments(
        reaction,
        chunk_size,
        step,
        peak_tolerance,
        save_to_file=GENERATE_FULL_ALIGNMENT_VIDEO,
    )

    if seed_segments is not None:
        segments += seed_segments

    splay_paint(
        reaction,
        segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="original_segments",
    )

    print("ABSORB")
    consolidated_segments = consolidate_segments(segments)
    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="consolidate-segments-1",
    )

    consolidated_segments = bridge_gaps(consolidated_segments)

    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="bridge-gaps-1",
    )

    print("PRUNE UNREACHABLE")

    pruned_segments, __ = prune_unreachable_segments(
        reaction, consolidated_segments, allowed_spacing, allowed_spacing_on_ends, prune_links=False
    )

    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="prune-unreachable-1",
    )

    print("SHARPEN INTERCEPT")
    sharpen_intercept(reaction, pruned_segments)

    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="sharpen-intercepts-1",
    )

    print("ABSORB2")
    pruned_segments = consolidate_segments(pruned_segments)
    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="consolidate-segments-2",
    )

    pruned_segments = bridge_gaps(pruned_segments)

    # for seg in pruned_segments:
    #     rs,re,bs,be = seg.get('end_points')
    #     print(f"[{rs-bs}] {bs/sr}-{be/sr} / {rs/sr}-{re/sr}   [{seg.get('pruned')}]")

    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="bridge-gaps-2",
    )

    print("PRUNING POOR SEGMENTS")

    pruned_segments = prune_poor_segments(reaction, pruned_segments)

    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="prune-poor-segments",
    )

    # print("SHARPEN ENDPOINTS")

    # sharpen_endpoints(
    #     reaction,
    #     chunk_size,
    #     step,
    #     pruned_segments,
    # )

    # splay_paint(
    #     reaction,
    #     consolidated_segments,
    #     stroke_alpha=1,
    #     show_live=False,
    #     chunk_size=chunk_size,
    #     id="sharpen-endpoints",
    # )

    print("SHARPEN INTERCEPT2")

    sharpen_intercept(reaction, pruned_segments)

    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="sharpen-intercepts-2",
    )

    print("ABSORB3")
    pruned_segments = consolidate_segments(pruned_segments)
    pruned_segments = bridge_gaps(pruned_segments)

    # for seg in pruned_segments:
    #     rs,re,bs,be = seg.get('end_points')
    #     print(f"[{rs-bs}] {bs/sr}-{be/sr} / {rs/sr}-{re/sr}   [{seg.get('pruned')}]")

    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="consolidate-segments-3",
    )

    print("PRUNE UNREACHABLE2")

    pruned_segments, joinable_segment_map = prune_unreachable_segments(
        reaction, pruned_segments, allowed_spacing, allowed_spacing_on_ends, prune_links=False
    )

    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="prune-unreachable-2",
    )

    print(f"Constructing paths from {len(pruned_segments)} viable segments")

    segments_by_key = {}
    for segment in pruned_segments:
        segments_by_key[segment["key"]] = segment

    paths = construct_all_paths(
        reaction,
        pruned_segments,
        joinable_segment_map,
        allowed_spacing,
        allowed_spacing_on_ends,
        chunk_size,
        segments_by_key,
    )

    if len(paths) == 0 and attempts < len(attempts_progression["chunk_size"]) - 1:
        print(f"No paths found.")

        if not GENERATE_FULL_ALIGNMENT_VIDEO:
            splay_paint(
                reaction,
                consolidated_segments,
                stroke_alpha=1,
                show_live=False,
                chunk_size=chunk_size,
                id="end-failed",
            )

        if seed_segments is not None:
            for s in seed_segments:
                s["pruned"] = False

        return paint_paths(reaction, seed_segments, peak_tolerance, attempts=attempts + 1)

    print(f"Found {len(paths)} paths")

    best_path = find_best_path(reaction, paths, segments_by_key)

    # micro_aligned = micro_align_path(reaction, best_path, segments_by_key)

    # best_path = find_best_path(reaction, [best_path, micro_aligned])

    # last_segment = None
    # error_found = False
    # for segment in best_path:
    #     reaction_start, reaction_end, music_start, music_end = segment[:4]

    #     if last_segment:
    #         if music_start < last_segment[3]:
    #             print("\tERROR! BEND detected", last_segment, segment)
    #             error_found = True
    #     last_segment = segment
    # if error_found:
    #     print(best_path)

    # if the last segment of the best path is fill, replace it by extending the last good segment
    if best_path[-1][4]:
        filled = best_path.pop()
        fill_len = filled[1] - filled[0]
        best_path[-1][3] += fill_len
        best_path[-1][1] += fill_len

    alignment_duration = (time.perf_counter() - start) / 60  # in minutes

    splay_paint(
        reaction,
        consolidated_segments,
        stroke_alpha=1,
        stroke_color="gray",
        show_live=False,
        best_path=best_path,
        chunk_size=chunk_size,
        id="finished",
    )

    if GENERATE_FULL_ALIGNMENT_VIDEO:
        compile_images_to_video(get_paint_portfolio_path(reaction, chunk_size), "video.mp4", FPS=5)

    best_path_score = path_score(best_path, reaction, segments_by_key)
    best_path_output = print_path(best_path, reaction, segments_by_key).get_string()

    return best_path, best_path_score, best_path_output


def get_chunk_size(reaction, attempts=0):
    chunk_size = max(reaction.get("chunk_size", 0), attempts_progression["chunk_size"][attempts])
    chunk_size *= sr
    chunk_size = int(chunk_size)
    return chunk_size


import matplotlib.colors as mcolors


def visualize_clusters(clusters, base_audio_len, reaction_audio_len):
    # Create a colormap from red (0) to green (1)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "green"])

    # Initialize the plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Iterate over each cluster
    for cluster in clusters:
        # Get the best score within the cluster for color scaling
        best_score = max(segment["score"] for segment in cluster)

        # Plot each segment in the cluster
        for segment in cluster:
            base_start, base_end = segment["end_points"][2], segment["end_points"][3]
            reaction_start, reaction_end = (
                segment["end_points"][0],
                segment["end_points"][1],
            )
            score = segment["score"]
            is_pruned = segment.get("pruned", False)

            # Scale the color by the score relative to the best score in the cluster
            color = cmap(score / best_score)
            line_style = "--" if is_pruned else "-"

            # Draw the segment
            ax.plot(
                [base_start, base_end],
                [reaction_start, reaction_end],
                linestyle=line_style,
                color=color,
                linewidth=2,
            )

    # Set plot limits and labels
    ax.set_xlim([0, base_audio_len])
    ax.set_ylim([0, reaction_audio_len])
    ax.set_xlabel("Position in Base Audio")
    ax.set_ylabel("Position in Reaction Audio")

    # Show the plot
    plt.show()


#######


def sharpen_intercept(reaction, segments, use_image_scores=False):
    for segment in segments:
        if segment.get("pruned", False):
            continue

        sharpen_segment_intercept(reaction, segment, padding=int(sr / 2), use_image_scores=False)


def sharpen_endpoints(reaction, chunk_size, step, segments):
    for segment in segments:
        if segment.get("pruned", False):
            continue

        sharpen_segment_endpoints(reaction, segment, step=step, padding=chunk_size)


def get_audio_signals(reaction, start, reaction_start, chunk_size):
    base_audio = conf.get("song_audio_data")
    reaction_audio = reaction.get("reaction_audio_data")

    hop_length = conf.get("hop_length")
    reaction_audio_mfcc = reaction.get("reaction_audio_mfcc")
    song_audio_mfcc = conf.get("song_audio_mfcc")

    chunk = base_audio[start : start + chunk_size]
    chunk_mfcc = song_audio_mfcc[
        :, round(start / hop_length) : round((start + chunk_size) / hop_length)
    ]

    open_chunk = reaction_audio[reaction_start:]
    open_chunk_mfcc = reaction_audio_mfcc[:, round(reaction_start / hop_length) :]

    # predominantly_silent = is_silent(chunk, threshold_db=-40)

    signals = {
        "standard": (1, chunk, open_chunk),
        "standard mfcc": (hop_length, chunk_mfcc, open_chunk_mfcc),
    }

    # we'll also send the accompaniment if it isn't vocally dominated
    if False and predominantly_silent:
        base_audio_accompaniment = conf.get("song_audio_accompaniment_data")
        reaction_audio_accompaniment = reaction.get("reaction_audio_accompaniment_data")

        reaction_audio_accompaniment_mfcc = reaction.get("reaction_audio_accompaniment_mfcc")
        base_audio_accompaniment_mfcc = conf.get("song_audio_accompaniment_mfcc")

        print("SILENT!", start / sr)
        chunk = base_audio_accompaniment[start : start + chunk_size]
        open_chunk = reaction_audio_accompaniment[reaction_start:]

        chunk_mfcc = base_audio_accompaniment_mfcc[
            :, round(start / hop_length) : round((start + chunk_size) / hop_length)
        ]
        open_chunk_mfcc = reaction_audio_accompaniment_mfcc[:, round(reaction_start / hop_length) :]

        signals["accompaniment"] = (hop_length, chunk_mfcc, open_chunk_mfcc)
        signals["accompaniment mfcc"] = (hop_length, chunk_mfcc, open_chunk_mfcc)

        evaluate_with = "accompaniment"
    else:
        evaluate_with = "standard"

    return signals, evaluate_with


def get_candidate_starts(
    reaction,
    signals,
    peak_tolerance,
    open_start,
    closed_start,
    chunk_size,
    distance,
    upper_bound,
    evaluate_with="standard",
):
    peak_indices = {}
    for signal, (hop_length, chunk, open_chunk) in signals.items():
        new_candidates = find_segment_starts(
            signal=signal,
            reaction=reaction,
            open_chunk=open_chunk,
            closed_chunk=chunk,
            current_chunk_size=chunk_size,
            peak_tolerance=peak_tolerance,
            open_start=open_start,
            closed_start=closed_start,
            distance=1 * sr,
            upper_bound=upper_bound,
            hop_length=hop_length,
        )

        if new_candidates is not None:
            peak_indices[signal] = new_candidates

    candidates = score_start_candidates(
        reaction=reaction,
        signals=signals,
        peak_indices=peak_indices,
        open_chunk=signals[evaluate_with][2],
        closed_chunk=signals[evaluate_with][1],
        open_chunk_mfcc=signals[f"{evaluate_with} mfcc"][2],
        closed_chunk_mfcc=signals[f"{evaluate_with} mfcc"][1],
        current_chunk_size=chunk_size,
        peak_tolerance=peak_tolerance,
        open_start=open_start,
        closed_start=closed_start,
    )
    return candidates


def get_lower_bounds(reaction, chunk_size):
    manual_bounds = reaction.get("manual_bounds", None)
    lower_bounds = []
    if reaction.get("start_reaction_search_at", None):
        lower_bounds.append((0, reaction.get("start_reaction_search_at")))

    if manual_bounds:
        for mbound in manual_bounds:
            ts, upper = mbound
            lower = upper - 1.5
            ts = int(ts * sr)
            lower = int(lower * sr)  # + chunk_size / 2)
            lower_bounds.append((ts, lower))
    lower_bounds.reverse()
    return lower_bounds


def find_segments(reaction, chunk_size, step, peak_tolerance, save_to_file=False):
    base_audio = conf.get("song_audio_data")
    reaction_audio = reaction.get("reaction_audio_data")

    starting_points = range(0, len(base_audio) - chunk_size, step)

    start_reaction_search_at = int(reaction.get("start_reaction_search_at"))

    minimums = [start_reaction_search_at]

    # Factor in manually configured bounds
    lower_bounds = get_lower_bounds(reaction, chunk_size)

    seg_cache_key = f"{minimums[0]}-{chunk_size}-{conf['first_n_samples']}-{reaction.get('unreliable_bounds','')}-{len(lower_bounds)}-{reaction.get('end_reaction_search_at',0)}-{reaction.get('start_reaction_search_at',0)}-{int(peak_tolerance*100)}"

    strokes = []
    active_strokes = []

    cache_dir = os.path.join(conf.get("song_directory"), "_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file_name = f"{reaction.get('channel')}-start_cache-{seg_cache_key}.json"
    candidate_cache_file = os.path.join(cache_dir, cache_file_name)

    print(f"LOOKING FOR {cache_file_name} {os.path.exists(candidate_cache_file)}")

    if os.path.exists(candidate_cache_file):
        candidate_cache = read_object_from_file(candidate_cache_file)
    else:
        candidate_cache = {}

    alignment_bounds = create_reaction_alignment_bounds(reaction, conf["first_n_samples"])

    reaction_span = reaction.get("end_reaction_search_at", len(reaction_audio))

    splay_paint(
        reaction,
        [],
        stroke_alpha=1,
        show_live=False,
        chunk_size=chunk_size,
        id="bounds-only",
    )

    all_candidates = []  # used only for painting full alignment
    for i, start in enumerate(starting_points):
        print(f"\tStroking...{i/len(starting_points)*100:.2f}%", end="\r")

        print_profiling()

        latest_lower_bound = 0
        for lower_bound_ts, lower_bound in lower_bounds:
            if lower_bound_ts <= start:
                latest_lower_bound = lower_bound + (start - lower_bound_ts)
                break

        candidate_lower_bound = max(min(minimums[-18:]), latest_lower_bound, chunk_size)

        assert candidate_lower_bound >= latest_lower_bound

        reaction_start = max(minimums[0] + start, candidate_lower_bound - chunk_size)

        upper_bound = len(reaction_audio) - len(base_audio)
        if str(start) not in candidate_cache:
            # print(f"START NOT IN CACHE {str(start) in candidate_cache} {start} {str(start)}")

            upper_bound = get_bound(
                alignment_bounds, start, reaction_span - (len(base_audio) - start)
            )
            # upper_bound = len(reaction_audio)  # comment this in if you want an unbounded painting (e.g. for the making-of video)

            signals, evaluate_with = get_audio_signals(reaction, start, reaction_start, chunk_size)
            candidates = get_candidate_starts(
                reaction,
                signals,
                peak_tolerance,
                reaction_start,
                start,
                chunk_size,
                1 * sr,
                upper_bound,
                evaluate_with=evaluate_with,
            )

            candidates = [
                c + reaction_start for c in candidates if c + reaction_start >= latest_lower_bound
            ]
            candidates.sort()

            candidate_cache[start] = candidates

            if len(candidates) == 0:
                continue

        else:
            candidates = candidate_cache[str(start)]

        for c in candidates:
            assert (c >= latest_lower_bound, (c - start) / sr)
            assert (c <= upper_bound, (c - start) / sr)

        if len(candidates) > 0:
            minimums.append(candidates[0])
        else:
            minimums.append(0)
        still_active_strokes = []

        already_matched = {}
        active_strokes.sort(key=lambda x: x["end_points"][2])
        for segment in active_strokes:
            best_match = None
            best_match_overlap = None

            for reaction_start in candidates:
                if reaction_start in already_matched:
                    continue

                reaction_end = reaction_start + chunk_size

                music_start = start
                music_end = start + chunk_size

                new_stroke = (reaction_start, reaction_end, music_start, music_end)

                overlap = are_continuous_or_overlap(new_stroke, segment["end_points"])
                if overlap is not None:
                    if best_match is None or best_match_overlap > overlap:
                        best_match = new_stroke
                        best_match_overlap = overlap

                    if overlap == 0:
                        break

            if best_match is not None:
                new_stroke = best_match
                segment["end_points"][3] = new_stroke[3]
                segment["end_points"][1] = (
                    segment["end_points"][0] + segment["end_points"][3] - segment["end_points"][2]
                )
                segment["strokes"].append(new_stroke)

                if best_match_overlap != 0:
                    segment["imprecise"] = True

                already_matched[new_stroke[0]] = True

        for reaction_start in candidates:
            all_candidates.append(
                {
                    "pruned": False,
                    "end_points": [
                        reaction_start,
                        reaction_start + chunk_size,
                        start,
                        start + chunk_size,
                    ],
                    "strokes": [
                        [
                            reaction_start,
                            reaction_start + chunk_size,
                            start,
                            start + chunk_size,
                        ]
                    ],
                }
            )

        if save_to_file or i == len(starting_points) - 1:
            splay_paint(
                reaction,
                strokes=all_candidates,
                stroke_alpha=step / chunk_size,
                stroke_color="blue",
                show_live=False,
                chunk_size=chunk_size,
                id=f"stroke-{start}",
                copy_to_main=False,
            )

        for reaction_start in candidates:
            if reaction_start in already_matched:
                continue

            reaction_end = reaction_start + chunk_size

            music_start = start
            music_end = start + chunk_size

            new_stroke = (reaction_start, reaction_end, music_start, music_end)

            # create a new segment
            segment = {
                "end_points": [reaction_start, reaction_end, music_start, music_end],
                "strokes": [new_stroke],
                "pruned": False,
                "source": "audio-alignment",
                "chunk_size": chunk_size,
            }
            strokes.append(segment)
            still_active_strokes.append(segment)

        for stroke in active_strokes:
            if stroke["end_points"][3] >= start - chunk_size:
                still_active_strokes.append(stroke)
        active_strokes = still_active_strokes

        # if save_to_file:
        #     print(step / chunk_size)
        #     splay_paint(
        #         reaction,
        #         strokes,
        #         stroke_alpha=2 * step / chunk_size,
        #         show_live=False,
        #         chunk_size=chunk_size,
        #         id=f"{start}",
        #         copy_to_main=False,
        #     )

    save_object_to_file(candidate_cache_file, candidate_cache)

    for segment in strokes:
        segment["key"] = str(segment["end_points"])
        if len(segment["strokes"]) < 4:
            segment["pruned"] = True

    return [stroke for stroke in strokes if not stroke["pruned"]]


####################
# Utility functions


def are_continuous_or_overlap(line1, line2, epsilon=None):
    if epsilon is None:
        epsilon = (
            0.25 * sr
        )  # this should be less than the distance parameter for find_segment_starts
    cy, dy, cx, dx = line1
    ay, by, ax, bx = line2

    close_enough = ax <= cx and cx <= bx and ay <= cy and cy <= by

    # if close_enough:
    #     slope_if_continued = (dy-ay) / (dx-ax)

    #     if abs(slope_if_continued - 1) < epsilon:
    #         return abs(slope_if_continued - 1)

    if close_enough:
        intercept_1 = cy - cx
        intercept_2 = ay - ax

        if abs(intercept_1 - intercept_2) < epsilon:
            return abs(intercept_1 - intercept_2)

    return None
