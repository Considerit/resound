import numpy as np
import librosa
import math
import os
import tempfile
import cv2
import imagehash
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm  # For the progress bar
from collections import defaultdict, Counter


from sklearn.neighbors import KDTree

from utilities import conversion_audio_sample_rate as sr
from utilities import (
    conf,
    read_object_from_file,
    save_object_to_file,
)

from silence import get_quiet_parts_of_song

from inventory.inventory import get_reactions_manifest
from aligner.align_by_image.frame_operations import crop_with_noise
from aligner.align_by_image.embedded_video_finder import get_bounding_box_of_music_video_in_reaction

from aligner.segment_pruner import prune_unreachable_segments
from aligner.segment_refiner import sharpen_segment_intercept

from tabulate import tabulate
import seaborn


def build_image_matches(
    reaction,
    fps=2,
    min_coverage=0.5,
    similarity_threshold=0.95,
    lag=0,
    allowed_spacing_on_ends=10,
    visualize=False,
):
    lag = int(lag)
    channel = reaction.get("channel")

    music_video_path = conf.get("base_video_path")
    reaction_video_path = reaction.get("video_path")

    hash_output_dir = os.path.join(conf.get("temp_directory"), "image_hashes")
    if not os.path.exists(hash_output_dir):
        os.makedirs(hash_output_dir)

    crop_coordinates = get_bounding_box_of_music_video_in_reaction(reaction)
    if crop_coordinates is None:
        print(f"***** {channel}: crop coordinates failed so we can't use image alignment")
        return None, None, None

    print("Found embedded music video at", crop_coordinates)

    if lag == 0:
        hash_cache_file_name = (
            f"{channel}-{fps}fps-{tuple(crop_coordinates)}-{similarity_threshold}.json"
        )
    else:
        hash_cache_file_name = (
            f"{channel}-{fps}fps-{tuple(crop_coordinates)}-{similarity_threshold}-lag{lag}.json"
        )

    reaction_crop_coordinates = crop_coordinates
    music_hashes_file_name = f"{conf.get('song_key')}-{fps}fps.pckl"

    hashes_file_name = f"{channel}-hashes-{fps}fps-{tuple(reaction_crop_coordinates)}.pckl"
    hashes_file_path = os.path.join(hash_output_dir, hashes_file_name)

    hash_cache_path = os.path.join(hash_output_dir, hash_cache_file_name)
    music_hashes_path = os.path.join(hash_output_dir, music_hashes_file_name)

    if not os.path.exists(hash_cache_path):
        # Extract frames from both videos

        if not os.path.exists(music_hashes_path):
            print("Extracting frames from the music video")
            music_frames = extract_frames(
                music_video_path, fps=fps, crop_coords=music_crop_coordinates
            )

            save_object_to_file(
                music_hashes_path,
                {
                    "phashes": calculate_phash(music_frames, hash_method="phash"),
                    "dhashes": calculate_phash(music_frames, hash_method="dhash"),
                },
            )

        if not os.path.exists(hashes_file_path):
            print("Extracting frames from the reaction video")
            reaction_frames = extract_frames(
                reaction_video_path, fps=fps, crop_coords=reaction_crop_coordinates
            )

            save_object_to_file(
                hashes_file_path,
                {
                    "phashes": calculate_phash(reaction_frames, hash_method="phash"),
                    "dhashes": calculate_phash(reaction_frames, hash_method="dhash"),
                },
            )

        music_hashes = read_object_from_file(music_hashes_path)
        reaction_hashes = read_object_from_file(hashes_file_path)

        print("Matching frames")
        phash_matches = match_frames(
            music_hashes["phashes"],
            reaction_hashes["phashes"],
            similarity_threshold=similarity_threshold,
            lag=lag,
        )
        dhash_matches = match_frames(
            music_hashes["dhashes"],
            reaction_hashes["dhashes"],
            similarity_threshold=similarity_threshold,
            lag=lag,
        )

        save_object_to_file(
            hash_cache_path,
            {"phash_matches": phash_matches, "dhash_matches": dhash_matches},
        )

    music_hashes = read_object_from_file(music_hashes_path)
    reaction_hashes = read_object_from_file(hashes_file_path)

    reaction["image_alignment_matrix"] = create_image_based_score_matrix(
        reaction, music_hashes, reaction_hashes
    )

    hash_matches = read_object_from_file(hash_cache_path)
    for k, v in hash_matches.items():
        hash_matches[k] = {float(key): value for key, value in v.items()}

    matches_by_music = {}
    for match_group in hash_matches.values():
        for music_time, reaction_times in match_group.items():
            if music_time not in matches_by_music:
                matches_by_music[music_time] = []
            matches_by_music[music_time] += reaction_times

    matches = []
    for mt, reaction_times in matches_by_music.items():
        for rt in reaction_times:
            yint = rt - mt
            if 0 <= yint:
                matches.append([mt, rt, yint, True])  # True means not rejected

    # Refine the matches
    while True:
        num_rejected = filter_matches_by_intercept(matches, fps=fps)
        # print("num_rejected by intercept", num_rejected)

        if num_rejected == 0:
            break

    if len([m for m in matches if m[3]]) == 0:
        print(f"**** {channel}: Failed to get reliable image alignment (matches=0)")

        del reaction["image_alignment_matrix"]
        return None, None, None

    stutters, stutter_assignments = label_stutters(matches, fps=fps)

    # Create segments from the matches
    segments = create_segments_from_matches(matches, stutter_assignments, spacing=1 / fps)

    # Calculate coverage of the song by segments. If coverage is too little, then
    # we consider image alignment to have failed and be too unreliable with this
    # reaction.
    coverage = segment_coverage_of_song(segments, fps=fps)

    # Get the end points (and filter out segments outside those bounds)
    if coverage >= 1.0:
        segments, reaction_start, reaction_end = find_reaction_endpoints(
            segments, max_dist_from_end=allowed_spacing_on_ends
        )

    normalized_segments = normalize_segment_format_with_audio_segments(segments, fps)

    if coverage >= 1:
        prune_unreachable_segments(
            reaction,
            normalized_segments,
            allowed_spacing=5 * sr,
            allowed_spacing_on_ends=allowed_spacing_on_ends * sr,
        )

    # Account for A/V lag
    detected_lag = get_lag(reaction, normalized_segments)
    if detected_lag != 0:
        print(f"DETECTED {detected_lag / sr}s A/V LAG for {channel}, trying to adjust...")
        print("\n")
        return build_image_matches(
            reaction,
            fps=fps,
            min_coverage=min_coverage,
            similarity_threshold=similarity_threshold,
            lag=detected_lag + lag,
            visualize=visualize,
        )

    if visualize:
        show_image_based_score_matrix(reaction, filter_to_segments=normalized_segments)
        show_image_based_score_matrix(reaction)

    for segment in normalized_segments:
        sharpen_segment_intercept(reaction, segment, padding=sr, use_image_scores=False)

    coverage = segment_coverage_of_song(segments, fps=fps)

    if coverage < min_coverage:
        print(f"**** {channel}: Failed to get reliable image alignment (coverage={coverage})")
        del reaction["image_alignment_matrix"]
        return None, None, None
    else:
        print(f"**** {channel}: Segment coverage of image alignment is {coverage}")

    plot_matches(reaction, segments, matches, normalized_segments, visualize=visualize)

    segments_to_pass_along = [s for s in normalized_segments if not s.get("pruned")]

    if coverage >= 1:
        return (
            segments_to_pass_along,
            int((reaction_start or 0) * sr),
            int((reaction_end or conf.get("song_length")) * sr),
        )
    else:
        return segments_to_pass_along, None, None


def create_image_based_score_matrix(reaction, music_hashes, reaction_hashes):
    music_times = [x[0] for x in music_hashes["phashes"]]
    reaction_times = [x[0] for x in reaction_hashes["phashes"]]

    scores = {}

    for mt, music_time in enumerate(music_times):
        music_dhash = music_hashes["dhashes"][mt][1]
        music_phash = music_hashes["phashes"][mt][1]
        for rt, reaction_time in enumerate(reaction_times):
            reaction_dhash = reaction_hashes["dhashes"][rt][1]
            reaction_phash = reaction_hashes["phashes"][rt][1]

            # Calculate perceptual hash similarity
            distance = music_phash - reaction_phash
            psimilarity = 1 - (distance / len(music_phash.hash) ** 2)

            # Calculate difference hash similarity
            distance = music_dhash - reaction_dhash
            dsimilarity = 1 - (distance / len(music_dhash.hash) ** 2)

            # Create a key for the current time pair
            key = str((int(sr * music_time), int(sr * reaction_time)))
            # Store the average of perceptual and difference similarities
            scores[key] = ((psimilarity + dsimilarity) / 2, psimilarity, dsimilarity)

    music_times = [int(sr * x[0]) for x in music_hashes["phashes"]]
    reaction_times = [int(sr * x[0]) for x in reaction_hashes["phashes"]]

    return scores, music_times, reaction_times


def show_image_based_score_matrix(reaction, filter_to_segments=None):
    scores, music_times, reaction_times = reaction["image_alignment_matrix"]

    # Create a 2D matrix for heatmap visualization
    score_matrix = np.zeros((len(reaction_times), len(music_times)))

    if filter_to_segments is None:
        for mt, music_time in enumerate(music_times):
            for rt, reaction_time in enumerate(reaction_times):
                key = str((music_time, reaction_time))
                if key in scores:
                    score_matrix[len(reaction_times) - 1 - rt, mt] = scores[key][
                        0
                    ]  # Average similarity score
                else:
                    score_matrix[len(reaction_times) - 1 - rt, mt] = np.nan  # No score available

    else:
        for segment in filter_to_segments:
            if not segment.get("pruned"):
                mt = 2 * int(segment["end_points"][2] / sr)
                rt = 2 * int(segment["end_points"][0] / sr)
                my_scores = []
                while mt <= 2 * int(segment["end_points"][3] / sr):
                    key = str((int(mt / 2 * sr), int(rt / 2 * sr)))
                    if key in scores:
                        score_matrix[len(reaction_times) - 1 - rt, mt] = scores[key][0]
                        my_scores.append(scores[key][0])
                    else:
                        print(key, "key not in scores...")
                    mt += 2
                    rt += 2
                print(np.mean(np.array(my_scores)), my_scores)

    # Set up the plot
    plt.figure(figsize=(12, 8))
    ax = seaborn.heatmap(
        score_matrix,
        cmap="magma",
        vmin=0,
        vmax=1,
        cbar=False,
        # xticklabels=music_times,
        # yticklabels=reaction_times,
        # linewidths=0.5,
        # linecolor="black",
        square=False,
        mask=np.isnan(score_matrix),
    )

    ax.set_xlabel("Music Time")
    ax.set_ylabel("Reaction Time")
    ax.set_title(f"{reaction.get('channel')}: Segment-Specific Scores Heatmap")

    plt.show()


# Find average segment adjustments for LOUD parts of the song. Are they consistent?
# If so, we can say that the avg represents A/V lag, and adjust all segments by the
# same amount.
def get_lag(reaction, segments, max_lag=2):
    quiet_parts = get_quiet_parts_of_song()

    changes = []
    for segment in segments:
        current_start, current_end = segment.get("end_points")[2:4]

        if (current_end - current_start) / sr < 10:
            continue

        total_in_quiet = 0

        for qs, qe in quiet_parts:
            qs *= sr
            qe *= sr

            if max(current_start, qs) <= min(current_end, qe):
                total_in_quiet += min(current_end, qe) - max(current_start, qs)

        if total_in_quiet / (current_end - current_start) > 0.3:
            continue

        adjusted_intercept, adjusted_end_points = sharpen_segment_intercept(
            reaction,
            segment,
            apply_changes=False,
            padding=int(2 * max_lag * sr),
            use_image_scores=False,
        )
        change = adjusted_intercept - segment.get("y-intercept")
        if change / sr <= max_lag:  # filter outliers
            changes.append(change)

    if len(changes) == 0:
        print("DETECT LAG: Could not find any suitable segments")
        return 0

    changes = np.array(changes)
    median = np.median(changes)
    min_change = np.min(changes)
    max_change = np.max(changes)

    if abs(median / sr) > 0.1 and min_change * max_change > 0:
        lag = median
    else:
        lag = 0

    # print("lags", lag / sr, [ch / sr for ch in changes])
    return lag


def normalize_segment_format_with_audio_segments(segments, fps):
    normalized_segments = []
    chunk_size = 1 / fps

    for segment in segments:
        music_start = segment.get("min_music_time")
        music_end = segment.get("max_music_time")
        reaction_start = segment.get("y-intercept") + music_start
        reaction_end = reaction_start + (music_end - music_start)

        normie = {
            "y-intercept": int(sr * segment.get("y-intercept")),
            "end_points": [
                int(sr * reaction_start),
                int(sr * reaction_end),
                int(sr * music_start),
                int(sr * music_end),
            ],
            "strokes": [
                [
                    int(sr * (yint + mt)),
                    int(sr * (yint + mt + chunk_size)),
                    int(sr * mt),
                    int(sr * (mt + chunk_size)),
                ]
                for (mt, rt, yint, valid) in segment["matches"]
                if valid
            ],
            "pruned": False,
            "source": "image-alignment",
            "chunk_size": chunk_size,
        }
        if "backfilled_by" in segment:
            normie["backfilled_by"] = segment.get("backfilled_by")

        normie["key"] = str(normie["end_points"])
        normalized_segments.append(normie)

    return normalized_segments


def label_stutters(matches, fps, visualize=False):
    x_min = np.min([m[0] for m in matches])
    x_max = np.max([m[0] for m in matches])
    y_min = np.min([m[1] for m in matches])
    y_max = np.max([m[1] for m in matches])

    spacing = 1 / fps

    matrix = create_stutter_matrix(matches, x_min, x_max, y_min, y_max, spacing)

    # print("Detecting stutters")
    stutters = detect_stutters(matrix)

    stutter_assignments = assign_stutters(matches, stutters, spacing, x_min, y_min)
    if visualize:
        plot_points_and_stutters(matches, stutters, stutter_assignments, spacing, y_min, x_min)

    return stutters, stutter_assignments


def assign_stutters(matches, stutters, spacing, x_min, y_min):
    # Mark which points belong to each stutter
    assignments = {}

    for stutter_idx, stutter in enumerate(stutters):
        (start_x, start_y), width, height = stutter
        for match in matches:
            x, y, y_intercept, valid = match
            if valid:
                x_idx = int((x - x_min) / spacing)
                y_idx = int((y - y_min) / spacing)
                # Check if the point belongs to the current stutter
                if start_x <= x_idx < start_x + width and start_y <= y_idx < start_y + height:
                    assignments[str(match)] = stutter_idx
    return assignments


def check_expansion(
    matrix, i, j, width, height, direction, threshold, visited, min_height, min_width
):
    """
    Check if expansion in the given direction (left, right, up, down) is valid.
    It ensures the following:
    - No visited cells in the newly expanded row or column.
    - Expanded columns must have at least min_height occupied cells.
    - Expanded rows must have at least min_width occupied cells.
    - Expanded columns/rows must satisfy the threshold compared to the adjacent column/row.

    Returns the number of non-empty cells that would be added if expanded.
    """

    def get_slices(i, j, width, height, direction):
        """
        Returns slices for the new dimension and adjacent dimension based on the direction of expansion.
        """
        if direction == "right":
            return (slice(i, i + height), slice(j + width, j + width + 1)), (
                slice(i, i + height),
                slice(j + width - 1, j + width),
            )
        elif direction == "down":
            return (slice(i + height, i + height + 1), slice(j, j + width)), (
                slice(i + height - 1, i + height),
                slice(j, j + width),
            )
        elif direction == "left":
            return (slice(i, i + height), slice(j - 1, j)), (slice(i, i + height), slice(j, j + 1))
        elif direction == "up":
            return (slice(i - 1, i), slice(j, j + width)), (slice(i, i + 1), slice(j, j + width))

    def validate_expansion(matrix, visited, new_slice, adjacent_slice, min_cells, threshold):
        """
        Validates whether the new expansion meets the conditions.
        """
        new_dim = matrix[new_slice]
        adjacent_dim = matrix[adjacent_slice]

        # Check for visited cells
        if np.count_nonzero(visited[new_slice]) > 0:
            return 0

        # Check if the new dimension meets the minimum number of occupied cells
        new_occupied = np.count_nonzero(new_dim)
        if new_occupied < min_cells:
            return 0

        # Check if the new dimension satisfies the threshold relative to the adjacent dimension
        adjacent_occupied = np.count_nonzero(adjacent_dim)
        if new_occupied < threshold * adjacent_occupied:
            return 0

        return new_occupied

    # Get the slices based on the direction
    new_slice, adjacent_slice = get_slices(i, j, width, height, direction)

    # Determine min_cells based on the direction
    if direction in ["right", "left"]:
        min_cells = min_height  # We are expanding a column
    else:
        min_cells = min_width  # We are expanding a row

    # Validate the expansion using the common logic
    new_occupied = validate_expansion(
        matrix, visited, new_slice, adjacent_slice, min_cells, threshold
    )

    return new_occupied


def detect_stutters(matrix, min_width=1, min_height=7, threshold=0.6):
    """
    Detect stutters of points in the matrix that are at least min_size x min_size
    and expand them to the maximal size while ensuring each row and column has
    no more than the threshold of empty cells.
    """
    stutters = []
    rows, cols = matrix.shape
    visited = np.zeros((rows, cols), dtype=int)  # Track visited cells

    # Iterate over the matrix to find potential stutter starting points
    for i in range(rows - min_height + 1):
        for j in range(cols - min_width + 1):
            if visited[i, j]:
                continue  # Skip already visited cells that are part of previous stutters

            if matrix[i, j] == 0:
                continue  # Skip cells that are empty

            # Start with a 3x3 stutter
            width = min_width
            height = min_height

            # Confirm that this seed stutter is full.
            # Allow no empties in the initial 3x3 block.
            submatrix = matrix[i : i + height, j : j + width]
            total_cells = width * height
            non_empty_cells = np.count_nonzero(submatrix)
            if non_empty_cells < total_cells:
                continue
            if np.count_nonzero(visited[i : i + height, j : j + width]) > 0:
                continue

            # Expand the stutter in all directions while maintaining the threshold per row and per column
            while True:
                best_expansion_direction = None
                max_non_empty_added = 0

                # Check all four directions: left, right, up, down
                for direction in ["right", "down", "left", "up"]:
                    non_empty_added = check_expansion(
                        matrix,
                        i,
                        j,
                        width,
                        height,
                        direction,
                        threshold,
                        visited,
                        min_height,
                        min_width,
                    )
                    if non_empty_added > max_non_empty_added:
                        max_non_empty_added = non_empty_added
                        best_expansion_direction = direction

                # Break if no more expansions can be made
                if best_expansion_direction is None:
                    break

                # Apply the best expansion direction
                if best_expansion_direction == "right":
                    width += 1
                elif best_expansion_direction == "down":
                    height += 1
                elif best_expansion_direction == "left":
                    j -= 1
                    width += 1
                elif best_expansion_direction == "up":
                    i -= 1
                    height += 1

            # Mark the current cells as part of a stutter (visited)
            visited[i : i + height, j : j + width] = 1

            # Once expansion is done, record the stutter (starting position, width, height)
            stutters.append(((j, i), width, height))

    return stutters


def create_stutter_matrix(points, x_min, x_max, y_min, y_max, spacing):
    """
    Create a matrix based on points (x, y_intercept) and count the number of points in each cell.
    """
    # Determine the size of the matrix based on spacing and the range of x, y_intercept values
    x_range = int((x_max - x_min) / spacing) + 1
    y_range = int((y_max - y_min) / spacing) + 1

    # Initialize matrix with zeros
    matrix = np.zeros((y_range, x_range), dtype=int)

    # Populate the matrix with point counts
    for x, y, __, valid in points:
        if valid:
            i = int((x - x_min) / spacing)  # x-index
            j = int((y - y_min) / spacing)  # y-intercept index
            matrix[j, i] += 1

    return matrix


def plot_points_and_stutters(points, stutters, stutter_assignments, spacing, y_min, x_min):
    """
    Visualize points and color-code them based on which stutter they belong to.

    Parameters:
    - points: List of (x, y, y_intercept, valid) tuples
    - stutters: List of stutters as ((x_start, y_start), width, height)
    - spacing: Spacing of points in x and y dimensions
    - y_int_min: Minimum y_intercept (used to index matrix rows)
    - x_min: Minimum x value (used to index matrix columns)
    """
    # Set up color map to assign different colors to each stutter
    stutter_colors = list(mcolors.TABLEAU_COLORS.values())
    non_stutter_color = "gray"  # Color for points not in any stutter

    plt.figure(figsize=(10, 8))

    # Plot points, color-coding them by stutter
    for point_idx, match in enumerate(points):
        x, y, y_intercept, valid = match
        if valid:
            stutter_idx = stutter_assignments.get(str(match), -1)
            color = (
                stutter_colors[stutter_idx % len(stutter_colors)]
                if stutter_idx != -1
                else non_stutter_color
            )
            plt.scatter(
                x,
                y,
                color=color,
                label=f"stutter {stutters[stutter_idx]}" if stutter_idx != -1 else "No stutter",
                alpha=0.4 if stutter_idx == -1 else 1,
                s=2 if stutter_idx == -1 else 3,
            )

    plt.xlabel("x (time in first video)")
    plt.ylabel("y_intercept (time in second video)")
    plt.title("2D Points Color-Coded by stutters")

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.show()


def segment_coverage_of_song(segments, fps):
    song_length = conf.get("song_length") / sr

    checks = 0
    covered = 0

    for t in range(0, int(song_length), 4):
        for s in segments:
            if s["min_music_time"] <= t <= s["max_music_time"]:
                covered += 1
                break

        checks += 1

    return covered / checks


def find_reaction_endpoints(segments, max_dist_from_end=10):
    song_length = conf.get("song_length") / sr

    # Identify reaction_start and reaction_end.
    reaction_start = None
    start_intercept = None
    reaction_end = None
    end_intercept = None

    for segment in segments:
        y_int = segment["y-intercept"]
        min_t = segment["min_music_time"]
        max_t = segment["max_music_time"]

        if min_t - max_dist_from_end <= 0 and (not reaction_start or start_intercept > y_int):
            start_intercept = y_int
            reaction_start = segment

        if max_t + max_dist_from_end >= song_length and (not reaction_end or end_intercept < y_int):
            end_intercept = y_int
            reaction_end = segment

    ################
    # Fill to start or end if necessary
    # if reaction_start:
    #     min_t = reaction_start["min_music_time"]
    #     if min_t > 1:
    #         reaction_start["backfilled_by"] = min_t

    #     reaction_start["min_music_time"] = 0

    # if reaction_end:
    #     max_t = reaction_end["max_music_time"]
    #     if song_length - max_t > 1:
    #         reaction_end["backfilled_by"] = song_length - max_t
    #     reaction_end["max_music_time"] = song_length

    # Filter out segments that aren't within the new bounds
    if reaction_start and reaction_end:
        print("Got reaction start and end:", start_intercept, end_intercept + song_length)
        segments = [
            s
            for s in segments
            if s["y-intercept"] >= start_intercept and s["y-intercept"] <= end_intercept
        ]

    return segments, start_intercept, None if not end_intercept else end_intercept + song_length


def match_frames(
    music_hashes, reaction_hashes, lag=0, similarity_threshold=0.95, at_most_x_per_reaction_time=5
):
    """
    Match frames between the music video and the reaction video based on perceptual hashes.

    :param music_hashes: List of (timestamp, hash) from the music video.
    :param reaction_hashes: List of (timestamp, hash) from the reaction video.
    :param similarity_threshold: Similarity threshold (0.0 to 1.0) to consider a match.
    :return: A dictionary where the key is the music video timestamp, and the value is a list of reaction timestamps.
    """
    matches = defaultdict(list)
    all_similarities = {}
    similarity_by_reaction_time = defaultdict(list)

    for music_time, music_hash in music_hashes:
        similarities = []
        for reaction_time, reaction_hash in reaction_hashes:
            delagged_reaction_time = reaction_time + lag / sr
            # Compute similarity as inverse of Hamming distance
            distance = music_hash - reaction_hash
            similarity = 1 - (distance / len(music_hash.hash) ** 2)

            similarities.append((delagged_reaction_time, similarity))

            similarity_by_reaction_time[delagged_reaction_time].append((music_time, similarity))

        all_similarities[music_time] = similarities

    # build a histogram of reaction times, and exclude the most popular because it
    # is likely a big cause of false positives, which lead to false negatives on the
    # correct frames.
    black_list = {}
    # Only allow the top 2 * at_most_x_per_reaction_time matches for each reaction_time
    for reaction_time, matches_for_reaction_time in similarity_by_reaction_time.items():
        if len(matches_for_reaction_time) > 4 * at_most_x_per_reaction_time:
            matches_for_reaction_time.sort(key=lambda x: x[1], reverse=True)
            for music_time, similarity in matches_for_reaction_time[
                4 * at_most_x_per_reaction_time :
            ]:
                black_list[f"{reaction_time}-{music_time}"] = True

    # print(f"BLACKLISTED {len(black_list.keys())}")

    reaction_time_histogram = defaultdict(list)
    for music_time, similarities in all_similarities.items():
        filtered_similarities = [
            (reaction_time, similarity)
            for reaction_time, similarity in similarities
            if f"{reaction_time}-{music_time}" not in black_list
        ]
        # print(len(filtered_similarities))

        if len(filtered_similarities) == 0:
            continue

        max_similarity = max([sim for _, sim in filtered_similarities])

        # Save reaction times that are within the threshold of the max similarity
        for reaction_time, similarity in filtered_similarities:
            if similarity >= similarity_threshold * max_similarity:
                reaction_time_histogram[reaction_time].append((music_time, similarity))

    # Blacklisting repeat high-similarity matching reaction times
    for reaction_time, matches_for_reaction_time in reaction_time_histogram.items():
        if len(matches_for_reaction_time) > at_most_x_per_reaction_time:
            matches_for_reaction_time.sort(key=lambda x: x[1], reverse=True)
            for music_time, similarity in matches_for_reaction_time[at_most_x_per_reaction_time:]:
                black_list[f"{reaction_time}-{music_time}"] = True

    # print(f"BLACKLISTED {len(black_list.keys())}")
    for music_time, similarities in all_similarities.items():
        filtered_similarities = [
            (reaction_time, similarity)
            for reaction_time, similarity in similarities
            if f"{reaction_time}-{music_time}" not in black_list
        ]
        # print(len(filtered_similarities))

        if len(filtered_similarities) == 0:
            continue

        max_similarity = max([sim for _, sim in filtered_similarities])

        # Save reaction times that are within the threshold of the max similarity
        for reaction_time, similarity in filtered_similarities:
            if similarity >= similarity_threshold * max_similarity:
                if music_time not in matches:
                    matches[music_time] = []
                matches[music_time].append(reaction_time)

    return matches


def filter_matches_by_intercept(matches, fps, time_window=8, intercept_tolerance=0.1, density=0.25):
    """
    Filters out (music_vid_time, reaction_vid_time) matches that do not form a line
    with some other match within a certain distance and with a slope near 1.

    :param matches: List of tuples (music_vid_time, reaction_vid_time).
    :param time_window: The window of time (in seconds) to look for nearby points.
    :param intercept_tolerance: The allowed deviation from a slope of 1.
    :param density: The % of peers in region that must form a slope of 1.
    :return: Filtered list of matches.
    """

    matches = [m for m in matches if m[3]]

    if len(matches) == 0:
        return 0

    num_samples_in_window = time_window * fps
    peers_sharing_intercept = int(num_samples_in_window * density)

    music_times = np.array([m[0] for m in matches])
    reaction_times = np.array([m[1] for m in matches])

    # print(f"Filtering {len(matches)} matches by slope")

    # Build a KDTree for efficient neighbor search
    points = np.vstack([music_times, reaction_times]).T

    tree = KDTree(points)

    num_rejected = 0
    # Loop through each point
    for i, (m1, r1, intercept1, __) in enumerate(matches):
        # Query neighbors within the time_window around the current point (m1, r1)
        indices = tree.query_radius([[m1, r1]], r=time_window / 2)[0]

        # Count valid neighbors based on intercept similarity
        valid_neighbors = 0
        for j in indices:
            if i == j or not matches[j][3]:
                continue  # Skip self-comparison or comparison to rejects

            m2, r2, intercept2, __ = matches[j]

            # Check if the intercept difference is within tolerance
            if abs(intercept1 - intercept2) <= intercept_tolerance:
                valid_neighbors += 1

        # If we didn't find enough valid neighbors with the correct slope, reject this point
        if valid_neighbors < peers_sharing_intercept:
            matches[i][3] = False
            num_rejected += 1

    return num_rejected


def create_segments_from_matches(
    all_matches,
    stutter_assignments,
    spacing,
    time_constraint=6,
    intercept_min_spacing=3,
    min_samples=4,
    min_length=3,
    max_stuttered=0.8,
    min_coverage=0.2,
    tabulate=False,
):
    """
    Cluster matches into segments based on the y-intercept with a time constraint.

    :param matches: List of tuples (music_time, reaction_time, rejected).
    :param time_constraint: Maximum allowed time gap between matches in music_time.
    :param intercept_min_spacing: Maximum allowed distance between intercepts for DBSCAN clustering.
    :param min_samples: Minimum number of matches to form a valid cluster.
    :return: List of valid clusters, each containing matches that satisfy the constraints.
    """

    matches = [m for m in all_matches if m[3]]

    matches_by_intercept = {}
    for m in matches:
        yint = round(m[2])
        if yint not in matches_by_intercept:
            matches_by_intercept[yint] = []
        already_in = False
        for m2 in matches_by_intercept[yint]:
            if m2[0] == m[0] and m2[1] == m[1]:
                already_in = True
                m[3] = False
                break
        if not already_in:
            matches_by_intercept[yint].append(m)

    initial_segments = []
    for matches_for_yint in matches_by_intercept.values():
        current_cluster = []

        matches_for_yint.sort(key=lambda x: x[0])
        last_music_time = matches_for_yint[0][0]

        for match in matches_for_yint:
            if abs(match[0] - last_music_time) > time_constraint:
                initial_segments.append(current_cluster)
                current_cluster = []
            current_cluster.append(match)
            last_music_time = match[0]
        initial_segments.append(current_cluster)

    segments = []
    for s in initial_segments:
        non_stuttered_matches = [m for m in s if str(m) not in stutter_assignments]
        if len(non_stuttered_matches) >= min_samples:
            segments.append(
                {
                    "y-intercept": np.median(np.array([m[2] for m in s])),
                    "matches": s,
                    "min_music_time": np.min(np.array([m[0] for m in s])),
                    "max_music_time": np.max(np.array([m[0] for m in s])),
                }
            )

    while True:
        changes_made = 0
        segments.sort(key=lambda x: len(x["matches"]), reverse=True)

        while True:
            num_absorbed = 0
            for i, s in enumerate(segments):
                for j, s2 in enumerate(segments):
                    if i >= j:
                        continue
                    if (
                        abs(s["y-intercept"] - s2["y-intercept"]) <= intercept_min_spacing
                        and max(s["min_music_time"], s2["min_music_time"])
                        <= min(s["max_music_time"], s2["max_music_time"]) + 1
                    ) or (
                        abs(s["y-intercept"] - s2["y-intercept"]) <= 0.5
                        and max(s["min_music_time"], s2["min_music_time"])
                        <= min(s["max_music_time"], s2["max_music_time"]) + time_constraint
                    ):
                        s["min_music_time"] = min(s["min_music_time"], s2["min_music_time"])
                        s["max_music_time"] = max(s["max_music_time"], s2["max_music_time"])
                        s["matches"] += s2["matches"]
                        num_absorbed += 1
                        segments.pop(j)
                        break
                if num_absorbed > 0:
                    break
            changes_made += num_absorbed
            if num_absorbed == 0:
                break

        # See if any stray unreliable matches might fit with a segment
        unreliable_matches = [m for m in all_matches if not m[3]]
        matches_by_intercept = defaultdict(list)
        segments_by_intercept = defaultdict(list)

        for m in unreliable_matches:
            yint = round(m[2])
            matches_by_intercept[yint].append(m)

        while True:
            absorbtions = 0
            for s in segments:
                yint = s["y-intercept"]

                for myint in range(
                    math.floor(yint - intercept_min_spacing / 2),
                    math.ceil(yint + intercept_min_spacing / 2),
                ):
                    for m in matches_by_intercept[myint]:
                        if (
                            not m[3]
                            and abs(m[2] - yint) <= intercept_min_spacing / 2
                            and max(s["min_music_time"], m[0])
                            <= min(s["max_music_time"], m[0]) + time_constraint
                        ):
                            s["min_music_time"] = min(s["min_music_time"], m[0])
                            s["max_music_time"] = max(s["max_music_time"], m[0])
                            m[3] = True
                            s["matches"].append(m)
                            absorbtions += 1
            # print(f"Absorbed {absorbtions} unreliable matches!")
            changes_made += absorbtions
            if absorbtions == 0:
                break

        if changes_made == 0:
            break

    # filter segments for quality
    quality_segments = []
    evaluations = []

    for s in segments:
        # ...by length
        length = s["max_music_time"] - s["min_music_time"]

        # ...by % in stutter...
        in_stutters = [m for m in s["matches"] if stutter_assignments.get(str(m), False)]
        stuttered = len(in_stutters) / len(s["matches"])

        # ...by coverage
        unique_music_times = {
            m[0]: True for m in s["matches"] if not stutter_assignments.get(str(m), False)
        }

        full_coverage = length / spacing + 1

        coverage = len(unique_music_times.keys()) / full_coverage

        overall = coverage * math.log(length + 1, 2)

        evaluations.append((s, overall, coverage, length, stuttered))

    evaluations.sort(key=lambda x: x[1])

    if tabulate:
        table_data = []
        headers = [
            "Status",
            "Y-Intercept",
            "Music Time",
            "Quality Score",
            "Length",
            "% Stuttered",
            "% Coverage",
        ]

        for s, quality_score, coverage, length, stuttered in evaluations:
            row = [
                "KEPT"
                if quality_score >= 1.5
                and coverage >= min_coverage
                and length >= min_length
                and stuttered <= max_stuttered
                else "FILTERED",
                s["y-intercept"],
                f"{s['min_music_time']}-{s['max_music_time']}",
                f"{quality_score:.2f}",
                f"{length:.2f}",
                f"{stuttered:.2%}",
                f"{coverage:.2%}",
            ]
            table_data.append(row)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    for s, quality_score, coverage, length, stuttered in evaluations:
        if (
            quality_score >= 1.5
            and coverage >= min_coverage
            and length >= min_length
            and stuttered <= max_stuttered
        ):
            quality_segments.append(s)
        else:
            for m in s["matches"]:
                m[3] = False
            changes_made += 1

    quality_segments

    return quality_segments


def get_frame_sample(fps, video_path=None, video=None):
    assert video_path is not None or video is not None
    if video is None:
        video = cv2.VideoCapture(video_path)

    # Get video properties
    total_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / total_fps  # total duration in seconds

    # Frame interval based on target FPS
    frame_interval = int(total_fps / fps)

    # Create list of frame indices to extract
    frame_indices = list(range(0, total_frames, frame_interval))

    return frame_indices, total_fps


def extract_frames(video_path, fps=4, crop_coords=None):
    """
    Extract frames from a video at the given FPS, optionally cropping each frame.

    :param video_path: Path to the video file.
    :param fps: Frames per second to extract.
    :param crop_coords: Optional tuple (top, left, width, height) for cropping frames.
    :return: List of tuples (timestamp, frame) where each frame is a resized PIL image.
    """

    # print(f"Cropping {video_path} with {crop_coords}")
    video = cv2.VideoCapture(video_path)

    frames = []

    frame_indices, total_fps = get_frame_sample(fps=fps, video=video)

    for frame_idx in tqdm(frame_indices, desc="Extracting frames", ncols=100):
        # Set the video position to the desired frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()

        if not ret:
            break  # Break if reading the frame fails

        timestamp = frame_idx / total_fps

        if crop_coords:
            frame = crop_with_noise(frame, crop_coords)

        # Convert to grayscale and resize for normalization
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (256, 256))
        frame_pil = Image.fromarray(frame)

        frames.append((timestamp, frame_pil))

    video.release()
    return frames


def calculate_phash(frames, hash_method="dhash"):
    """
    Calculate perceptual hash for each frame using different hash methods.
    :param frames: List of (timestamp, frame) where frame is a PIL image.
    :param hash_method: 'phash', 'dhash', or 'ahash'.
    :return: List of (timestamp, hash) where hash is the perceptual hash of the frame.
    """
    phashes = []
    for timestamp, frame in frames:
        if hash_method == "phash":
            phash = imagehash.phash(frame)
        elif hash_method == "dhash":
            phash = imagehash.dhash(frame)
        elif hash_method == "ahash":
            phash = imagehash.average_hash(frame)
        else:
            raise ValueError("Unknown hash method")
        phashes.append((timestamp, phash))
    return phashes


def plot_matches(reaction, segments, matches, normalized_segments, visualize=False):
    """
    Plot the matched music and reaction video frames on a 2D plot.

    :param matches: A dictionary where the key is the music video timestamp, and the value is a list of reaction timestamps.
    """

    fig = plt.figure(figsize=(10, 6))
    plt.style.use("dark_background")
    fig.patch.set_facecolor((0, 0, 0))

    # matches.reverse()

    # for music_times, reaction_times in matches:
    #     plt.scatter(music_times, reaction_times, s=5)

    all_rejects = [m for m in matches if not m[3]]

    plt.scatter(
        [r[0] for r in all_rejects], [r[1] for r in all_rejects], s=1, alpha=0.7, color="white"
    )

    for segment in segments:
        plt.scatter(
            [r[0] for r in segment["matches"] if r[3]],
            [r[1] for r in segment["matches"] if r[3]],
            s=2,
            alpha=0.5,
        )
    segments.sort(key=lambda x: x["y-intercept"])
    for segment in segments:
        if segment.get("y-intercept"):
            x_values = [segment["min_music_time"], segment["max_music_time"]]
            y_values = [
                x_values[0] + segment["y-intercept"],
                x_values[1] + segment["y-intercept"],
            ]
            # print(
            #     "Segment:",
            #     segment["y-intercept"],
            #     segment["min_music_time"],
            #     segment["max_music_time"],
            # )
            plt.plot(x_values, y_values, linestyle="-", lw=1, color="red")

    for segment in normalized_segments:
        if not segment.get("pruned", False):
            x_values = [segment["end_points"][2] / sr, segment["end_points"][3] / sr]
            y_values = [segment["end_points"][0] / sr, segment["end_points"][1] / sr]
            # print(
            #     "NORMALIZED Segment:",
            #     (segment["end_points"][0] - segment["end_points"][2]) / sr,
            #     segment["end_points"][1] / sr,
            #     segment["end_points"][2] / sr,
            # )
            plt.plot(x_values, y_values, linestyle="-", lw=1, color="green")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.title(f"Image-based Alignment for {reaction.get('channel')}")
    plt.xlabel("Music Video Time (s)")
    plt.ylabel("Reaction Video Time (s)")

    plt.ylim(bottom=0)
    plt.xlim(left=0)

    plt.grid(True, color=(0.35, 0.35, 0.35))

    if visualize:
        plt.show()

    plot_fname = os.path.join(
        conf.get("temp_directory"), f"{reaction.get('channel')}-image-alignment-painting.png"
    )
    plt.savefig(plot_fname, dpi=300)
