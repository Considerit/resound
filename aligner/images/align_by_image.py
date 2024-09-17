import numpy as np
import librosa
import math
import os
import tempfile
import cv2
import imagehash
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm  # For the progress bar

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

from utilities import conversion_audio_sample_rate as sr
from utilities import (
    conf,
    read_object_from_file,
    save_object_to_file,
)

from inventory.inventory import get_reactions_manifest
from aligner.images.frame_operations import crop_with_noise
from aligner.images.embedded_video_finder import (
    get_bounding_box_of_music_video_in_reaction,
)


def build_image_matches(
    reaction, use_dhash=True, use_phash=True, fps=4, visualize=True
):
    channel = reaction.get("channel")

    music_video_path = conf.get("base_video_path")
    reaction_video_path = reaction.get("video_path")

    hash_output_dir = os.path.join(conf.get("temp_directory"), "image_hashes")
    if not os.path.exists(hash_output_dir):
        os.makedirs(hash_output_dir)

    crop_coordinates = get_bounding_box_of_music_video_in_reaction(reaction)
    if crop_coordinates is None:
        print(
            f"***** {channel}: crop coordinates failed so we can't use image alignment"
        )
        return

    print("Found embedded music video at", crop_coordinates)

    hash_cache_file_name = f"{channel}-{fps}fps-{tuple(crop_coordinates)}.json"

    hash_cache_path = os.path.join(hash_output_dir, hash_cache_file_name)

    if not os.path.exists(hash_cache_path):
        # Step 1: Extract frames from both videos

        print("Extracting frames from reaction")

        music_hashes_file_name = f"{conf.get('song_key')}-{fps}fps.pckl"
        music_hashes_path = os.path.join(hash_output_dir, music_hashes_file_name)
        if not os.path.exists(music_hashes_path):
            print("Extracting frames from the music video")
            music_frames = extract_frames(music_video_path, fps=fps)

            save_object_to_file(
                music_hashes_path,
                {
                    "phashes": calculate_phash(music_frames, hash_method="phash"),
                    "dhashes": calculate_phash(music_frames, hash_method="dhash"),
                },
            )

        hashes_file_name = f"{channel}-hashes-{fps}fps-{tuple(crop_coordinates)}.pckl"
        hashes_file_path = os.path.join(hash_output_dir, hashes_file_name)
        if not os.path.exists(hashes_file_path):
            print("Extracting frames from the music video")
            reaction_frames = extract_frames(
                reaction_video_path, fps=fps, crop_coords=crop_coordinates
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
            music_hashes["phashes"], reaction_hashes["phashes"]
        )
        dhash_matches = match_frames(
            music_hashes["dhashes"], reaction_hashes["dhashes"]
        )

        save_object_to_file(
            hash_cache_path,
            {"phash_matches": phash_matches, "dhash_matches": dhash_matches},
        )

    hash_matches = read_object_from_file(hash_cache_path)
    for k, v in hash_matches.items():
        hash_matches[k] = {float(key): value for key, value in v.items()}

    match_groups = []
    if use_phash:
        match_groups.append(hash_matches["phash_matches"])

    if use_dhash:
        match_groups.append(hash_matches["dhash_matches"])

    # Step 4: Refine the matches
    unzipped_hash_matches = []

    music_times = []
    reaction_times = []

    for matches in match_groups:
        for music_time, reaction_time in matches.items():
            for reaction_time in reaction_time:
                music_times.append(music_time)
                reaction_times.append(reaction_time)

    unzipped_hash_matches.append([music_times, reaction_times])

    filtered_matches = filter_matches_by_intercept(unzipped_hash_matches[0])
    filtered_first = filtered_matches[1]

    # filtered_matches = unzipped_hash_matches

    num_rejected = len(filtered_matches[1][0])
    while num_rejected > 0:
        filtered_matches = filter_matches_by_intercept(filtered_matches[0])
        num_rejected = len(filtered_matches[1][0])
        print("num_rejected", num_rejected)

    # Step 5: Create segments from the matches
    segments = create_segments_from_matches(
        list(zip(filtered_matches[0][0], filtered_matches[0][1]))
    )

    # Step 6: Plot the matches
    print("Plotting")
    if visualize:
        plot_matches(reaction, segments)

    return segments

    # alignment_video = reaction.get("aligned_path")
    # alignment_metadata_file = reaction.get("alignment_metadata")

    # metadata = read_object_from_file(alignment_metadata_file)
    # reaction.update(metadata)
    # best_path = reaction.get("best_path")


def extract_frames(video_path, fps=4, crop_coords=None):
    """
    Extract frames from a video at the given FPS, optionally cropping each frame.

    :param video_path: Path to the video file.
    :param fps: Frames per second to extract.
    :param crop_coords: Optional tuple (top, left, width, height) for cropping frames.
    :return: List of tuples (timestamp, frame) where each frame is a resized PIL image.
    """
    video = cv2.VideoCapture(video_path)
    frames = []

    # Get video properties
    total_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / total_fps  # total duration in seconds

    # Frame interval based on target FPS
    frame_interval = int(total_fps / fps)

    # Create list of frame indices to extract
    frame_indices = list(range(0, total_frames, frame_interval))

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


def match_frames(music_hashes, reaction_hashes, similarity_threshold=0.95):
    """
    Match frames between the music video and the reaction video based on perceptual hashes.

    :param music_hashes: List of (timestamp, hash) from the music video.
    :param reaction_hashes: List of (timestamp, hash) from the reaction video.
    :param similarity_threshold: Similarity threshold (0.0 to 1.0) to consider a match.
    :return: A dictionary where the key is the music video timestamp, and the value is a list of reaction timestamps.
    """
    matches = {}
    for music_time, music_hash in music_hashes:
        similarities = []
        for reaction_time, reaction_hash in reaction_hashes:
            # Compute similarity as inverse of Hamming distance
            distance = music_hash - reaction_hash
            similarity = 1 - (distance / len(music_hash.hash) ** 2)

            similarities.append((reaction_time, similarity))

        max_similarity = max([sim for _, sim in similarities])

        # Save reaction times that are within the threshold of the max similarity
        for reaction_time, similarity in similarities:
            if similarity >= similarity_threshold * max_similarity:
                if music_time not in matches:
                    matches[music_time] = []
                matches[music_time].append(reaction_time)

    return matches


def filter_matches_by_intercept(
    matches, time_window=8, intercept_tolerance=0.1, peers_sharing_intercept=8
):
    """
    Filters out (music_vid_time, reaction_vid_time) matches that do not form a line
    with some other match within a certain distance and with a slope near 1.

    :param matches: List of tuples (music_vid_time, reaction_vid_time).
    :param time_window: The window of time (in seconds) to look for nearby points.
    :param intercept_tolerance: The allowed deviation from a slope of 1.
    :param peers_sharing_intercept: The number of neighboring peers that must form a slope of 1.
    :return: Filtered list of matches.
    """

    music_times, reaction_times = np.array(matches[0]), np.array(matches[1])

    print(f"Filtering matches by slope for {len(music_times)} matches")

    # Calculate intercepts for all matches
    intercepts = reaction_times - music_times

    # Build a KDTree for efficient neighbor search
    points = np.vstack([music_times, reaction_times]).T
    tree = KDTree(points)

    # List to store filtered matches
    filtered_music_times = []
    filtered_reaction_times = []

    rejected_music_times = []
    rejected_reaction_times = []

    # Loop through each point
    for i, (m1, r1) in enumerate(zip(music_times, reaction_times)):
        intercept1 = intercepts[i]

        # Query neighbors within the time_window around the current point (m1, r1)
        indices = tree.query_radius([[m1, r1]], r=time_window / 2)[0]

        # Count valid neighbors based on intercept similarity
        found_valid_neighbor = 0
        for j in indices:
            if i == j:
                continue  # Skip self-comparison

            m2, r2 = music_times[j], reaction_times[j]
            intercept2 = intercepts[j]

            # Check if the intercept difference is within tolerance
            if abs(intercept1 - intercept2) <= intercept_tolerance:
                found_valid_neighbor += 1

        # If we found enough valid neighbors with the correct slope, keep this point
        if found_valid_neighbor >= peers_sharing_intercept:
            filtered_music_times.append(m1)
            filtered_reaction_times.append(r1)
        else:
            rejected_music_times.append(m1)
            rejected_reaction_times.append(r1)

    return [
        [filtered_music_times, filtered_reaction_times],
        [rejected_music_times, rejected_reaction_times],
    ]


def calculate_median_of_averages_intercept(cluster):
    matches_by_music_time = {}
    for match in cluster["matches"]:
        mt = match[0]
        if mt not in matches_by_music_time:
            matches_by_music_time[mt] = []
        matches_by_music_time[mt].append(match)

    mt_avg_intercepts = np.array(
        [
            np.mean(np.array([m[1] - m[0] for m in mts]))
            for mts in matches_by_music_time.values()
        ]
    )
    median_of_averages_intercept = np.median(mt_avg_intercepts)
    return median_of_averages_intercept


def create_segments_from_matches(
    matches, time_constraint=10, intercept_eps=1, min_samples=4
):
    """
    Cluster matches into segments based on the y-intercept with a time constraint.

    :param matches: List of tuples (music_time, reaction_time).
    :param time_constraint: Maximum allowed time gap between matches in music_time.
    :param intercept_eps: Maximum allowed distance between intercepts for DBSCAN clustering.
    :param min_samples: Minimum number of matches to form a valid cluster.
    :return: List of valid clusters, each containing matches that satisfy the constraints.
    """

    # Step 1: Calculate y-intercept for each match
    intercepts = np.array(
        [reaction_time - music_time for music_time, reaction_time in matches]
    )

    # Step 2: Apply DBSCAN clustering on y-intercepts
    intercept_clusterer = DBSCAN(
        eps=intercept_eps, min_samples=min_samples, metric="euclidean"
    )
    labels = intercept_clusterer.fit_predict(
        intercepts.reshape(-1, 1)
    )  # DBSCAN on 1D data (intercepts)

    # Step 3: Group matches by cluster label
    segments = {}
    for label, match in zip(labels, matches):
        if label == -1:
            continue  # Ignore noise (label -1)
        if label not in segments:
            segments[label] = {
                "y-intercept": None,
                "matches": [],
                "rejected_matches": [],
                "min_music_time": None,
                "max_music_time": None,
            }
        segments[label]["matches"].append(match)

    # Step 4: Apply time constraint to segments
    for segment in segments.values():
        # cluster the matches by music time, subject to time_constraint
        music_time_clusters = []
        current_cluster = []
        segment["matches"].sort(key=lambda x: x[0])
        last_music_time = segment["matches"][0][0]
        for match in segment["matches"]:
            if abs(match[0] - last_music_time) > time_constraint:
                music_time_clusters.append(current_cluster)
                current_cluster = []
            current_cluster.append(match)
            last_music_time = match[0]
        music_time_clusters.append(current_cluster)

        # find the biggest segment
        best_segment = []
        for music_cluster in music_time_clusters:
            if len(best_segment) < len(music_cluster):
                best_segment = music_cluster
        for music_cluster in music_time_clusters:
            if music_cluster != best_segment:
                segment["rejected_matches"] += music_cluster

        segment["matches"] = best_segment

        median_of_averages_intercept = calculate_median_of_averages_intercept(segment)

        # clustered whole segment
        filtered_cluster_matches = [segment["matches"][0]]
        for match in segment["matches"][1:]:
            if match[0] == filtered_cluster_matches[-1][0]:
                intercept = match[1] - match[0]
                last_intercept = (
                    filtered_cluster_matches[-1][1] - filtered_cluster_matches[-1][0]
                )
                if abs(last_intercept - median_of_averages_intercept) < abs(
                    intercept - median_of_averages_intercept
                ):
                    segment["rejected_matches"].append(match)
                else:
                    removed = filtered_cluster_matches.pop()
                    segment["rejected_matches"].append(removed)
                    filtered_cluster_matches.append(match)
            else:
                filtered_cluster_matches.append(match)

        segment["matches"] = filtered_cluster_matches

        # Re-calculate the dominant intercept
        intercepts = np.array(
            [
                reaction_time - music_time
                for music_time, reaction_time in segment["matches"]
            ]
        )
        median_intercept = segment["y-intercept"] = np.median(intercepts)

        music_times = [music_time for music_time, reaction_time in segment["matches"]]
        segment["min_music_time"] = min(music_times)
        segment["max_music_time"] = max(music_times)

    # Step 5: Absorb segments to get rid of inferior segments
    segments_to_keep = []

    return segments.values()


def plot_matches(reaction, segments):
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

    all_rejects = []
    for segment in segments:
        all_rejects += segment["rejected_matches"]

    plt.scatter([r[0] for r in all_rejects], [r[1] for r in all_rejects], s=3)

    for segment in segments:
        plt.scatter(
            [r[0] for r in segment["matches"]], [r[1] for r in segment["matches"]], s=5
        )

    for segment in segments:
        if segment.get("y-intercept"):
            x_values = [segment["min_music_time"], segment["max_music_time"]]
            y_values = [
                x_values[0] + segment["y-intercept"],
                x_values[1] + segment["y-intercept"],
            ]
            print(
                "Segment:",
                segment["y-intercept"],
                segment["min_music_time"],
                segment["max_music_time"],
            )
            plt.plot(x_values, y_values, linestyle="--")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.title(f"Image-based Alignment for {reaction.get('channel')}")
    plt.xlabel("Music Video Time (s)")
    plt.ylabel("Reaction Video Time (s)")

    plt.ylim(bottom=0)
    plt.xlim(left=0)

    plt.grid(True, color=(0.35, 0.35, 0.35))
    plt.show()
