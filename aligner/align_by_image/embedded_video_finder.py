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

from utilities.audio_processing import find_matches_for_loudest_parts_of_song
from utilities import conversion_audio_sample_rate as sr
from utilities import (
    conf,
    read_object_from_file,
    save_object_to_file,
)

from inventory.inventory import get_reactions_manifest
from face_finder.extract_faces import (
    extract_frame,
    get_faces_from_music_video,
    get_faces_from_reaction_video,
    detect_faces_in_frame,
)
from aligner.align_by_image.frame_operations import (
    crop_with_noise,
    crop_to_center_percent,
)

from aligner.align_by_image.edge_finder import find_edges_in_region_of_video


def get_bounding_box_of_music_video_in_reaction(reaction, visualize=False):
    """
    Identify where the music video is embedded in the reaction video using face detection and geometric analysis.

    :param reaction: The reaction we're trying to locate an embedded music video in.
    """

    hash_output_dir = os.path.join(conf.get("temp_directory"), "image_hashes")
    channel = reaction.get("channel")

    coordinates_path = os.path.join(hash_output_dir, f"coordinates_for_music_vid.json")
    if os.path.exists(coordinates_path):
        coordinates = read_object_from_file(coordinates_path)
    else:
        coordinates = {}

    if channel not in coordinates or visualize:
        try:
            coordinates[channel] = find_music_video_in_reaction(reaction, visualize=visualize)
            save_object_to_file(coordinates_path, coordinates, check_collisions=True)
        except Exception as e:
            print(f"Failed to get bounding box for {channel}", e)
            return None

    elif visualize:
        frames = get_frames_likely_to_have_music_video_embedded(reaction, num_frames=1)
        plot_edges_on_reaction_frame(frames[0][1], best_box=coordinates[channel])

    return coordinates[channel]


def find_music_video_in_reaction(reaction, visualize=False):
    # Frames to consider when constructing edge map and evaluating candidate edges
    print("Getting frames likely to have a music video embedded")
    frames = get_frames_likely_to_have_music_video_embedded(reaction, num_frames=60)

    all_boxes = []
    # Iterate through each likely match to process and visualize

    print("Extracting faces from those frames")

    music_frames = {music_time: frame for frame, _, music_time, _ in frames}

    reaction_frames = {reaction_time: frame for _, frame, _, reaction_time in frames}

    faces_in_song, fps = get_faces_from_music_video(
        frames=music_frames,
        times=music_frames.keys(),
        face_detection_threshold=0.9,
        resolution=1,
    )

    faces_in_reaction, fps = get_faces_from_reaction_video(
        reaction=reaction,
        frames=reaction_frames,
        times=reaction_frames.keys(),
        face_detection_threshold=0.9,
        resolution=1,
    )

    for music_frame, reaction_frame, music_time, reaction_time in tqdm(
        frames, desc="Boxes from faces", ncols=100
    ):
        # Step 1: Detect faces in both frames
        reaction_faces = faces_in_reaction[reaction_time]
        music_faces = faces_in_song[music_time]

        # Step 2: Match faces between the music video and reaction video
        matches = match_faces(music_faces, reaction_faces, reaction_frame, music_frame)

        reaction_frame_size = (
            reaction_frame.shape[1],
            reaction_frame.shape[0],
        )
        music_frame_size = music_frame.shape[1], music_frame.shape[0]  # (width, height)

        # Step 3: Infer the bounding box of the embedded music video
        inferred_boxes = infer_bounding_box(
            matches, music_faces, reaction_faces, reaction_frame_size, music_frame_size
        )
        if inferred_boxes is not None:
            all_boxes += inferred_boxes

        # Visualize the music and reaction frames with face bounding boxes
        if False:
            visualize_frames_with_faces(
                music_frame, reaction_frame, music_faces, reaction_faces, matches
            )

    all_edges = {"top": {}, "bottom": {}, "left": {}, "right": {}}
    tops = all_edges["top"]
    bottoms = all_edges["bottom"]
    lefts = all_edges["left"]
    rights = all_edges["right"]

    def break_box_into_edges(box):
        x, y, x2, y2 = box[0]
        x = int(x)
        y = int(y)
        x2 = int(x2)
        y2 = int(y2)

        if x not in lefts:
            lefts[x] = [x, y, x, y2]
        lefts[x][1] = min(y, lefts[x][1])
        lefts[x][3] = max(y2, lefts[x][3])

        if x2 not in rights:
            rights[x2] = [x2, y, x2, y2]
        rights[x2][1] = min(y, rights[x2][1])
        rights[x2][3] = max(y2, rights[x2][3])

        if y not in tops:
            tops[y] = [x, y, x2, y]
        tops[y][0] = min(x, tops[y][0])
        tops[y][2] = max(x2, tops[y][2])

        if y2 not in bottoms:
            bottoms[y2] = [x, y2, x2, y2]
        bottoms[y2][0] = min(x, bottoms[y2][0])
        bottoms[y2][2] = max(x2, bottoms[y2][2])

    reaction_frames = [f[1] for f in frames]
    if len(all_boxes) == 0:
        print(f"Could not find the location of the music video for {reaction.get('channel')}")
        return None

    best_score = np.max(np.array([x[1] for x in all_boxes]))
    candidate_boxes = [x for x in all_boxes if x[1] > 0.8 * best_score]

    best_boxes = [x for x in candidate_boxes if x[1] > 0.9 * best_score]

    median_best_box = [
        np.median(np.array([b[0][0] for b in best_boxes])),
        np.median(np.array([b[0][1] for b in best_boxes])),
        np.median(np.array([b[0][2] for b in best_boxes])),
        np.median(np.array([b[0][3] for b in best_boxes])),
    ]

    candidate_boxes.append([median_best_box, best_score])
    aligned_median_bbox = align_aspect_ratios(
        music_shape=frames[0][0].shape, reaction_box=median_best_box
    )

    print("Bests", median_best_box, aligned_median_bbox)
    candidate_boxes.append([aligned_median_bbox, best_score])

    for box in candidate_boxes:
        break_box_into_edges(box)

    tops_from_boxes = list(tops.values())
    for edge in tops_from_boxes:
        x, y, x2, y2 = edge
        new_tops = detect_nearby_edges(reaction_frames, edge, direction="top")

        for new_y in new_tops:
            if new_y not in tops:
                tops[new_y] = [x, new_y, x2, new_y]
            tops[new_y][0] = min(x, tops[new_y][0])
            tops[new_y][2] = max(x2, tops[new_y][2])

    bottoms_from_boxes = list(bottoms.values())
    for edge in bottoms_from_boxes:
        x, y, x2, y2 = edge
        new_bottoms = detect_nearby_edges(reaction_frames, edge, direction="bottom")

        for new_y2 in new_bottoms:
            if new_y2 not in bottoms:
                bottoms[new_y2] = [x, new_y2, x2, new_y2]
            bottoms[new_y2][0] = min(x, bottoms[new_y2][0])
            bottoms[new_y2][2] = max(x2, bottoms[new_y2][2])

    lefts_from_boxes = list(lefts.values())
    for edge in lefts_from_boxes:
        x, y, x2, y2 = edge
        new_lefts = detect_nearby_edges(reaction_frames, edge, direction="left")

        for new_x in new_lefts:
            if new_x not in lefts:
                lefts[new_x] = [new_x, y, new_x, y2]
            lefts[new_x][1] = min(y, lefts[new_x][1])
            lefts[new_x][3] = max(y2, lefts[new_x][3])

    rights_from_boxes = list(rights.values())
    for edge in rights_from_boxes:
        x, y, x2, y2 = edge
        new_rights = detect_nearby_edges(reaction_frames, edge, direction="right")

        for new_x2 in new_rights:
            if new_x2 not in rights:
                rights[new_x2] = [new_x2, y, new_x2, y2]
            rights[new_x2][1] = min(y, rights[new_x2][1])
            rights[new_x2][3] = max(y2, rights[new_x2][3])

    best_bbox = None
    best_score = -1

    # Generate all possible boxes from the identified edges
    candidate_bboxes = []

    # manage n^4 explosion
    def prune_edges(edges, sparsity):
        pos = sorted(edges.keys())
        new_edges = [pos[0]]
        for e in pos:
            if abs(e - new_edges[-1]) >= sparsity:
                new_edges.append(e)
        return {e: edges[e] for e in new_edges}

    sparsity = 2  # how close each edge is allowed to be
    while True:
        for edge in ["top", "bottom", "right", "left"]:
            all_edges[edge] = prune_edges(all_edges[edge], sparsity)

        tops = all_edges["top"]
        bottoms = all_edges["bottom"]
        rights = all_edges["right"]
        lefts = all_edges["left"]
        total_boxes = len(tops) * len(bottoms) * len(rights) * len(lefts)

        print(
            f"\tPruning {total_boxes} down to {len(tops) * len(bottoms) * len(rights) * len(lefts)}"
        )

        if total_boxes < 1500 or sparsity > 5:
            break

        sparsity += 1

    for y in tops.keys():
        for y2 in bottoms.keys():
            if y2 <= y:
                continue

            for x in lefts.keys():
                for x2 in rights.keys():
                    if x2 <= x:
                        continue

                    if x2 - x < y2 - y:  # enforce width > height constraint
                        continue

                    bounding_box = (x, y, x2, y2)
                    candidate_bboxes.append(bounding_box)

    # if visualize:
    #     plot_edges_on_reaction_frame(
    #         reaction_frames[0],
    #         tops=tops,
    #         bottoms=bottoms,
    #         lefts=lefts,
    #         rights=rights,
    #     )

    bboxes_with_scores = []
    music_frame_phashes = {}
    music_frame_dhashes = {}

    best_score = -1
    for bbox in tqdm(candidate_bboxes, desc="Scoring bboxes", ncols=100):
        scores = []
        for music_frame, reaction_frame, music_time, reaction_time in frames:
            if music_time not in music_frame_phashes:
                music_frame_phashes[music_time] = compute_perceptual_hash(music_frame, hash_size=8)

            # TODO: A speed improvement could probably be seen by, for very large
            #       bboxes, extracting only the center 50%.

            cropped_reaction_frame = crop_with_noise(reaction_frame, bbox)

            mhash = music_frame_phashes[music_time]
            rhash = compute_perceptual_hash(cropped_reaction_frame, hash_size=8)

            perceptual_hash_diff = mhash - rhash
            pscore = 1 / (1 + perceptual_hash_diff)

            scores.append(pscore)

        overall_score = np.median(np.array(scores))
        if overall_score > best_score:
            print(f"Best score is {overall_score}", bbox)
            best_score = overall_score

        bboxes_with_scores.append((bbox, overall_score))

    bboxes_with_scores.sort(key=lambda x: x[1], reverse=True)

    best_box = bboxes_with_scores[0][0]

    if visualize:
        plot_edges_on_reaction_frame(
            reaction_frames[0],
            tops=tops,
            bottoms=bottoms,
            lefts=lefts,
            rights=rights,
            best_box=best_box,
        )

    return best_box


def get_frames_likely_to_have_music_video_embedded(reaction, num_frames=60):
    """
    Sample frames from the loudest sections of the reaction video that likely contain faces.

    :param reaction: Data about the target reaction.
    :param num_frames: Number of frames to sample.
    :return: List of sampled frames from the video.
    """

    reaction_video_path = reaction.get("video_path")
    reaction_video = cv2.VideoCapture(reaction_video_path)
    reaction_fps = reaction_video.get(cv2.CAP_PROP_FPS)

    music_video_path = conf.get("base_video_path")
    music_video = cv2.VideoCapture(music_video_path)
    music_fps = music_video.get(cv2.CAP_PROP_FPS)

    loudest_matches = get_loudest_matches_between_reaction_and_song_with_faces(
        reaction, num_matches=num_frames
    )

    frames = []

    for (song_time, reaction_time), score in tqdm(
        loudest_matches, desc="Extracting frames", ncols=100
    ):
        reaction_frame_idx = reaction_time / sr * reaction_fps
        reaction_frame = extract_frame(frame_idx=reaction_frame_idx, cap=reaction_video)

        song_frame_idx = song_time / sr * music_fps
        music_frame = extract_frame(frame_idx=song_frame_idx, cap=music_video)

        if reaction_frame is not None and music_frame is not None:
            frames.append((music_frame, reaction_frame, song_time, reaction_time))

    reaction_video.release()
    music_video.release()
    return frames


def get_loudest_matches_between_reaction_and_song_with_faces(reaction, num_matches):
    reaction_audio = reaction.get("reaction_audio_data")
    song_mfcc = conf.get("song_audio_mfcc")
    reaction_mfcc = reaction.get("reaction_audio_mfcc")

    faces_in_music_video, fps = get_faces_from_music_video(
        face_detection_threshold=0.9, resolution=1
    )

    def likely_has_a_face(song_time, reaction_time):
        t = int(song_time / sr) * fps
        prob_has_face = t in faces_in_music_video and len(faces_in_music_video[t].keys()) > 0
        return prob_has_face

    loudest_matches = find_matches_for_loudest_parts_of_song(
        reaction,
        conf.get("song_audio_data"),
        reaction_audio,
        song_mfcc,
        reaction_mfcc,
        segment_length=2,  # second
        top_n=num_matches,
        good_match_threshold=0.8,
        is_valid=likely_has_a_face,
    )

    return loudest_matches


def detect_nearby_edges(
    reaction_frames,
    edge,
    direction,
    padding=50,
    visualize=False,
):
    """
    Uses edge detection (Canny) to find edges that are nearby the bounding box.

    :param reaction_frames: The reaction frames to use to synthesize nearby edges.
    :param bounding_box: Initial bounding box (x, y, x2, y2) in the reaction frame.
    :param padding: Padding around the edge to consider for edge detection.
    :param visualize: If True, visualize the detection process.
    :return: The refined bounding box (x, y, x2, y2).
    """

    # Extract bounding box coordinates
    x, y, x2, y2 = edge

    height, width = reaction_frames[0].shape[:2]

    # Crop only the region of interest with padding around the left edge
    if direction == "left":
        edge_region = (max(0, x - padding), y, min(width, x + padding), y2)

    elif direction == "right":
        edge_region = (max(0, x2 - padding), y, min(width, x2 + padding), y2)

    elif direction == "top":
        edge_region = (x, max(0, y - padding), x2, min(height, y + padding))

    elif direction == "bottom":
        edge_region = (x, max(0, y2 - padding), x2, min(height, y2 + padding))

    # Apply Canny edge detection within this region
    lines, heatmap = find_edges_in_region_of_video(
        reaction_frames,
        edge_region,
        is_vertical=direction in ["left", "right"],
        visualize=visualize,
    )

    candidates = []
    candidate_idx = 0 if direction in ["left", "right"] else 1
    for line in lines or []:
        edge_pos = round(line["start"][candidate_idx])
        candidates.append(edge_pos)

    # Add the frame boundary if the original line is close enough to it
    if direction == "left" and x < padding:
        candidates.append(0)  # Snap to left boundary
    elif direction == "right" and width - x2 < padding:
        candidates.append(width)  # Snap to right boundary
    elif direction == "top" and y < padding:
        candidates.append(0)  # Snap to top boundary
    elif direction == "bottom" and height - y2 < padding:
        candidates.append(height)  # Snap to bottom boundary

    return candidates


def compute_perceptual_hash(frame, hash_size=8, highfreq_factor=4):
    """
    Compute the perceptual hash of a given frame.
    The frame can be either a numpy array (which will be converted to a PIL Image)
    or already a PIL Image.
    """
    # Check if the frame is a NumPy array and convert it to a PIL Image if necessary
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)

    # Compute and return the perceptual hash (phash)
    return imagehash.phash(frame, hash_size=hash_size, highfreq_factor=highfreq_factor)


def match_faces(
    music_faces,
    reaction_faces,
    reaction_frame,
    music_frame,
    similarity_threshold=0.9,
):
    """
    Attempt to find high-quality matches between music video faces and reaction video faces,
    using both facial landmarks and perceptual hashing. Ensure that each face from the music
    video can only be matched with one face from the reaction.

    :param music_faces: Detected faces in the music video frame.
    :param reaction_faces: Detected faces in the reaction video frame.
    :param reaction_frame: Frame from the reaction video.
    :param music_frame: Frame from the music video.
    :param similarity_threshold: Minimum similarity score to consider a match.
    :return: List of high-quality matches (music_face_key, reaction_face_key, combined_score).
    """
    # List to store all pairwise match scores
    pairwise_scores = []

    # Step 1: Compute all pairwise match scores
    for m_key, music_face in music_faces.items():
        # Crop the music video face and compute perceptual hash
        music_face_image = crop_face_from_frame(music_frame, music_face["facial_area"])
        music_face_hash = compute_perceptual_hash(music_face_image)

        for r_key, reaction_face in reaction_faces.items():
            perceptual_hash_score, landmark_score, combined_score = score_match(
                music_frame,
                reaction_frame,
                music_face,
                reaction_face,
                music_face_image,
                music_face_hash,
            )

            # Only consider this match if the landmark score is above the threshold
            if landmark_score >= similarity_threshold:
                # Add to the pairwise score list (music_face_key, reaction_face_key, combined_score)
                pairwise_scores.append((m_key, r_key, combined_score))

    # Step 2: Sort all pairwise matches by the combined score (descending order)
    pairwise_scores.sort(key=lambda x: x[2], reverse=True)

    # Step 3: Perform greedy assignment (each face is paired at most once)
    matched_music_faces = set()
    matched_reaction_faces = set()
    matches = []

    for m_key, r_key, score in pairwise_scores:
        # Skip if either face is already matched
        if m_key in matched_music_faces or r_key in matched_reaction_faces:
            continue

        music_face = music_faces[m_key]
        reaction_face = reaction_faces[r_key]

        best_facial_area, best_score = gradient_descent_face_matching(
            reaction_face,
            music_face,
            reaction_frame,
            music_frame,
            score,
        )

        if best_score != score:
            reaction_face["facial_area"] = best_facial_area
            score = best_score

        # Add this match to the final list
        matches.append(((m_key, r_key), score))

        # Mark these faces as matched
        matched_music_faces.add(m_key)
        matched_reaction_faces.add(r_key)

    return matches


def compare_face_landmarks(
    music_landmarks, reaction_landmarks, music_facial_area, reaction_facial_area
):
    """
    Compare facial landmarks between two faces (one from music video, one from reaction video).
    Normalizes the landmarks relative to the face bounding box.

    :param music_landmarks: Landmarks from the music video frame.
    :param reaction_landmarks: Landmarks from the reaction video frame.
    :param music_facial_area: Facial bounding box from the music video.
    :param reaction_facial_area: Facial bounding box from the reaction video.
    :return: Similarity score between 0 and 1 (1 means very similar).
    """
    # Normalize the landmarks relative to their respective facial bounding boxes
    normalized_music_landmarks = normalize_landmarks(music_landmarks, music_facial_area)
    normalized_reaction_landmarks = normalize_landmarks(reaction_landmarks, reaction_facial_area)

    # Compute the Euclidean distance between the normalized landmarks
    distances = []
    for key in normalized_music_landmarks:
        if key in normalized_reaction_landmarks:
            music_point = np.array(normalized_music_landmarks[key])
            reaction_point = np.array(normalized_reaction_landmarks[key])
            distance = np.linalg.norm(music_point - reaction_point)
            distances.append(distance)

    # Return inverse of average distance as a similarity score
    if distances:
        avg_distance = np.mean(distances)
        return 1 / (1 + avg_distance)  # Inverse of mean distance for similarity score
    else:
        return 0  # No landmarks to compare


def normalize_landmarks(landmarks, facial_area):
    """
    Normalize the facial landmarks relative to the facial bounding box.

    :param landmarks: Dictionary of facial landmarks (eyes, nose, mouth, etc.).
    :param facial_area: Bounding box of the face in the form [top_x, top_y, bottom_x, bottom_y].
    :return: Normalized landmarks.
    """
    top_x, top_y, bottom_x, bottom_y = facial_area
    width = bottom_x - top_x
    height = bottom_y - top_y

    # Normalize each landmark by subtracting the top-left of the bounding box
    # and scaling by the width and height of the bounding box
    normalized_landmarks = {}
    for key, point in landmarks.items():
        normalized_x = (point[0] - top_x) / width
        normalized_y = (point[1] - top_y) / height
        normalized_landmarks[key] = [normalized_x, normalized_y]

    return normalized_landmarks


def score_match(
    music_frame,
    reaction_frame,
    music_face,
    reaction_face,
    music_face_image=None,
    music_face_hash=None,
    hash_size=8,
    highfreq_factor=4,
):
    if music_face_image is None:
        music_face_image = crop_face_from_frame(music_frame, music_face["facial_area"])
        music_face_hash = compute_perceptual_hash(
            music_face_image, hash_size=hash_size, highfreq_factor=highfreq_factor
        )

    # Crop the reaction video face and compute perceptual hash
    reaction_face_image = crop_face_from_frame(reaction_frame, reaction_face["facial_area"])
    reaction_face_hash = compute_perceptual_hash(
        reaction_face_image, hash_size=hash_size, highfreq_factor=highfreq_factor
    )

    # Compare perceptual hashes (smaller difference is better)
    perceptual_hash_diff = music_face_hash - reaction_face_hash
    perceptual_hash_score = 1 / (1 + perceptual_hash_diff)  # Convert to similarity score (0 to 1)

    # Compare normalized landmarks
    landmark_score = compare_face_landmarks(
        music_face["landmarks"],
        reaction_face["landmarks"],
        music_face["facial_area"],
        reaction_face["facial_area"],
    )

    combined_score = perceptual_hash_score  # landmark_score * perceptual_hash_score

    return perceptual_hash_score, landmark_score, combined_score


def infer_bounding_box(matches, music_faces, reaction_faces, reaction_frame_size, music_frame_size):
    """
    Infer the position and size of the embedded music video based on face matches.

    :param matches: List of matched faces (music_face_key, reaction_face_key, score).
    :param music_faces: Detected faces in the music video frame.
    :param reaction_faces: Detected faces in the reaction video frame.
    :param reaction_frame_size: Size of the reaction video frame (width, height).
    :param music_frame_size: Size of the Music video frame (width, height).
    :return: Inferred bounding box as (x1, y1, x2, y2), or None if no valid box can be determined.
    """
    if not matches:
        return None

    M_w, M_h = music_frame_size
    R_w, R_h = reaction_frame_size

    inferences = []

    for (music_face_key, reaction_face_key), score in matches:
        # Get face bounding boxes for both frames
        mf_x, mf_y, mf_xr, mf_yb = music_faces[music_face_key]["facial_area"]
        mf_w = mf_xr - mf_x
        mf_h = mf_yb - mf_y

        music_face_relative_pos = (mf_x / M_w, mf_h / M_h)

        rf_x, rf_y, rf_xr, rf_yb = reaction_faces[reaction_face_key]["facial_area"]
        rf_w = rf_xr - rf_x
        rf_h = rf_yb - rf_y

        reaction_face_relative_pos = (rf_x / R_w, rf_h / R_h)

        embedded_M_x = rf_x - rf_w / mf_w * mf_x
        embedded_M_y = rf_y - rf_h / mf_h * mf_y

        embedded_M_w = rf_w / mf_w * M_w
        embedded_M_h = rf_h / mf_h * M_h

        inference = [
            round(embedded_M_x),
            round(embedded_M_y),
            embedded_M_x + round(embedded_M_w),
            embedded_M_y + round(embedded_M_h),
        ]

        if not (
            inference[0] < -R_w * 0.2
            and inference[1] < -R_h * 0.2
            and inference[0] + inference[2] > R_w * 1.2
            and inference[1] + inference[3] > R_h * 1.2
        ):
            inferences.append(
                (
                    inference,
                    score,
                )
            )
        # else:
        #     print("Inference seems wacky", inference)

    return inferences


def align_aspect_ratios(music_shape, reaction_box):
    x, y, x2, y2 = map(int, reaction_box)
    w = x2 - x
    h = y2 - y
    center_x, center_y = x + w // 2, y + h // 2

    # Calculate the aspect ratio of the music frame
    music_aspect_ratio = music_shape[1] / music_shape[0]

    # Calculate the aspect ratio of the reaction frame's initial bounding box
    reaction_aspect_ratio = w / h

    # Adjust the bounding box to match the music aspect ratio while keeping the center the same
    if reaction_aspect_ratio != music_aspect_ratio:
        if reaction_aspect_ratio > music_aspect_ratio:
            # Bounding box is wider than the music frame, so reduce width
            new_w = int(h * music_aspect_ratio)
            new_h = h
        else:
            # Bounding box is taller than the music frame, so reduce height
            new_h = int(w / music_aspect_ratio)
            new_w = w

        # Expand or contract the bounding box symmetrically
        x = center_x - new_w // 2
        y = center_y - new_h // 2
        w, h = new_w, new_h

    return [x, y, x + w, y + h]


def gradient_descent_face_matching(
    reaction_face,
    music_face,
    reaction_frame,
    music_frame,
    initial_score,
    learning_rate=1,
    pixel_change=1,
    hash_size=32,
):
    """
    Iteratively adjust the size and position of the reaction face bounding box using pixel adjustments.

    :param reaction_face: Detected face data from the reaction video (landmarks, facial_area).
    :param music_face: Detected face data from the music video (landmarks, facial_area).
    :param reaction_frame: Frame from the reaction video.
    :param music_frame: Frame from the music video.
    :param initial_score: Initial matching score.
    :param learning_rate: Amount of pixel shift applied in each step.
    :param pixel_change: Number of pixels to change width/height in each step.
    :return: The best refined reaction_face bounding box and the best score.
    """
    best_facial_area = reaction_face["facial_area"]
    best_score = initial_score

    # Calculate aspect ratios for both faces
    def aspect_ratio(box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width / height

    music_aspect_ratio = aspect_ratio(music_face["facial_area"])
    reaction_aspect_ratio = aspect_ratio(reaction_face["facial_area"])

    # Pixel-based width/height changes based on aspect ratio difference
    change_options = [(0, 0)]  # No change in width/height

    if music_aspect_ratio > reaction_aspect_ratio:
        # Widen the reaction face or decrease height to match aspect ratio
        change_options.append((pixel_change, 0))  # Increase width
        change_options.append((0, -pixel_change))  # Decrease height
        change_options.append((2 * pixel_change, 0))  # Increase width
        change_options.append((0, -2 * pixel_change))  # Decrease height

    elif music_aspect_ratio < reaction_aspect_ratio:
        # Narrow the reaction face or increase height to match aspect ratio
        change_options.append((-pixel_change, 0))  # Decrease width
        change_options.append((0, pixel_change))  # Increase height
        change_options.append((-2 * pixel_change, 0))  # Decrease width
        change_options.append((0, 2 * pixel_change))  # Increase height

    shifts = [
        (0, 0),
        (learning_rate, 0),
        (-learning_rate, 0),
        (0, learning_rate),
        (0, -learning_rate),
        (learning_rate, learning_rate),
        (-learning_rate, -learning_rate),
        (-learning_rate, learning_rate),
        (learning_rate, -learning_rate),
    ]

    music_face_hash = compute_perceptual_hash(
        crop_face_from_frame(music_frame, music_face["facial_area"]), hash_size=32
    )

    improved = True

    # print("Trying to improve face")

    while improved:
        improved = False

        # Try all combinations of shifts and pixel changes
        for shift_x, shift_y in shifts:
            for change_width, change_height in change_options:
                # Adjust the bounding box
                adjusted_facial_area = adjust_face_box(
                    best_facial_area, shift_x, shift_y, change_width, change_height
                )

                adjusted_reaction_face = dict(reaction_face)
                adjusted_reaction_face["facial_area"] = adjusted_facial_area

                perceptual_hash_score, landmark_score, combined_score = score_match(
                    music_frame,
                    reaction_frame,
                    music_face,
                    adjusted_reaction_face,
                    music_face_hash=music_face_hash,
                    hash_size=32,
                )

                # If the new score is better, update the best score and bounding box
                if combined_score > best_score:
                    best_facial_area = adjusted_facial_area
                    best_score = combined_score
                    improved = True

        # If no improvement after all shifts/pixel changes, break
        if not improved:
            break

    # if (best_score - initial_score) > 0.01:
    #     print(
    #         f"\tIMPROVED {best_score - initial_score}! {initial_score} => {best_score}"
    #     )
    # else:
    #     print(f"\tMarginal improvement {best_score}")

    # Return the best face box and score
    return best_facial_area, best_score


def crop_face_from_frame(frame, facial_area, target_size=(128, 128)):
    """
    Crop the face region from the frame based on the given facial area, resize, and convert to grayscale.

    :param frame: The full frame (numpy array) containing the face.
    :param facial_area: Bounding box of the face [top_x, top_y, bottom_x, bottom_y].
    :param target_size: The size to which the face should be resized for perceptual hashing.
    :return: Cropped and resized face as a grayscale PIL Image.
    """
    x1, y1, x2, y2 = facial_area
    # face = frame[y1:y2, x1:x2]  # Crop the face region

    face = crop_with_noise(frame, facial_area)

    # Convert to PIL Image and resize to the target size
    face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize(target_size)

    # Convert to grayscale
    face_image = face_image.convert("L")

    return face_image


def adjust_face_box(facial_area, shift_x=0, shift_y=0, change_width=0, change_height=0):
    """
    Adjust the facial area by shifting and resizing the bounding box by pixel changes.

    :param facial_area: Bounding box of the face [x1, y1, x2, y2].
    :param shift_x: Horizontal shift in pixels.
    :param shift_y: Vertical shift in pixels.
    :param change_width: Change in width in pixels.
    :param change_height: Change in height in pixels.
    :return: Adjusted bounding box [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = facial_area
    width = x2 - x1
    height = y2 - y1

    # Apply the pixel changes to the width and height
    new_width = width + change_width
    new_height = height + change_height

    # Recalculate the new coordinates with resizing and shifting
    new_x1 = x1 + shift_x  # - change_width / 2
    new_y1 = y1 + shift_y  # - change_height / 2
    new_x2 = new_x1 + new_width
    new_y2 = new_y1 + new_height

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


def draw_face_boxes(frame, faces, color=(0, 255, 0), label="Face"):
    """Draws rectangles around detected faces."""
    for key, face in faces.items():
        print("SFSF", face)
        bbox = face["facial_area"]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def visualize_frames_with_faces(music_frame, reaction_frame, music_faces, reaction_faces, matches):
    """Visualizes music and reaction frames side by side with face bounding boxes and matches."""

    # Clone frames to draw bounding boxes
    music_frame_copy = music_frame.copy()
    reaction_frame_copy = reaction_frame.copy()

    # Draw face boxes for music frame
    draw_face_boxes(music_frame_copy, music_faces, color=(0, 255, 0), label="Music Face")

    # Draw face boxes for reaction frame
    draw_face_boxes(reaction_frame_copy, reaction_faces, color=(255, 0, 0), label="Reaction Face")

    # Plot the music and reaction frames side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Convert frames from BGR to RGB (since OpenCV loads images in BGR format)
    music_frame_rgb = cv2.cvtColor(music_frame_copy, cv2.COLOR_BGR2RGB)
    reaction_frame_rgb = cv2.cvtColor(reaction_frame_copy, cv2.COLOR_BGR2RGB)

    ax[0].imshow(music_frame_rgb)
    ax[0].set_title("Music Frame")
    ax[0].axis("off")

    ax[1].imshow(reaction_frame_rgb)
    ax[1].set_title("Reaction Frame")
    ax[1].axis("off")

    # Optionally, you can draw lines to visualize the matches
    # Example of how to draw lines for matching faces
    for (music_key, reaction_key), match_score in matches:
        music_bbox = music_faces[music_key]["facial_area"]
        reaction_bbox = reaction_faces[reaction_key]["facial_area"]

        # You can draw a line between corresponding faces in music and reaction frames, or just highlight the match
        x1_m, y1_m, x2_m, y2_m = map(int, music_bbox)
        x1_r, y1_r, x2_r, y2_r = map(int, reaction_bbox)

        # You can modify the visualization logic here if you want to draw arrows, annotations, etc.

    plt.show()


import matplotlib.patches as patches


def plot_edges_on_reaction_frame(
    reaction_frame, tops=None, bottoms=None, lefts=None, rights=None, best_box=None
):
    """
    Plot a sample reaction frame with all the edges (tops, bottoms, lefts, and rights) overlaid.

    :param reaction_frame: A sample reaction frame from reaction_frames.
    :param tops: Dictionary of top edges (y-coordinates).
    :param bottoms: Dictionary of bottom edges (y2-coordinates).
    :param lefts: Dictionary of left edges (x-coordinates).
    :param rights: Dictionary of right edges (x2-coordinates).
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the reaction frame
    ax.imshow(cv2.cvtColor(reaction_frame, cv2.COLOR_BGR2RGB))

    if tops:
        # Overlay top edges in red
        for top in tops.values():
            x, y, x2, y2 = top
            ax.add_patch(
                patches.Rectangle((x, y), x2 - x, 0, edgecolor="red", facecolor="none", lw=2)
            )
    if bottoms:
        # Overlay bottom edges in blue
        for bottom in bottoms.values():
            x, y, x2, y2 = bottom
            ax.add_patch(
                patches.Rectangle((x, y2), x2 - x, 0, edgecolor="blue", facecolor="none", lw=2)
            )

    if lefts:
        # Overlay left edges in green
        for left in lefts.values():
            x, y, x2, y2 = left
            ax.add_patch(
                patches.Rectangle((x, y), 0, y2 - y, edgecolor="green", facecolor="none", lw=2)
            )

    if rights:
        # Overlay right edges in orange
        for right in rights.values():
            x, y, x2, y2 = right
            ax.add_patch(
                patches.Rectangle((x2, y), 0, y2 - y, edgecolor="orange", facecolor="none", lw=2)
            )

    if best_box:
        # Draw the best box!
        x, y, x2, y2 = best_box
        ax.add_patch(
            patches.Rectangle((x, y), x2 - x, y2 - y, edgecolor="magenta", facecolor="none", lw=4)
        )

    # Add a legend
    handles = [
        patches.Patch(color="red", label="Top edges"),
        patches.Patch(color="blue", label="Bottom edges"),
        patches.Patch(color="green", label="Left edges"),
        patches.Patch(color="orange", label="Right edges"),
        patches.Patch(color="magenta", label="Best Box"),
    ]
    ax.legend(handles=handles, loc="upper right")

    # Show the plot
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def adjust_coordinates_for_offscreen_embed(
    reaction_video_path,
    music_video_path,
    reaction_crop_coordinates,
    threshold=0.02,
    visualize=False,
):
    # Load video properties for reaction video and music video
    reaction_video = cv2.VideoCapture(reaction_video_path)
    music_video = cv2.VideoCapture(music_video_path)

    # Get dimensions of both videos
    reaction_width = int(reaction_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    reaction_height = int(reaction_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    music_width = int(music_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    music_height = int(music_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Unpack the reaction crop coordinates
    x1, y1, x2, y2 = reaction_crop_coordinates

    # Calculate the width and height of the reaction crop box
    crop_width = x2 - x1
    crop_height = y2 - y1

    # Calculate the percentage out of bounds for each edge
    out_of_bounds_percentages = {
        "left": -x1 / crop_width if x1 < 0 else 0,
        "right": (x2 - reaction_width) / crop_width if x2 > reaction_width else 0,
        "top": -y1 / crop_height if y1 < 0 else 0,
        "bottom": (y2 - reaction_height) / crop_height if y2 > reaction_height else 0,
    }

    # Find the maximum out-of-bounds percentage
    shrink_factor = max(out_of_bounds_percentages.values())

    # If the maximum out-of-bounds percentage is less than the threshold, no adjustment needed
    if shrink_factor <= threshold:
        return reaction_crop_coordinates, None

    # Adjust the reaction crop coordinates proportionally
    new_x1 = int(x1 + crop_width * shrink_factor)
    new_y1 = int(y1 + crop_height * shrink_factor)
    new_x2 = int(x2 - crop_width * shrink_factor)
    new_y2 = int(y2 - crop_height * shrink_factor)

    reaction_adjusted_coordinates = [new_x1, new_y1, new_x2, new_y2]

    # Apply the same shrinkage factor to the music video dimensions
    music_new_x1 = int(music_width * shrink_factor / 2)
    music_new_y1 = int(music_height * shrink_factor / 2)
    music_new_x2 = music_width - music_new_x1
    music_new_y2 = music_height - music_new_y1

    music_adjusted_coordinates = [music_new_x1, music_new_y1, music_new_x2, music_new_y2]

    if visualize:
        # Get the reaction frame 25% into the video
        reaction_frame_index = int(reaction_video.get(cv2.CAP_PROP_FRAME_COUNT) * 0.25)
        reaction_video.set(cv2.CAP_PROP_POS_FRAMES, reaction_frame_index)
        ret, reaction_frame = reaction_video.read()

        # Get a frame from the music video
        music_frame_index = int(music_video.get(cv2.CAP_PROP_FRAME_COUNT) * 0.25)
        music_video.set(cv2.CAP_PROP_POS_FRAMES, music_frame_index)
        ret, music_frame = music_video.read()

        # Plot side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Reaction frame with original and adjusted crop
        axes[0].imshow(cv2.cvtColor(reaction_frame, cv2.COLOR_BGR2RGB))
        axes[0].add_patch(
            plt.Rectangle(
                (x1, y1),
                crop_width,
                crop_height,
                edgecolor="red",
                facecolor="none",
                lw=2,
                label="Original",
            )
        )
        axes[0].add_patch(
            plt.Rectangle(
                (new_x1, new_y1),
                new_x2 - new_x1,
                new_y2 - new_y1,
                edgecolor="green",
                facecolor="none",
                lw=2,
                label="Adjusted",
            )
        )
        axes[0].set_title("Reaction Frame")
        axes[0].legend()

        # Music frame with adjusted and original crop
        axes[1].imshow(cv2.cvtColor(music_frame, cv2.COLOR_BGR2RGB))
        axes[1].add_patch(
            plt.Rectangle(
                (0, 0),
                music_width,
                music_height,
                edgecolor="red",
                facecolor="none",
                lw=2,
                label="Original",
            )
        )
        axes[1].add_patch(
            plt.Rectangle(
                (music_new_x1, music_new_y1),
                music_new_x2 - music_new_x1,
                music_new_y2 - music_new_y1,
                edgecolor="green",
                facecolor="none",
                lw=2,
                label="Adjusted",
            )
        )
        axes[1].set_title("Music Video Frame")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return reaction_adjusted_coordinates, music_adjusted_coordinates
