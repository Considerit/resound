import numpy as np
import os
from pathlib import Path
import cv2
from retinaface import RetinaFace  # Assuming RetinaFace is installed
from tqdm import tqdm  # For the progress bar
from utilities import (
    read_object_from_file,
    save_object_to_file,
    conf,
    conversion_audio_sample_rate as sr,
)
from aligner.images.frame_operations import extract_frame


def detect_faces_in_frame(frame, face_detection_threshold=0.9):
    """
    Detect faces in the frame using RetinaFace's detect_faces method.
    """
    faces = RetinaFace.detect_faces(frame, threshold=face_detection_threshold)
    return faces


def get_faces_from_music_video(
    face_detection_threshold=0.9, resolution=1, times=None, frames={}
):
    music_video_path = conf.get("base_video_path")

    return get_faces_from_video(
        video_path=music_video_path,
        cache_dir=conf.get("song_directory"),
        face_detection_threshold=face_detection_threshold,
        resolution=resolution,
        times=times,
        frames=frames,
    )


def get_faces_from_reaction_video(
    reaction, face_detection_threshold=0.9, resolution=1, times=None, frames={}
):
    video_path = reaction.get("video_path")

    return get_faces_from_video(
        video_path=video_path,
        cache_dir=conf.get("temp_directory"),
        face_detection_threshold=face_detection_threshold,
        resolution=resolution,
        times=times,
        frames=frames,
    )


def get_faces_from_video(
    video_path,
    cache_dir,
    face_detection_threshold=0.9,
    resolution=1,
    times=None,
    frames={},
):
    """
    Extract faces from the video.

    :param video_path: Path to the video file.
    :param cache_dir: Where to cache results.
    :param resolution: Time in seconds between extractions (if times is None).
    :param times: List of frame indicies to extract faces from.
    :param face_detection_threshold: Confidence threshold for face detection in RetinaFace.
    """

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    faces_file = os.path.join(cache_dir, f"{Path(video_path).stem}-faces.json")

    modified = False
    cached = read_object_from_file(faces_file)
    if cached is None:
        cached = {}

    if f"{face_detection_threshold}" not in cached:
        cached[f"{face_detection_threshold}"] = {}

    faces = {int(f): v for f, v in cached[f"{face_detection_threshold}"].items()}

    if times is None:
        if fps <= 0:
            raise ValueError(
                f"Could not retrieve FPS from the video {music_video_path}"
            )

        # Calculate every_n_frames based on FPS and resolution (time in seconds between extractions)
        every_n_frames = int(fps * resolution)

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Generate an array of frame indices taken every every_n_frames
        times = np.arange(0, total_frames, every_n_frames)

    for idx, time in enumerate(tqdm(times, desc="Finding Faces", ncols=100)):
        time = int(time)
        if time not in faces:
            frame = frames.get(time, extract_frame(video_path, time / fps, cap=video))

            detected_faces = detect_faces_in_frame(
                frame=frame, face_detection_threshold=face_detection_threshold
            )

            for k, face in detected_faces.items():
                face["facial_area"] = [int(n) for n in face["facial_area"]]
                face["landmarks"] = {
                    kk: [float(n) for n in vv] for kk, vv in face["landmarks"].items()
                }

            faces[time] = detected_faces
            modified = True

    if modified:
        cached[f"{face_detection_threshold}"] = faces
        save_object_to_file(faces_file, cached)

    return faces, fps
