import numpy as np
import librosa
import math

import os
import sys

sys.path.append(os.getcwd())

from utilities import conversion_audio_sample_rate as sr
from utilities import conf, make_conf, read_object_from_file, unload_reaction

from reactor_core import load_songs

from aligner.sound_landmarks import contains_landmark, find_sound_landmarks_in_reaction
from inventory.inventory import get_reactions_manifest


def do_something_per_reaction(callback):
    all_reactions = list(conf.get("reactions").keys())
    all_reactions.sort()

    reactions_manifest = get_reactions_manifest(
        conf.get("artist"), conf.get("song_name")
    )["reactions"]
    all_manifests = {}
    for vidID, reaction_manifest in reactions_manifest.items():
        reaction_file_prefix = reaction_manifest.get(
            "file_prefix", reaction_manifest["reactor"]
        )
        all_manifests[reaction_file_prefix] = reaction_manifest

    for i, channel in enumerate(all_reactions):
        reaction = conf.get("reactions").get(channel)
        manifest = all_manifests[channel]

        if not manifest.get("alignment_done"):
            continue

        # if channel != "Bta Entertainment" and channel != "AR":
        #     continue

        # if channel != "COUNTY GAINS":
        #     continue

        try:
            conf["load_reaction"](channel)

        except:
            continue

        callback(reaction)

        unload_reaction(channel)


def sound_landmark_ground_truth_testing(song):
    all_results = {}

    def check_ground_truth(reaction):
        if not reaction.get("manifest").get("alignment_done"):
            return

        channel = reaction.get("channel")

        print(f"\n{channel} has completed alignment")

        alignment_video = reaction.get("aligned_path")
        alignment_metadata_file = reaction.get("alignment_metadata")

        metadata = read_object_from_file(alignment_metadata_file)
        reaction.update(metadata)
        best_path = reaction.get("best_path")
        song_data = conf.get("song_audio_data")
        song_length = len(song_data)

        if best_path[-1][3] < song_length:
            extend_by = song_length - best_path[-1][3]

            best_path[-1][1] += extend_by
            best_path[-1][3] += extend_by

        landmarks = find_sound_landmarks_in_reaction(reaction, show_visual=False)
        num_landmarks = len(landmarks.keys())

        total_landmarks_spanned = 0
        total_landmarks_matched = 0
        total_landmark_matches = {}

        for segment in best_path:
            landmarks_spanned, landmarks_matched, landmark_matches = contains_landmark(
                landmarks, segment[2], segment[3], segment[0], segment[1]
            )
            total_landmarks_spanned += landmarks_spanned
            total_landmarks_matched += landmarks_matched
            for landmark_start in landmark_matches.keys():
                cnt = total_landmark_matches.get(landmark_start, 0)
                cnt += 1
                total_landmark_matches[landmark_start] = cnt

        unique_landmarks_matched = len(total_landmark_matches.keys())

        print(
            f"\t\tLandmarks spanned = {total_landmarks_spanned} | matched = {total_landmarks_matched} | has multiple = {unique_landmarks_matched < total_landmarks_matched}"
        )
        if total_landmarks_matched > 0:
            precision = unique_landmarks_matched / total_landmarks_matched
            recall = unique_landmarks_matched / num_landmarks
            print(f"\t\tPrecision={precision} | Recall={recall}")
        else:
            precision = recall = 0

        all_results[channel] = {
            "channel": channel,
            "num_landmarks": num_landmarks,
            "unique_landmarks_matched": unique_landmarks_matched,
            "total_landmarks_matched": total_landmarks_matched,
            "total_landmarks_spanned": total_landmarks_spanned,
            "multiple_matches": unique_landmarks_matched < total_landmarks_matched,
            "precision": precision,
            "recall": recall,
        }

        num_channels = len(all_results.keys())
        incorrect_best_path = [
            v for v in all_results.values() if v["total_landmarks_spanned"] == 0
        ]
        correct_best_path_channels = num_channels - len(incorrect_best_path)
        correct = [
            v
            for v in all_results.values()
            if v["unique_landmarks_matched"] == v["num_landmarks"]
            and not v["multiple_matches"]
        ]
        if correct_best_path_channels > 0:
            print(
                f"**** RESULTS: {len(correct)} / {correct_best_path_channels} ({100 * len(correct) / correct_best_path_channels}%). {len(incorrect_best_path)} had incorrect best paths."
            )

    do_something_per_reaction(check_ground_truth)


if __name__ == "__main__":
    songs = ["Ren - Kujo Beatdown"]
    songs = load_songs(songs)

    options = {}
    output_dir = "bounded"

    for song in songs:
        make_conf(song, options, output_dir)
        conf.get("load_reactions")()
        sound_landmark_ground_truth_testing(song)
