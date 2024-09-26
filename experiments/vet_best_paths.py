import numpy as np
import librosa
import math
import os
import sys

sys.path.append(os.getcwd())


import tempfile
import cv2
import imagehash
from PIL import Image
import matplotlib.pyplot as plt

from utilities import conversion_audio_sample_rate as sr
from utilities import (
    conf,
    make_conf,
    read_object_from_file,
    save_object_to_file,
    unload_reaction,
)

from reactor_core import load_songs

from inventory.inventory import get_reactions_manifest


def something(reaction):
    music_video_path = conf.get("base_video_path")
    reaction_video_path = reaction.get("video_path")
    channel = reaction.get("channel")

    alignment_metadata_file = reaction.get("alignment_metadata")
    if os.path.exists(alignment_metadata_file):
        metadata = read_object_from_file(alignment_metadata_file)

        assert "best_path" in metadata

        best_path = metadata.get("best_path")
        last_segment = None
        print(f"Validating best path for {channel}")
        for segment in best_path:
            reaction_start, reaction_end, music_start, music_end = segment[:4]

            if last_segment:
                if music_start < last_segment[3]:
                    print("\tERROR! BEND detected in music time", last_segment, segment)
                if reaction_start < last_segment[1]:
                    print("\tERROR! BEND detected in reaction time", last_segment, segment)

            last_segment = segment


def do_something_per_reaction(callback):
    all_reactions = list(conf.get("reactions").keys())
    all_reactions.sort()

    reactions_manifest = get_reactions_manifest(conf.get("artist"), conf.get("song_name"))[
        "reactions"
    ]
    all_manifests = {}
    for vidID, reaction_manifest in reactions_manifest.items():
        reaction_file_prefix = reaction_manifest.get("file_prefix", reaction_manifest["reactor"])
        all_manifests[reaction_file_prefix] = reaction_manifest

    for i, channel in enumerate(all_reactions):
        reaction = conf.get("reactions").get(channel)
        manifest = all_manifests[channel]

        # if channel not in ["Beard Wizard Man"]:
        #     continue

        # try:
        #     conf["load_reaction"](channel)
        # except:
        #     continue

        callback(reaction)

        unload_reaction(channel)


if __name__ == "__main__":
    songs = ["Ren - Kujo Beatdown"]
    songs = load_songs(songs)

    options = {}
    output_dir = "bounded"

    for song in songs:
        make_conf(song, options, output_dir)
        conf.get("load_reactions")()

        do_something_per_reaction(something)
