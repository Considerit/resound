import numpy as np
import math
import os
import sys

sys.path.append(os.getcwd())

import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

from utilities import conversion_audio_sample_rate as sr
from utilities import (
    conf,
    make_conf,
    unload_reaction,
)

from reactor_core import load_songs
from inventory.inventory import get_reactions_manifest

from aligner.images.edge_finder import find_edges_in_region_of_video


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

        # if channel not in ["mister energy", "Audio Lunch"]:
        #     continue

        try:
            conf["load_reaction"](channel)
        except:
            continue

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

        do_something_per_reaction(find_edges_in_region_of_video)
