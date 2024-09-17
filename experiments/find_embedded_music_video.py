import numpy as np
import os
import sys

sys.path.append(os.getcwd())

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

from aligner.images.embedded_video_finder import (
    get_bounding_box_of_music_video_in_reaction,
)


def do_something_per_reaction(callback):
    all_reactions = list(conf.get("reactions").keys())
    all_reactions.sort()
    all_reactions.reverse()

    reactions_manifest = get_reactions_manifest(
        conf.get("artist"), conf.get("song_name")
    )["reactions"]
    all_manifests = {}
    for vidID, reaction_manifest in reactions_manifest.items():
        reaction_file_prefix = reaction_manifest.get(
            "file_prefix", reaction_manifest["reactor"]
        )
        all_manifests[reaction_file_prefix] = reaction_manifest

    yet_to_encounter = True
    jump_to = None  # "LIYA Official"  # Melvin Thinks

    for i, channel in enumerate(all_reactions):
        reaction = conf.get("reactions").get(channel)
        manifest = all_manifests[channel]

        if not manifest.get("alignment_done"):
            continue

        if channel == jump_to:
            yet_to_encounter = False

        if (
            jump_to and yet_to_encounter and channel not in [jump_to]
        ):  # , "Bta Entertainment"]:
            continue

        try:
            conf["load_reaction"](channel)
        except Exception as e:
            print(e)
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

        do_something_per_reaction(get_bounding_box_of_music_video_in_reaction)
