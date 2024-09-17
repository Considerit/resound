import os
import sys

sys.path.append(os.getcwd())


from utilities import conversion_audio_sample_rate as sr
from utilities import (
    conf,
    make_conf,
    unload_reaction,
)


from aligner.images.align_by_image import build_image_matches
from reactor_core import load_songs
from inventory.inventory import get_reactions_manifest


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
    jump_to = None  # "iamsickflowz" "RJJ's Reactions"  "Justin Hackert"  "Headbangers Guild" "Gimmickless Reactions" "Crypt" "COUNTY GAINS"

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

        do_something_per_reaction(build_image_matches)
