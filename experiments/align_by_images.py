import os
import sys
import traceback

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

from aligner.images.embedded_video_finder import get_bounding_box_of_music_video_in_reaction


def do_something_per_reaction(callback):
    all_reactions = list(conf.get("reactions").keys())
    all_reactions.sort()
    # all_reactions.reverse()

    reactions_manifest = get_reactions_manifest(conf.get("artist"), conf.get("song_name"))[
        "reactions"
    ]
    all_manifests = {}
    for vidID, reaction_manifest in reactions_manifest.items():
        reaction_file_prefix = reaction_manifest.get("file_prefix", reaction_manifest["reactor"])
        all_manifests[reaction_file_prefix] = reaction_manifest

    yet_to_encounter = True
    jump_to = None

    for i, channel in enumerate(all_reactions):
        reaction = conf.get("reactions").get(channel)
        manifest = all_manifests[channel]

        # if not manifest.get("alignment_done"):
        #     continue

        # if channel not in []:
        #     continue

        # if channel not in [
        #     "mister energy",
        #     "Gimmickless Reactions",
        #     "Headbangers Guild",
        #     "Second Covers",
        #     "Touchy Reactions",
        #     "iamsickflowz",
        #     "RJJ's Reactions",
        #     "Justin Hackert",
        #     "Headbangers Guild",
        #     "Crypt",
        #     "COUNTY GAINS",
        # ]:
        #     continue

        if channel == jump_to:
            yet_to_encounter = False

        if jump_to and yet_to_encounter and channel not in [jump_to]:  # , "Bta Entertainment"]:
            continue

        hash_output_dir = os.path.join(conf.get("temp_directory"), "image_hashes")
        crop_coordinates = get_bounding_box_of_music_video_in_reaction(reaction)

        if crop_coordinates is not None:
            print("Found embedded music video at", crop_coordinates)

            hash_cache_file_name = f"{channel}-{2}fps-{tuple(crop_coordinates)}-{0.95}.json"
            hash_cache_path = os.path.join(hash_output_dir, hash_cache_file_name)

            if os.path.exists(hash_cache_path):
                continue

        try:
            conf["load_reaction"](channel)

        except Exception as e:
            print(traceback.format_exc())

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
