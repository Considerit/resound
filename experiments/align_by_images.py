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


from aligner.align_by_image.align_by_image import build_image_matches
from reactor_core import load_songs
from inventory.inventory import get_reactions_manifest

from aligner.align_by_image.embedded_video_finder import get_bounding_box_of_music_video_in_reaction
from utilities.utilities import check_and_fix_fps


def do_something_per_reaction(callback):
    all_reactions = list(conf.get("reactions").keys())
    all_reactions.sort()
    all_reactions.reverse()

    reactions_manifest = get_reactions_manifest(conf.get("artist"), conf.get("song_name"))[
        "reactions"
    ]
    all_manifests = {}
    for vidID, reaction_manifest in reactions_manifest.items():
        reaction_file_prefix = reaction_manifest.get("file_prefix", reaction_manifest["reactor"])
        all_manifests[reaction_file_prefix] = reaction_manifest

    yet_to_encounter = True
    jump_to = None

    lags = []
    for i, channel in enumerate(all_reactions):
        reaction = conf.get("reactions").get(channel)
        manifest = all_manifests[channel]

        # if not manifest.get("alignment_done"):
        #     continue

        # if channel not in [
        #     "1K Z A Y"
        # ]:  # , "AR", "Chris Liepe", "LEE REACTS - IMR Media & Gaming"]:
        #     continue

        if channel not in [
            # "1K Z A Y",
            # "BigNickOfficial",
            # "Bisscute",
            # "Adnan Reacts",
            # "CADZ Crew",
            # "COM8E Reacts",  # bad transitions
            # "DLace Reacts",
            # "DonVon",
            "Elmo Reacts",  # bad transitions
            # "Face Famous",
            # "FrankValchiria",
            # # "Jahherbz",
            # "JND",  # transitions
            # "JördzReacts",  ## didnt' finish
            # "LEE REACTS - IMR Media & Gaming",
            # "LucieV Reacts",
            # "Mr Network",  # There's a middle section that is slightly misaligned
            # "MRLBOYD MUSIC",  # Still poorly chosen (maybe even filtered)
            # "Niloyasha",
            # "RikaShae",
            # "Shadow Aeternum",  # Slow
            # "Simply Not Simple",
            # "SnakeVenomV",  # Slow
            # "TeeSpicerReacts",  # Slow
            # "TheReactTwinss",
            # "UglyAceEnt",
            # "Youngblood Poetry",
        ]:
            continue
        if channel == jump_to:
            yet_to_encounter = False

        if jump_to and yet_to_encounter and channel not in [jump_to]:  # , "Bta Entertainment"]:
            continue

        # hash_output_dir = os.path.join(conf.get("temp_directory"), "image_hashes")
        # crop_coordinates = get_bounding_box_of_music_video_in_reaction(reaction)

        # if crop_coordinates is not None:
        #     print("Found embedded music video at", crop_coordinates)

        #     hash_cache_file_name = f"{channel}-{2}fps-{tuple(crop_coordinates)}-{0.95}.json"
        #     hash_cache_path = os.path.join(hash_output_dir, hash_cache_file_name)

        #     if os.path.exists(hash_cache_path):
        #         continue

        try:
            conf["load_reaction"](channel)

        except Exception as e:
            print(traceback.format_exc())

            continue

        # check_and_fix_fps(reaction.get("video_path"))

        # conf["load_base"]()

        callback(reaction)
        # if type(lag) == float or type(lag) == int:
        #     lags.append((channel, lag / sr))
        #     print("")
        #     for channel, lag in lags:
        #         print(f"{lag} ({channel})")
        #     print("")
        # else:
        #     print(lag)

        unload_reaction(channel)


def something(reaction):
    return build_image_matches(reaction, visualize=True)


if __name__ == "__main__":
    songs = ["Ren - Kujo Beatdown"]
    songs = load_songs(songs)

    options = {}
    output_dir = "bounded"

    for song in songs:
        make_conf(song, options, output_dir)
        conf.get("load_reactions")()

        do_something_per_reaction(something)
