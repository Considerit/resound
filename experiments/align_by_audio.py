import os
import sys
import traceback
import glob
import shutil

sys.path.append(os.getcwd())


from utilities import conversion_audio_sample_rate as sr
from utilities import (
    conf,
    make_conf,
    read_object_from_file,
    save_object_to_file,
    unload_reaction,
)


from aligner.align_by_image.align_by_image import build_image_matches
from reactor_core import load_songs
from inventory.inventory import get_reactions_manifest

from aligner.create_alignment import create_aligned_reaction_video


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

        if channel not in ["AleksReacts"]:
            continue

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

        use_best_path_filter = False
        ###############################
        #### For examining best paths

        if use_best_path_filter:
            alignment_metadata_file = reaction.get("alignment_metadata")
            if os.path.exists(alignment_metadata_file):
                metadata = read_object_from_file(alignment_metadata_file)

                assert "best_path" in metadata

                best_path = metadata.get("best_path")
                last_segment = None
                error_detected = False
                print(f"Validating best path for {channel}")
                for segment in best_path:
                    reaction_start, reaction_end, music_start, music_end = segment[:4]

                    if last_segment:
                        if music_start < last_segment[3]:
                            print("\tERROR! BEND detected", last_segment, segment)
                            error_detected = True
                        if reaction_start < last_segment[1]:
                            print("\tERROR! BEND detected in reaction time", last_segment, segment)
                            error_detected = True

                    last_segment = segment

                if not error_detected:
                    continue
            else:
                continue

        if channel == jump_to:
            yet_to_encounter = False

        if jump_to and yet_to_encounter and channel not in [jump_to]:  # , "Bta Entertainment"]:
            continue

        ###############################
        # warning, this is destructive!
        to_delete = glob.glob(
            os.path.join(conf.get("temp_directory"), f"{channel}-CROSS-EXPANDER**"),
            recursive=False,
        )

        to_delete += glob.glob(
            f"{conf.get('temp_directory')}/{channel}-isolated-**", recursive=False
        )
        to_delete += glob.glob(
            os.path.join(conf.get("temp_directory"), f"{channel}-aside-**"), recursive=False
        )
        to_delete += glob.glob(f"{conf.get('temp_directory')}/{channel}-painting**", recursive=True)

        # cache_dir = os.path.join(conf.get("song_directory"), "_cache")
        # to_delete += glob.glob(os.path.join(cache_dir, f"{channel}-**"), recursive=False)

        for f in to_delete:
            print("DELETING", f)
            if os.path.exists(f):  # and "intercept_bounds" not in f:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
        ###############################
        continue

        try:
            conf["load_reaction"](channel)

        except Exception as e:
            print(traceback.format_exc())

            continue

        callback(reaction)

        unload_reaction(channel)


def something(reaction):
    create_aligned_reaction_video(reaction, extend_by=12, force=True)


if __name__ == "__main__":
    songs = ["Ren - Kujo Beatdown"]
    songs = load_songs(songs)

    options = {}
    output_dir = "bounded"

    for song in songs:
        make_conf(song, options, output_dir)
        conf.get("load_reactions")()

        do_something_per_reaction(something)
