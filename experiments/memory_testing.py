import numpy as np
import librosa
import math
import os
import sys
import time

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

from pympler import asizeof

import gc
import sys

from collections import defaultdict

memory_tracker = defaultdict(list)


def something(reaction):
    music_video_path = conf.get("base_video_path")
    reaction_video_path = reaction.get("video_path")

    # time.sleep(2.5)
    # print(f"Total memory footprint of conf: {asizeof.asizeof(conf)/1000/1000}mb")


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

        try:
            conf["load_reaction"](channel)
        except:
            continue

        callback(reaction)

        unload_reaction(channel)

        if False:
            total_tracked = 0
            total_tracked_pimple = 0
            objects = gc.get_objects()
            uncounted = 0
            counted = 0
            for obj in objects:
                key = repr(obj)[0:300]
                size = sys.getsizeof(obj)
                total_tracked += size
                try:
                    size_pympler = asizeof.asizeof(obj)
                    counted += 1
                except Exception as e:
                    size_pympler = size
                    uncounted += 1

                total_tracked_pimple += size_pympler
                memory_tracker[key].append(size_pympler)

            print(counted, uncounted, uncounted / (counted + uncounted))

            conf_size = asizeof.asizeof(conf)
            print(
                f"\n\nTOTAL TRACKED: {total_tracked/1000/1000}mb TOTAL TRACKED PYMPLER: {total_tracked_pimple/1000/1000}mb  conf={conf_size/1000/1000}mb"
            )

            objs = [(k, v) for k, v in memory_tracker.items() if len(v) > 1]
            objs.sort(key=lambda x: x[1][-1], reverse=True)
            total_increase = total_increase_not_mono = 0

            objs = list(memory_tracker.items())
            objs.sort(key=lambda x: x[1][-1], reverse=True)

            # print("\nLarge and not changing")
            total_size = total_sizes_not_mono = total_sizes_mono = total_sizes_singular = 0
            num_same = 0
            for key, sizes in objs:
                if len(sizes) <= 1:
                    continue

                is_same = all(x == y for x, y in zip(sizes, sizes[1:]))

                if is_same:
                    num_same += 1
                    size = sizes[-1]
                    total_size += size
                    # if size > 1000000:
                    #     print(f"Object: {key}\n\tTotal Size: {sizes[-1]/1000/1000} mb")
            print(f"Same: total_size={total_size/1000/1000}mb")

            print("\n\nMonotonically increasing")
            for key, sizes in objs:
                if len(sizes) <= 1:
                    continue

                is_monotonically_increasing = all(x <= y for x, y in zip(sizes, sizes[1:]))

                if is_monotonically_increasing:
                    increase = sizes[-1] - sizes[-2]
                    total_sizes_mono += sizes[-1]
                    total_increase += increase

                    if increase > 500 * 1000000:
                        print(
                            f"Object: {key}\n\tTotal Size: {sizes[-1]/1000/1000} mb\n\tIncrease: {increase/1000/1000}mb"
                        )
            print(
                f"Mono: total_size={total_sizes_mono/1000/1000}mb total_increase={total_increase/1000/1000}mb"
            )

            # print("\n\nNOT monotonically increasing")
            for key, sizes in objs:
                if len(sizes) <= 1:
                    continue

                is_monotonically_increasing = all(x <= y for x, y in zip(sizes, sizes[1:]))

                if not is_monotonically_increasing:
                    increase = sizes[-1] - sizes[-2]
                    total_increase_not_mono += increase
                    total_sizes_not_mono += sizes[-1]
                    # if sizes[-1] > 1000000:
                    #     print(
                    #         f"Object: {key}\n\tTotal Size: {sizes[-1]/1000/1000} mb\n\tIncrease: {(sizes[-1]-sizes[-2])/1000/1000}mb"
                    #     )
            print(
                f"Not mono: total_size={total_sizes_not_mono/1000/1000}mb total_increase={total_increase_not_mono/1000/1000}mb"
            )

            # print("\n\nNOT monotonically increasing")
            for key, sizes in objs:
                if len(sizes) >= 2:
                    continue

                total_sizes_singular += sizes[-1]
                # if sizes[-1] > 1000000:
                #     print(
                #         f"Object: {key}\n\tTotal Size: {sizes[-1]/1000/1000} mb"
                #     )
            print(f"Singular: total_size={total_sizes_singular/1000/1000}mb")
            print("")


if __name__ == "__main__":
    songs = ["Ren - Kujo Beatdown"]
    songs = load_songs(songs)

    options = {}
    output_dir = "bounded"

    for song in songs:
        make_conf(song, options, output_dir)
        conf.get("load_reactions")()

        do_something_per_reaction(something)
