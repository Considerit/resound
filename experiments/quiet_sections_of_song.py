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

from silence import get_quiet_parts_of_song


def do_something_per_reaction():
    conf.get("load_base")()
    get_quiet_parts_of_song(visualize=True)


if __name__ == "__main__":
    songs = ["Ren - Kujo Beatdown"]
    songs = load_songs(songs)

    options = {}
    output_dir = "bounded"

    for song in songs:
        make_conf(song, options, output_dir)
        conf.get("load_reactions")()

        do_something_per_reaction()
