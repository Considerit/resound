import noisereduce as nr
import soundfile as sf
import os, shutil
import subprocess
import shlex

from moviepy.editor import VideoFileClip


from utilities.audio_processing import highpass_filter, convert_to_mono
from utilities import conf, conversion_audio_sample_rate as sr

import demucs.separate

####################################
# Track separation and high pass filtering


def run_demucs(audio_path, output_dir):
    demucs.separate.main(
        [
            "--two-stems",
            "vocals",
            # "-n",
            # "mdx_extra",
            "-j",
            "7",
            "-d",
            "cpu",
            "-o",
            f"{output_dir}",
            f"{audio_path}",
        ]
    )


accompaniment_filename = "accompaniment.wav"
vocals_filename = "vocals.wav"
high_passed_vocals_filename = "vocals-post-high-passed.wav"


def separate_vocals_for_song():
    base_audio = conf.get("base_audio_path")
    output_dir = conf.get("temp_directory")

    base_dir = os.path.splitext(base_audio)[0].split("/")[-1]

    separation_path = os.path.join(output_dir, base_dir)
    song_length = conf.get("song_length") / sr + 1

    return separate_vocals_for_path(separation_path, base_audio, song_length)


def separate_vocals_for_aligned_reaction(reaction, aligned_reaction_path):
    output_dir = conf.get("temp_directory")

    separation_path = os.path.join(
        output_dir, os.path.splitext(aligned_reaction_path)[0].split("/")[-1]
    )

    song_length = conf.get("song_length") / sr + 1

    return separate_vocals_for_path(separation_path, aligned_reaction_path, song_length)


def separate_vocals_for_reaction(reaction):
    reaction_audio = reaction.get("reaction_audio_path")
    output_dir = conf.get("temp_directory")

    separation_path = os.path.join(
        output_dir, f"{os.path.splitext(reaction_audio)[0].split('/')[-1]}-full"
    )

    reaction_length = len(reaction.get("reaction_audio_data")) / sr + 1

    return separate_vocals_for_path(separation_path, reaction_audio, reaction_length)


def separate_vocals_for_path(separation_path, audio_path, audio_length):
    accompaniment_path = os.path.join(separation_path, accompaniment_filename)
    vocals_path = os.path.join(separation_path, vocals_filename)
    # high_passed_vocals_path = os.path.join(separation_path, high_passed_vocals_filename)

    if not os.path.exists(vocals_path):
        separate_vocals(
            separation_path,
            audio_path,
            duration=audio_length,
        )

    return accompaniment_path, vocals_path, separation_path


def separate_vocals(output_dir, audio_path, duration=None):
    # Create a separator with 2 stems (vocals and accompaniment)

    # vocals_high_passed_path = os.path.join(output_dir, output_filename)
    # # Post process
    # if not os.path.exists(vocals_high_passed_path):
    #     print("vocals highpassed path", vocals_high_passed_path)

    # Load the separated tracks
    vocals_path = os.path.join(output_dir, "vocals.wav")

    # Perform source separation
    if not os.path.exists(vocals_path):
        print("\tPerforming source separation")

        run_demucs(audio_path, output_dir)

        audio_file_prefix = os.path.splitext(audio_path)[0].split("/")[-1]
        demucs_outputdir = os.path.join(output_dir, "htdemucs")
        demucs_outputpath = os.path.join(demucs_outputdir, audio_file_prefix, "vocals.wav")

        os.rename(demucs_outputpath, vocals_path)

        shutil.rmtree(demucs_outputdir)

        print("\tDone with source separation")

        # vocals, sr_song = librosa.load( vocals_path, sr=sr, mono=True )
        # vocals = post_process_audio(vocals)
        # sf.write(song_vocals_high_passed_path, vocals.T, sr)

        # print("\tPost processing vocals", vocals_path)
        # audio_data, __ = sf.read(vocals_path)
        # vocals = convert_to_mono(audio_data)
        # vocals = post_process_audio(vocals)
        # sf.write(vocals_high_passed_path, vocals, sr)
        # print("\tDone post processing vocals")


def post_process_audio(commentary):
    # mono = commentary.ndim == 1

    # # If the audio is mono (1D array), add an extra dimension to make it look like single-channel multi-channel
    # if mono:
    #     commentary = np.expand_dims(commentary, axis=0)

    # processed_commentary = np.zeros_like(commentary)

    # for channel in range(commentary.shape[0]):
    #     # Reduce noise
    #     reduced_noise = nr.reduce_noise(commentary[channel], sr=sr)

    #     # Apply a highpass filter to remove low-frequency non-speech components
    #     sos = scipy.signal.butter(10, 100, 'hp', fs=sr, output='sos')
    #     filtered = scipy.signal.sosfilt(sos, reduced_noise)

    #     processed_commentary[channel] = filtered

    # if mono:
    #     processed_commentary = processed_commentary[0]

    # return processed_commentary

    commentary = highpass_filter(commentary, cutoff=100, fs=sr)

    # Reduce noise
    commentary = nr.reduce_noise(commentary, sr=sr)

    # # Apply a highpass filter to remove low-frequency non-speech components
    # sos = scipy.signal.butter(10, 100, 'hp', fs=sr, output='sos')
    # commentary = scipy.signal.sosfilt(sos, commentary)

    return commentary
