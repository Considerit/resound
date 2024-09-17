import noisereduce as nr
import soundfile as sf
import os
import subprocess
import shlex

from moviepy.editor import VideoFileClip


from utilities.audio_processing import highpass_filter, convert_to_mono
from utilities import conf, conversion_audio_sample_rate as sr


####################################
# Track separation and high pass filtering


# the spleeter python interface hangs a lot and seems to have some memory leaks, so I'm just going to call
# it directly from the command line via subprocess
# spleeter_separator = None
# def get_spleeter():
#     global spleeter_separator
#     if spleeter_separator is None:
#         from spleeter.separator import Separator
#         spleeter_separator = Separator('spleeter:2stems')
#     return spleeter_separator


def run_spleeter(audio_path, output_dir, duration):
    # audio_path = shlex.quote(audio_path)
    # output_dir = shlex.quote(output_dir)

    # get_spleeter().separate_to_file(audio_path, output_dir, duration=duration)
    command = f'spleeter separate -o "{output_dir}" -d {duration} "{audio_path}"'

    # command = shlex.quote(command)
    subprocess.check_output(["zsh", "-c", command], text=True)


accompaniment_filename = "accompaniment.wav"
vocals_filename = "vocals.wav"
high_passed_vocals_filename = "vocals-post-high-passed.wav"


def separate_vocals_for_song():
    base_audio = conf.get("base_audio_path")

    base_dir = os.path.splitext(base_audio)[0].split("/")[-1]

    separation_path = os.path.join(output_dir, base_dir)
    song_length = len(conf.get("song_audio_data")) / sr + 1

    return separate_vocals_for_path(separation_path, base_audio, song_length)


def separate_vocals_for_aligned_reaction(reaction):
    aligned_reaction_audio = reaction.get("aligned_audio_path")
    output_dir = conf.get("temp_directory")

    separation_path = os.path.join(
        output_dir, os.path.splitext(reaction_audio)[0].split("/")[-1]
    )

    song_length = len(conf.get("song_audio_data")) / sr + 1

    return separate_vocals_for_path(
        separation_path, aligned_reaction_audio, song_length
    )


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
    high_passed_vocals_path = os.path.join(separation_path, high_passed_vocals_filename)

    if not os.path.exists(high_passed_vocals_filename):
        separate_vocals(
            separation_path,
            audio_path,
            high_passed_vocals_filename,
            duration=audio_length,
        )

    return accompaniment_path, vocals_path, high_passed_vocals_path, separation_path


def separate_vocals(output_dir, audio_path, output_filename, duration=None):
    # Create a separator with 2 stems (vocals and accompaniment)

    vocals_high_passed_path = os.path.join(output_dir, output_filename)
    # Post process
    if not os.path.exists(vocals_high_passed_path):
        print("vocals highpassed path", vocals_high_passed_path)

        # Load the separated tracks
        vocals_path = os.path.join(output_dir, "vocals.wav")

        # Perform source separation
        if not os.path.exists(vocals_path):
            print("\tPerforming source separation")

            if duration is None:
                with sf.SoundFile(audio_path) as f:
                    frames = f.frames
                    rate = f.samplerate
                    duration = frames / float(rate)

            run_spleeter(audio_path, output_dir, duration)

            audio_file_prefix = os.path.splitext(audio_path)[0].split("/")[-1]
            weird_spleeter_outputpath = os.path.join(output_dir, audio_file_prefix)

            parent_dir = os.path.dirname(weird_spleeter_outputpath)
            for filename in os.listdir(weird_spleeter_outputpath):
                current_path = os.path.join(weird_spleeter_outputpath, filename)
                new_path = os.path.join(parent_dir, filename)
                os.rename(current_path, new_path)
            print("\tDone with source separation")

        # vocals, sr_song = librosa.load( vocals_path, sr=sr, mono=True )
        # vocals = post_process_audio(vocals)
        # sf.write(song_vocals_high_passed_path, vocals.T, sr)

        print("\tPost processing vocals", vocals_path)
        audio_data, __ = sf.read(vocals_path)
        vocals = convert_to_mono(audio_data)
        vocals = post_process_audio(vocals)
        sf.write(vocals_high_passed_path, vocals, sr)
        print("\tDone post processing vocals")


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
