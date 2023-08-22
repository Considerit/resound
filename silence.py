
from pydub import AudioSegment
import subprocess

import os

def detect_leading_silence(sound, silence_threshold=-20.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def get_edge_silence(video_file, trim_end=True):
    _, video_ext = os.path.splitext(video_file)

    sound = AudioSegment.from_file(video_file, format=video_ext[1:])

    start_trim = detect_leading_silence(sound, silence_threshold=-30.0)
    if trim_end:
        end_trim = detect_leading_silence(sound.reverse(), silence_threshold=-30.0)
    else: 
        end_trim = 0

    start = start_trim
    end = len(sound) - end_trim

    start /= 1000
    end /= 1000


    print(f"Edge silence for {video_file}: ({start_trim}, {end_trim})")

    return (start, end)


def trim_silence(video_file, output_file):

    # Step 1: Detect silence
    start, end = get_edge_silence(video_file)

    # Step 2: Remove silence

    if start_trim > 0 or end_trim > 10:

        cmd = ['ffmpeg', '-i', video_file, '-ss', f"{start}", '-to', f"{end}", output_file]
        print(cmd)
        subprocess.run(cmd)

        print(f"Trimmed video saved as {output_file}")

        subprocess.run(['mv', output_file, video_file])


def trim_silence_from_all(directory: str, reactions_dir: str = 'reactions', output_dir: str = 'aligned'):
    reaction_dir = os.path.join(directory, output_dir)

    base_video = glob.glob(os.path.join(directory, f"{os.path.basename(directory)}.mp4"))
    if len(base_video) > 0: 
        base_video = base_video[0]
        base_video_name, base_video_ext = os.path.splitext(base_video)

        trim_silence(base_video, f"{base_video_name}-trimmed{base_video_ext}")

    # react_videos = glob.glob(os.path.join(reaction_dir, "*.mp4"))

    # for react_video in react_videos:
    #     react_video_name, react_video_ext = os.path.splitext(react_video)
    #     trim_silence(react_video, f"{react_video_name}-trimmed{react_video_ext}")



if __name__ == '__main__':

    songs = [ "Ren - Fire","Ren - Genesis", "Ren - The Hunger"]

    for song in songs: 
        trim_silence_from_all(song, "reactions", "aligned")
