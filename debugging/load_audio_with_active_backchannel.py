import argparse
import glob
import subprocess
import os
import librosa
import numpy as np
import soundfile as sf

sr = 44100


def contains_non_silent_audio(file_path, time, window, sr):
    if time == -1:
        return True

    # Calculate start and end in samples
    start_sample = int(max(0, (time - window / 2) * sr))
    end_sample = int((time + window / 2) * sr)


    
    # Load the specific window of the audio file
    y, _ = librosa.load(file_path, sr=sr, offset=start_sample/sr, duration=window)

    # y, _ = librosa.load(file_path, sr=sr)
    # y = y[start_sample:end_sample]


    # print(len(y)/sr, np.any(y != 0), start_sample, start_sample / sr, end_sample, end_sample / sr)

    # Check if the signal in that window is above a silence threshold
    return np.any(y != 0)

def call_applescript(script_path, working_directory, file_list):
    # Convert file list to a string to pass to AppleScript
    file_list_str = ", ".join(f'"{file}"' for file in file_list)
    # Run the AppleScript with the working directory and file list as arguments
    subprocess.run(['osascript', script_path, working_directory, file_list_str])

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--d', default="Ren - Money Game Part 3/bounded", type=str, help='Directory to search for audio files')
    parser.add_argument('--t', type=float, default=-1, help='Time in seconds')
    parser.add_argument('--w', default=30, type=int, help='Window width in seconds')

    args = parser.parse_args()

    base_directory = os.path.join('/Users/travis/Documents/code/resound/Media/', args.d)
    
    # Find all matching audio files
    search_pattern = os.path.join(base_directory, "*isolated_backchannel.wav")
    all_files = glob.glob(search_pattern)

    # Filter files based on the presence of non-silent audio in the specified window
    if args.t != -1:
        file_list = [f for f in all_files if contains_non_silent_audio(f, args.t, args.w, sr)]
    else:
        file_list = all_files

    # Convert full paths back to filenames for AppleScript
    file_list = [os.path.basename(f) for f in file_list]

    # Debugging output
    for file in file_list:
        print(file)

    # Path to the AppleScript
    script_path = "./debugging/launch_audacity_with_audio.applescript"

    # Call the AppleScript with the directory and list of files
    if file_list:
        call_applescript(script_path, base_directory, file_list)
    else:
        print("No matching files found.")

if __name__ == "__main__":
    main()
