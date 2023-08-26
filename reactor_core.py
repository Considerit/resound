import os
import copy
import glob

from utilities import prepare_reactions, extract_audio
from inventory import download_and_parse_reactions, get_manifest_path
from cross_expander import create_aligned_reaction_video
from face_finder import create_reactor_view
from backchannel_isolator import process_reactor_audio
from compositor import compose_reactor_compilation

import cProfile
import pstats





def clean_up(song_def: dict):
    song = f"{song_def['artist']} - {song_def['song']}"
    song_directory = os.path.join('Media', song)

    print(f"Cleaning up {song}...")

    # Recursively find all .wav files starting from song_directory
    wav_files = glob.glob(f"{song_directory}/**/*.wav", recursive=True)

    # Delete each .wav file
    for wav_file in wav_files:
        try:
            os.remove(wav_file)
            print(f"\tDeleted: {wav_file}")
        except Exception as e:
            print(f"Error occurred while deleting file {wav_file}: {e}")


def handle_reaction_video(song:dict, output_dir: str, react_video, base_video, base_audio_data, base_audio_path, options, extend_by=15):


    react_video_name, react_video_ext = os.path.splitext(react_video)
    output_file = os.path.join(output_dir, os.path.basename(react_video_name) + f"-CROSS-EXPANDER.mp4")

    # if '40' not in react_video_name:
    #     return

    print("processing ", react_video_name)
    # Create the output video file name

    create_aligned_reaction_video(song, react_video_ext, output_file, react_video, base_video, base_audio_data, base_audio_path, options, extend_by=extend_by)

    if not options["isolate_commentary"]:
        return []

    _,sr,aligned_reaction_audio_path = extract_audio(output_file, preserve_silence=True)

    isolated_commentary = process_reactor_audio(output_dir, aligned_reaction_audio_path, base_audio_path, extended_by=extend_by, sr=sr)
    
    if not options["create_reactor_view"]:
        return []

    faces = create_reactor_view(output_file, base_video, replacement_audio=isolated_commentary, show_facial_recognition=False)

    return faces





from moviepy.editor import VideoFileClip

def create_reaction_compilations(song_def:dict, output_dir: str = 'aligned', include_base_video = True, options = {}):


    try:
        song = f"{song_def['artist']} - {song_def['song']}"
        song_directory = os.path.join('Media', song)
        reactions_dir = 'reactions'
        failed_reactions = []

        compilation_path = os.path.join(song_directory, f"{song} (compilation).mp4")
        if options.get('draft', False):
            compilation_path += "fast.mp4" 

        if os.path.exists(compilation_path):
          print("Compilation already exists", compilation_path)
          return []



        full_output_dir = os.path.join(song_directory, output_dir)
        if not os.path.exists(full_output_dir):
           # Create a new directory because it does not exist
           os.makedirs(full_output_dir)



        print("Processing directory", song_directory, "Outputting to", output_dir)

        lock_file = os.path.join(full_output_dir, 'locked')
        if os.path.exists( lock_file  ):
            print("...Skipping because another process is already working on this video")
            return []

        lock = open(lock_file, 'w')
        lock.write(f"yo")
        lock.close()


        if options.get('download_and_parse'):
            download_and_parse_reactions(song_def['artist'], song_def['song'], song_def['search'])

        if options.get('only_manifest', False):
            if os.path.exists(lock_file):
                os.remove(lock_file)

            return []

        manifest = open(get_manifest_path(song_def['artist'], song_def['song']), "r")


        base_video, react_videos = prepare_reactions(song_directory)


        # Extract the base audio and get the sample rate
        base_audio_data, _, base_audio_path = extract_audio(base_video)


        reaction_dir = os.path.join(song_directory, reactions_dir)

        reactors = []

        extend_by = 15
        for i, react_video in enumerate(react_videos):
            # if 'Cliff' not in react_video:
            #     continue

            try:
                # profiler = cProfile.Profile()
                # profiler.enable()

                faces = handle_reaction_video(song_def, full_output_dir, react_video, base_video, base_audio_data, base_audio_path, copy.deepcopy(options), extend_by=extend_by)
                
                if len(faces) > 1:
                    outputs = [{'file': f, 'group': i + 1} for f in faces]
                else:
                    outputs = [{'file': f} for f in faces]

                reactors.extend(outputs)



                # profiler.disable()
                # stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
                # stats.print_stats()

            except Exception as e: 
                traceback.print_exc()
                print(e)
                failed_reactions.append((react_video, e))

        reaction_videos = []
        for reactor in reactors: 
            input_file = reactor['file']

            featured = False
            
            for featured_r in song_def['featured']:
              if featured_r in input_file:
                featured = True
                break

            print(f"\tQueuing up {input_file}")

            reaction = {
                'key': input_file,
                'clip': VideoFileClip(input_file),
                'orientation': get_orientation(input_file),
                'group': reactor.get('group', None),
                'featured': featured
            }
            reaction_videos.append(reaction)

        if len(reaction_videos) > 0 and options['create_compilation']:
            compose_reactor_compilation(song_def, base_video, reaction_videos, compilation_path, options, extend_by=extend_by)
    except KeyboardInterrupt as e:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        else:
            print("Could not find lockfile to clean up")
        raise(e)

    except Exception as e:
        traceback.print_exc()
        print(e)
        

    if os.path.exists(lock_file):
        os.remove(lock_file)
    return failed_reactions

def get_orientation(input_file):
    base_name, ext = os.path.splitext(input_file)
    parts = base_name.split('-')
    orientation = parts[-1] if len(parts) > 3 else 'center'
    return orientation


import traceback
from library import get_library
if __name__ == '__main__':


    songs, drafts, manifest_only, finished = get_library()

    output_dir = "cheetah"

    for song in finished:
        clean_up(song)

    manifest_options = {
        "only_manifest": True,
    }

    failures = []
    for song in manifest_only: 
        failed = create_reaction_compilations(song, output_dir = output_dir, options=manifest_options)
        if(len(failed) > 0):
            failures.append((song, failed)) 

    options = {
        "output_alignment_metadata": True,
        "output_alignment_video": True,
        "isolate_commentary": True,
        "create_reactor_view": True,
        "create_compilation": True,
        "download_and_parse": True,
        "draft": True
    }
    failures = []
    for song in drafts: 
        failed = create_reaction_compilations(song, output_dir = output_dir, options=options)
        if(len(failed) > 0):
            failures.append((song, failed)) 

    options['draft'] = False
    failures = []
    for song in songs: 
        failed = create_reaction_compilations(song, output_dir = output_dir, options=options)
        if(len(failed) > 0):
            failures.append((song, failed)) 




    print(f"\n\nDone! {len(failures)} songs did not finish")

    for song, failed in failures:
        print(f"\n\n {len(failed)} Failures for song {song}")
        for react_video, e in failed:
            print(f"\n***{react_video} failed with:")
            print(e)







