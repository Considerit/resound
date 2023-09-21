import os
import copy
import glob

from prettytable import PrettyTable

from utilities import prepare_reactions, extract_audio, conf, make_conf
from inventory import download_and_parse_reactions, get_manifest_path
from cross_expander import create_aligned_reaction_video
from face_finder import create_reactor_view
from backchannel_isolator import process_reactor_audio
from compositor import compose_reactor_compilation
from cross_expander.scoring_and_similarity import print_path, ground_truth_overlap


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


def handle_reaction_video(reaction, extend_by=15):

    output_file = reaction.get('aligned_path')

    # if '40' not in reaction['channel']:
    #     return

    print("processing ", reaction['channel'])
    # Create the output video file name

    create_aligned_reaction_video(reaction, extend_by=extend_by)

    if not conf["isolate_commentary"]:
        return []

    conf['load_reaction'](reaction['channel'])

    _,_,aligned_reaction_audio_path = extract_audio(output_file, preserve_silence=True)
    reaction["aligned_audio_path"] = aligned_reaction_audio_path

    reaction["backchannel_audio"] = process_reactor_audio(reaction, extended_by=extend_by)
    
    if not conf["create_reactor_view"]:
        return []

    # backchannel_audio is used by create_reactor_view to replace the audio track of the reactor trace
    reaction["reactors"] = create_reactor_view(reaction, show_facial_recognition=False)



from moviepy.editor import VideoFileClip


def create_reaction_compilations(song_def:dict, output_dir: str = 'aligned', include_base_video = True, options = {}):



    failed_reactions = []


    try:


        locked = make_conf(song_def, options, output_dir)

        if locked:
            print(f"...Skipping {song_def['song']} because another process is already working on this video")
            return []


        temp_directory = conf.get("temp_directory")
        song_directory = conf.get('song_directory')

        lock_file = os.path.join(temp_directory, 'locked')


        lock = open(lock_file, 'w')
        lock.write(f"yo")
        lock.close()

        compilation_path = conf.get('compilation_path')

        if os.path.exists(compilation_path):
          print("Compilation already exists", compilation_path)
          return []

        print("Processing directory", song_directory, "Outputting to", output_dir)

        if conf.get('download_and_parse'):
            download_and_parse_reactions(song_def['artist'], song_def['song'], song_def['search'])

        if conf.get('only_manifest', False):
            if os.path.exists(lock_file):
                os.remove(lock_file)

            return []


        extend_by = 15
        for i, (name, reaction) in enumerate(conf.get('reactions').items()):

            # if 'Cliff' not in reaction.get('channel'):
            #     continue

            try:
                # profiler = cProfile.Profile()
                # profiler.enable()

                handle_reaction_video(reaction, extend_by=extend_by)

                # profiler.disable()
                # stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
                # stats.print_stats()

            except Exception as e: 
                traceback.print_exc()
                print(e)
                failed_reactions.append((reaction.get('channel'), e))

            print_progress()

            unload_reaction(name)


        if conf['create_compilation']:
            compose_reactor_compilation(extend_by=extend_by)
    

    except KeyboardInterrupt as e:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        else:
            print("Could not find lockfile to clean up")
        raise(e)

    except Exception as e:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        else:
            print("Could not find lockfile to clean up")

        traceback.print_exc()
        print(e)
        

    if os.path.exists(lock_file):
        os.remove(lock_file)
    return failed_reactions


def print_progress():
    for i, (name, reaction) in enumerate(conf.get('reactions').items()):
        if reaction.get('best_path'):
            print(f"************* best path for {name} ****************")
            print_path(reaction.get('best_path'), reaction)

    x = PrettyTable()
    x.field_names = ["Name", "Ground Truth", "Score", "Target Score"]
    x.align = "r"

    print("****************")
    print(f"Score Summary for reactions to {conf.get('song_key')}")

    for i, (name, reaction) in enumerate(conf.get('reactions').items()):

        if reaction.get('best_path'):
            if reaction.get('ground_truth'):
                overlap = f"{ground_truth_overlap(reaction.get('best_path'), reaction.get('ground_truth')):.1f}%"
            else: 
                overlap = '-'
            x.add_row([reaction.get('channel'),overlap, reaction.get('best_path_score')[0], reaction.get('target_score', None) or '-', ])
        else:
            x.add_row([reaction.get('channel'), '-', '-', reaction.get('target_score', None) or '-'])
    print(x)



import traceback
from library import get_library
if __name__ == '__main__':


    songs, drafts, manifest_only, finished = get_library()

    output_dir = "magnum"

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
        "create_alignment": True,
        "save_alignment_metadata": True,
        "output_alignment_video": True,
        "isolate_commentary": True,
        "create_reactor_view": True,
        "create_compilation": True,
        "download_and_parse": True,
        "alignment_test": False,
        "force_ground_truth_paths": False,
        "draft": False
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







