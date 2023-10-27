import os
import copy
import glob

from prettytable import PrettyTable

from utilities import prepare_reactions, extract_audio, conf, make_conf, unload_reaction
from utilities import conversion_audio_sample_rate as sr
from inventory import download_and_parse_reactions, get_manifest_path
from aligner import create_aligned_reaction_video
from face_finder import create_reactor_view
from backchannel_isolator import isolate_reactor_backchannel
from compositor import compose_reactor_compilation
from aligner.scoring_and_similarity import print_path, ground_truth_overlap

from aligner.path_painter import paint_paths

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


def handle_reaction_video(reaction, compilation_exists, extend_by=15):

    output_file = reaction.get('aligned_path')

    # if '40' not in reaction['channel']:
    #     return

    print("processing ", reaction['channel'])
    # Create the output video file name

    


    create_aligned_reaction_video(reaction, extend_by=extend_by)



    if not conf["isolate_commentary"] or compilation_exists:
        return []


    _,_,aligned_reaction_audio_path = extract_audio(output_file, preserve_silence=True)

    reaction["aligned_audio_path"] = aligned_reaction_audio_path

    reaction["backchannel_audio"] = isolate_reactor_backchannel(reaction, extended_by=extend_by)
    
    if not conf["create_reactor_view"]:
        return []

    # backchannel_audio is used by create_reactor_view to replace the audio track of the reactor trace
    reaction["reactors"] = create_reactor_view(reaction, show_facial_recognition=False)

    if reaction['asides']:
        from moviepy.editor import VideoFileClip
        reaction["aside_clips"] = {}
        for i, aside in enumerate(reaction['asides']):
            start, end, insertion_point = aside

            # first, isolate the correct part of the reaction video
            aside_video_clip = os.path.join(conf.get('temp_directory'), f"{reaction.get('channel')}-aside-{i}.mp4")
            
            if not os.path.exists(aside_video_clip):            
                react_video = VideoFileClip(reaction.get('video_path'), has_mask=True)
                aside_clip = react_video.subclip(float(start), float(end))
                aside_clip.set_fps(30)
                aside_clip.write_videofile(aside_video_clip, codec="h264_videotoolbox", audio_codec="aac", ffmpeg_params=['-q:v', '40'])
                react_video.close()

            # second, do face detection on it
            reaction["aside_clips"][insertion_point] = create_reactor_view(reaction, show_facial_recognition=False, aside_video = aside_video_clip)





current_locks = {}

def request_lock(lock_str):
    full_output_dir = conf.get('temp_directory')
    lock_file = os.path.join(full_output_dir, f'locked-{lock_str}')
    if os.path.exists( lock_file  ):
      return False

    if lock_str == 'compilation':
        if len(other_locks()) > 0:
            return False

    global current_locks
    lock = open(lock_file, 'w')
    lock.write(f"yo")
    lock.close()
    current_locks[lock_str] = True

    return True

def free_lock(lock_str): 
    global current_locks
    full_output_dir = conf.get('temp_directory')
    lock_file = os.path.join(full_output_dir, f'locked-{lock_str}')
    if os.path.exists( lock_file  ):
        os.remove(lock_file)
    del current_locks[lock_str]

def free_all_locks():
    global current_locks
    locks = current_locks.keys()
    for lock in locks:
        free_lock(lock)

def other_locks():
    global current_locks
    full_output_dir = conf.get('temp_directory')
    # Use glob to get all files that match the pattern "locked-*"
    lock_files = glob.glob(os.path.join(full_output_dir, "locked-*"))
    
    # Strip off the "locked-" prefix and directory structure for each file
    stripped_files = [os.path.basename(f)[7:] for f in lock_files]

    return [lock for lock in stripped_files if lock not in current_locks ]

from moviepy.editor import VideoFileClip


def create_reaction_compilation(song_def:dict, progress, output_dir: str = 'aligned', include_base_video = True, options = {}):



    failed_reactions = []

    

    try:


        make_conf(song_def, options, output_dir)



        conf.setdefault("step_size", 1)
        conf.setdefault("min_segment_length_in_seconds", 3)
        conf.setdefault("reverse_search_bound", conf['min_segment_length_in_seconds'])
        conf.setdefault("peak_tolerance", .5)
        conf.setdefault("expansion_tolerance", .7)
        step_size = conf.get('step_size')
        min_segment_length_in_seconds = conf.get('min_segment_length_in_seconds')

        # Convert seconds to samples
        n_samples = int(step_size * sr)
        first_n_samples = int(min_segment_length_in_seconds * sr)

        conf['n_samples'] = n_samples
        conf['first_n_samples'] = first_n_samples


        temp_directory = conf.get("temp_directory")
        song_directory = conf.get('song_directory')


        compilation_path = conf.get('compilation_path')

        compilation_exists = os.path.exists(compilation_path)



        if request_lock('downloading'):

            print("Processing directory", song_directory, "Outputting to", output_dir)

            if not compilation_exists and (conf.get('download_and_parse', False) or conf.get('refresh_manifest', False)):
                download_and_parse_reactions(song_def['artist'], song_def['song'], song_def['search'], force=conf.get('refresh_manifest', False))
        else:
            print(f"...Skipping {song_def['song']} because another process is already working on this video")
            return []

        free_lock('downloading')


        conf.get('load_reactions')()
        
        if conf.get('only_manifest', False):
            return []


        extend_by = 12
        for i, (channel, reaction) in enumerate(conf.get('reactions').items()):
            if not request_lock(channel):
                continue


            # if reaction.get('channel') != "IamKing":
            #     continue



            try:
                # profiler = cProfile.Profile()
                # profiler.enable()

                handle_reaction_video(reaction, compilation_exists=compilation_exists, extend_by=extend_by)

                # profiler.disable()
                # stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
                # stats.print_stats()

            except Exception as e: 
                traceback.print_exc()
                print(e)
                traceback_str = traceback.format_exc()
                failed_reactions.append((reaction.get('channel'), e, traceback_str))
                conf['remove_reaction'](reaction.get('channel'))

            log_progress(progress)
            print_progress(progress)

            unload_reaction(channel)
            free_lock(channel)

        
        compilation_exists = os.path.exists(compilation_path)
        if not compilation_exists and conf['create_compilation'] and request_lock('compilation'):
            compose_reactor_compilation(extend_by=extend_by)
    

    except KeyboardInterrupt as e:
        free_all_locks()
        raise(e)

    except Exception as e:
        free_all_locks()
        traceback.print_exc()
        print(e)
        

    free_all_locks()
    return failed_reactions


def log_progress(progress):

    key = conf.get('song_key')

    if key not in progress: 
        progress[key] = {}

    for i, (channel, reaction) in enumerate(conf.get('reactions').items()):

        if reaction.get('best_path'):

            if reaction.get('ground_truth'):
                overlap = f"{ground_truth_overlap(reaction.get('best_path'), reaction.get('ground_truth')):.1f}%"
            else: 
                overlap = '-'


            target_score = reaction.get('target_score', None)
            best_observed_ground_truth = '-'
            best_local_ground_truth = '-'
            if target_score:
                if isinstance(target_score, float):
                    target_score = target_score
                else: 
                    if len(target_score) == 2:
                        target_score, best_observed_ground_truth = target_score
                    else: 
                        target_score, best_observed_ground_truth, best_local_ground_truth  = target_score



            progress[key][channel] = {
                'best_path': reaction.get('best_path'),
                'best_path_output': reaction.get('best_path_output'),
                'best_path_score': reaction.get('best_path_score'),                
                'alignment_duration': reaction.get('alignment_duration'),
                'target_score': target_score,
                'best_observed_ground_truth': best_observed_ground_truth,
                'best_local_ground_truth': best_local_ground_truth,                
                'ground_truth': reaction.get('ground_truth', None),
                'ground_truth_overlap': overlap,

            }

def print_progress(progress):

    # for song_key, alignments in progress.items():
    #     for channel, reaction in alignments.items():
    #         if reaction.get('best_path'):
    #             print(f"************* best path for {channel} / {song_key} ****************")
    #             print(reaction.get('best_path_output'))

    x = PrettyTable()
    x.field_names = ["Song", "Channel", "Duration", "Score", "Best Seen Score", "Ground Truth", "Local Best Ground Truth", "Best Seen Ground Truth"]
    x.align = "r"

    print("****************")
    print(f"Score Summary")

    for song_key, alignments in progress.items():
        for channel, reaction in alignments.items():
            print(reaction.get('best_path_score'))
            if reaction.get('best_path'):
                x.add_row([song_key, channel, f"{reaction.get('alignment_duration'):.1f}", f"{reaction.get('best_path_score')[0]:.3f}", reaction.get('target_score', None) or '-', reaction.get('ground_truth_overlap'), f"{reaction.get('best_local_ground_truth')}%" , f"{reaction.get('best_observed_ground_truth')}%"])
            else:
                x.add_row([song_key, channel,'-', '-', '-', reaction.get('target_score', None) or '-'])
    print(x)



import traceback
from library import get_library
if __name__ == '__main__':

    progress = {}

    songs, drafts, manifest_only, finished = get_library()

    output_dir = 'micro_aligned'

    for song in finished:
        clean_up(song)

    manifest_options = {
        "only_manifest": True,
        "refresh_manifest": True,
        "download_and_parse": True
    }

    failures = []
    for song in manifest_only: 
        print(f"Updating manifest for {song.get('song')}")
        failed = create_reaction_compilation(song, progress, output_dir = output_dir, options=manifest_options)
        if(len(failed) > 0):
            failures.append((song, failed)) 

    options = {
        "create_alignment": True,
        "save_alignment_metadata": True,
        "output_alignment_video": True,
        "isolate_commentary": False,
        "create_reactor_view": False,
        "create_compilation": False,
        "download_and_parse": False,
        "alignment_test": False,
        "force_ground_truth_paths": False,
        "draft": True,
        "paint_paths": True
    }
    failures = []
    for song in drafts: 
        failed = create_reaction_compilation(song, progress, output_dir = output_dir, options=options)
        if(len(failed) > 0):
            failures.append((song, failed)) 

    options['draft'] = False
    failures = []
    for song in songs: 
        failed = create_reaction_compilation(song, progress, output_dir = output_dir, options=options)
        if(len(failed) > 0):
            failures.append((song, failed)) 




    print(f"\n\nDone! {len(failures)} songs did not finish")

    for song, failed in failures:
        print(f"\n\n {len(failed)} Failures for song {song}")
        for react_video, e, trace in failed:
            print(f"\n***{react_video} failed with:")
            print(trace)
            print(e)
            print('*****')








