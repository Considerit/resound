import os
import copy
import glob
import json 

from prettytable import PrettyTable

from utilities import prepare_reactions, extract_audio, conf, make_conf, unload_reaction
from utilities import conversion_audio_sample_rate as sr
from inventory import download_and_parse_reactions, get_manifest_path, filter_and_augment_manifest
from aligner import create_aligned_reaction_video
from face_finder import create_reactor_view
from backchannel_isolator import isolate_reactor_backchannel
from compositor import compose_reactor_compilation
from aligner.scoring_and_similarity import print_path, ground_truth_overlap

from aligner.path_painter import paint_paths

import cProfile
import pstats

from utilities import print_profiling

def clean_up(song_def: dict):
    song = f"{song_def['artist']} - {song_def['song']}"
    song_directory = os.path.join('Media', song)

    print(f"Cleaning up {song}...")

    wav_files = glob.glob(f"{song_directory}/**/*.wav", recursive=True)

    # Delete each .wav file
    for wav_file in wav_files:
        if 'isolated_backchannel' in wav_file:
            continue

        try:
            os.remove(wav_file)
            print(f"\tDeleted: {wav_file}")
        except Exception as e:
            print(f"Error occurred while deleting file {wav_file}: {e}")


    mp4_files = glob.glob(f"{song_directory}/**/*CROSS-EXPANDER*.mp4", recursive=True)

    for mp4 in mp4_files:
        if 'cropped' in mp4:
            continue

        try:
            os.remove(mp4)
            print(f"\tDeleted: {mp4}")
        except Exception as e:
            print(f"Error occurred while deleting file {mp4}: {e}")

    webm_files = glob.glob(f"{song_directory}/**/*.webm", recursive=True)

    for webm in webm_files:
        try:
            os.remove(webm)
            print(f"\tDeleted: {webm}")
        except Exception as e:
            print(f"Error occurred while deleting file {webm}: {e}")

def handle_reaction_video(reaction, compilation_exists, extend_by=15):

    output_file = reaction.get('aligned_path')

    # if '40' not in reaction['channel']:
    #     return

    # print("processing ", reaction['channel'])
    # Create the output video file name

    


    create_aligned_reaction_video(reaction, extend_by=extend_by)



    if not conf["isolate_commentary"]:
        return []


    _,_,aligned_reaction_audio_path = extract_audio(output_file, preserve_silence=True)

    reaction["aligned_audio_path"] = aligned_reaction_audio_path

    reaction["backchannel_audio"] = isolate_reactor_backchannel(reaction, extended_by=extend_by)
    
    if not conf["create_reactor_view"]:
        return []

    # backchannel_audio is used by create_reactor_view to replace the audio track of the reactor trace
    reaction["reactors"], __ = create_reactor_view(reaction)

    if reaction['asides']:
        from moviepy.editor import VideoFileClip, concatenate_videoclips

        reaction["aside_clips"] = {}
        # print(f"Asides: {reaction.get('channel')} has {len(reaction['asides'])}")

        all_asides = reaction['asides']

        def consolidate_adjacent_asides(all_asides):
            groups = []
            all_asides.sort(key=lambda x: x[0])  # sort by start value of the aside

            group_when_closer_than = 0.1  # put asides into the same group if one's start value 
                                          # is less than group_when_closer_than greater than the 
                                          # end value of the previous aside 

            # Initialize the first group
            current_group = []

            # Group asides
            for aside in all_asides:
                aside = start, end, insertion_point, rewind = get_aside_config(aside)

                # If current_group is empty or the start of the current aside is close enough to the end of the last aside in the group
                if not current_group or insertion_point - current_group[-1][2] < group_when_closer_than:
                    current_group.append(aside)
                else:
                    # Current aside starts a new group
                    groups.append(current_group)
                    current_group = [aside]

            # Add the last group if it's not empty
            if current_group:
                groups.append(current_group)


            consolidated_asides = []
            for group in groups:
                skip_segments = []

                # Get the start of the first aside and the end of the last aside in the group
                start, end = group[0][0], group[-1][1]

                # Iterate through the group and find gaps between asides
                for i in range(len(group) - 1):
                    current_end = group[i][1]
                    next_start = group[i + 1][0]
                    if next_start > current_end:
                        skip_segments.append((current_end - start, next_start - start))

                aside_conf = {
                    'range': (start, end),
                    'insertion_point': group[-1][2],
                    'rewind': group[-1][3],
                    'skip_segments': skip_segments
                }
                consolidated_asides.append(aside_conf)

            return consolidated_asides

            
        def get_aside_config(aside):
            if len(aside) == 4:
                start, end, insertion_point, rewind = aside
            else: 
                start, end, insertion_point = aside
                rewind = 3

            return start, end, insertion_point, rewind

        def remove_skipped_segments(aside_reactor_views, skip_segments):
            skip_segments.sort( key=lambda x: x[0], reverse=True )

            for reactor_view in aside_reactor_views:
                # Initialize a list to hold the subclips
                subclips = []
                last_end = 0  # Keep track of the end of the last segment added to subclips

                for start, end in skip_segments:
                    # Add the segment from the last end to the current start
                    if start > last_end:
                        subclips.append(reactor_view['clip'].subclip(last_end, start))

                    last_end = end

                # Add the remaining part of the clip after the last skip segment
                if last_end < reactor_view['clip'].duration:
                    subclips.append(reactor_view['clip'].subclip(last_end, reactor_view['clip'].duration))

                # Concatenate all the subclips together
                reactor_view['clip'] = concatenate_videoclips(subclips)

            return aside_reactor_views


        consolidated_asides = consolidate_adjacent_asides(all_asides)

        for i, aside in enumerate(consolidated_asides):

            start, end = aside['range']
            insertion_point = aside['insertion_point']
            rewind = aside['rewind']
            skip_segments = aside['skip_segments']

            # first, isolate the correct part of the reaction video
            aside_video_clip = os.path.join(conf.get('temp_directory'), f"{reaction.get('channel')}-aside-{i}.mp4")
            
            if not os.path.exists(aside_video_clip):            
                react_video = VideoFileClip(reaction.get('video_path'), has_mask=True)
                aside_clip = react_video.subclip(float(start), float(end))
                aside_clip.set_fps(30)
                aside_clip.write_videofile(aside_video_clip, codec="h264_videotoolbox", audio_codec="aac", ffmpeg_params=['-q:v', '40'])
                react_video.close()

            # do face detection on it
            aside_reactor_views, __ = create_reactor_view(reaction, aside_video = aside_video_clip, show_facial_recognition=False)
            aside_reactor_views = remove_skipped_segments(aside_reactor_views, skip_segments)

            song_audio_data, _, _ = extract_audio(aside_video_clip, conf.get('temp_directory'), sr, convert_to_mono=False)            

            for reactor_view in aside_reactor_views: 
                reactor_view['audio'] = reactor_view['clip'].audio
                reactor_view['clip']  = reactor_view['clip'].without_audio()

            reaction["aside_clips"][insertion_point] = (aside_reactor_views, rewind)





current_locks = {}


def is_locked(lock_str):
    full_output_dir = conf.get('temp_directory')
    lock_file = os.path.join(full_output_dir, f'locked-{lock_str}')
    if os.path.exists( lock_file  ):
      return True
    return False


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
    locks = list(current_locks.keys())
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

        if is_locked('compilation'):
            return []

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

            if conf.get('refresh_manifest', False) or (not compilation_exists and (conf.get('download_and_parse', False))):
                download_and_parse_reactions(song_def, song_def['artist'], song_def['song'], song_def.get('song_search', f"{song_def.get('artist')} {song_def.get('song')}"), song_def['search'], refresh_manifest=conf.get('refresh_manifest', False))
        
                print("Filtering and augmenting manifest", song_directory, "Outputting to", output_dir)
                filter_and_augment_manifest(song_def['artist'], song_def['song'])

        else:
            print(f"...Skipping {song_def['song']} because another process is already working on this video")
            return []

        free_lock('downloading')





        conf.get('load_reactions')()
        
        if conf.get('only_manifest', False):
            return []


        extend_by = 12

        all_reactions = list(conf.get('reactions').keys())
        all_reactions.sort()

        for i, channel in enumerate(all_reactions):
            reaction = conf.get('reactions').get(channel)

            print_profiling()

            if not request_lock(channel):
                continue


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
                if conf.get('break_on_exception'):
                    raise(e)

            log_progress(progress)

            unload_reaction(channel)
            free_lock(channel)

        print_progress(progress)        
        compilation_exists = os.path.exists(compilation_path)
        print("COMP EXISTS?", compilation_exists, compilation_path)
        if not compilation_exists and conf['create_compilation'] and request_lock('compilation'):

            # for channel, reaction in conf.get('reactions').items():
            #     conf['load_reaction'](channel) # make sure all reactions are loaded

            compose_reactor_compilation(extend_by=extend_by)
            free_lock('compilation')
    

    except KeyboardInterrupt as e:
        free_all_locks()
        raise(e)

    except Exception as e:
        free_all_locks()
        traceback.print_exc()
        print(e)
        if conf.get('break_on_exception'):
            raise(e)
        

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
            if reaction.get('best_path'):
                x.add_row([song_key, channel, f"{reaction.get('alignment_duration'):.1f}", f"{reaction.get('best_path_score')[0]:.3f}", reaction.get('target_score', None) or '-', reaction.get('ground_truth_overlap'), f"{reaction.get('best_local_ground_truth')}%" , f"{reaction.get('best_observed_ground_truth')}%"])
            else:
                x.add_row([song_key, channel,'-', '-', '-', reaction.get('target_score', None) or '-'])
    print(x)




results_output_dir = 'bounded'


import traceback

if __name__ == '__main__':

    def load_songs(lst):
        loaded = []
        for s in lst:
            f = os.path.join(f'library/{s}.json')
            print(f)
            defn = json.load( open(f))
            loaded.append( defn  )
        return loaded

    from library import songs, drafts, refresh_manifest, finished

    songs = load_songs(songs)
    drafts = load_songs(drafts)
    refresh_manifest = load_songs(refresh_manifest)
    finished = load_songs(finished)

    progress = {}

    for song in finished:
        clean_up(song)

    manifest_options = {
        "only_manifest": True,
        "refresh_manifest": True,
        "download_and_parse": True
    }

    failures = []
    for song in refresh_manifest: 
        print(f"Updating manifest for {song.get('song')}")
        failed = create_reaction_compilation(song, progress, output_dir = results_output_dir, options=manifest_options)
        if(len(failed) > 0):
            failures.append((song, failed)) 
        conf['free_conf']()


    options = {
        "create_alignment": True,
        "save_alignment_metadata": True,
        "output_alignment_video": True,
        "isolate_commentary": True,
        "create_reactor_view": True,
        "create_compilation": True,
        "download_and_parse": True,
        "alignment_test": False,
        "draft": True,
        "break_on_exception": False,
    }
    failures = []
    for song in drafts: 
        failed = create_reaction_compilation(song, progress, output_dir = results_output_dir, options=options)
        if(len(failed) > 0):
            failures.append((song, failed)) 

    options['draft'] = False
    failures = []
    for song in songs: 
        failed = create_reaction_compilation(song, progress, output_dir = results_output_dir, options=options)
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








