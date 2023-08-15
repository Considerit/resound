####################################
# Cross Expander Alignment algorithm
####################################

# A new cross-correlation algorithm for finding the sequences in a reaction video that match with a base video. 

# Get the first n seconds of the base audio. We'll call this the first chunk of the current segment. Use 
# cross-correlation to find the first match of this chunk in the reaction video that is within 
# peak_tolerance of the max match. Save the score of the max match as the current_sequence_match_score. 

# We're now going to see what portion of the next chunk of the base audio belongs in the current sequence. 
# So, set the next chunk of the base audio to the next n seconds. Correlate this next chunk with just the 
# next n seconds of the reaction video. Is the correlation strong enough to include with the previous chunk 
# as part of the current sequence? We can check by making sure the correlation is within peak_tolerance of 
# the current_sequence_match_score. If it is, we include this chunk into the current sequence. The crux 
# of the algorithm is in how we handle a poor match. 

# If it is a poor match, then we're going to try to identify exactly where in this chunk the sequence ends. 
# It is most likely that the first part of the chunk matches, and the latter part doesn't. We need to 
# find the transition. To do this, we can use binary search to progressively hone in on the transition point. 
# After we discover the poor match, we cut the chunk size in half to n / 2 seconds. Then we'll again use 
# cross-correlation to see if we can match this shorter chunk of the base audio to the corresponding part 
# of the reaction video. If so, then the transition point is probably between n / 2 seconds and n seconds, 
# and we can do the same thing for a chunk size of .75n. If not, then it is probably between 0 seconds and 
# n / 2 seconds, and we can instead do the same thing for a chunk size of .25n. We keep trying to hone in 
# on the transition point until we're within epsilon error tolerance. 

# Once we're satisfied with the transition point, we treat that transition point as the end of the current 
# segment. And then we take the next n seconds of the base video that has yet to be matched and repeat the 
# algorithm, making sure to perform the correlation only in area of the reaction video that hasn't yet 
# been matched. 


import os
import librosa
import copy
import pickle
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext


from typing import List, Tuple
import traceback

import cProfile
import pstats

from utilities import trim_and_concat_video, extract_audio, compute_precision_recall, universal_frame_rate, download_and_parse_reactions, is_close

from cross_expander.pruning_search import should_prune_path, initialize_path_pruning, prune_types, initialize_checkpoints, print_prune_data

from cross_expander.find_segment_start import find_next_segment_start_candidates, initialize_segment_start_cache
from cross_expander.find_segment_end import scope_segment, initialize_segment_end_cache
from cross_expander.scoring_and_similarity import path_score, find_best_path, initialize_path_score
from cross_expander.bounds import create_reaction_alignment_bounds, get_bound

from backchannel_isolator import audio_percentile_loudness


####################
# Recursively crawl through base and reaction audio, finding (and taking) multiple paths at each juncture.
# Returns a list of candidate paths. 
#
# get position in song & reaction
# get step
# get current path
#
# compute candidates (which might even be fill)
#
# if we've got more to go,
#   return result of each recurse on each candidate, on copies of current path (and try/catch each call to allow a path to fail)
# otherwise
#   return completed path / paths


profile_pathways = False
if profile_pathways:
    profiler = cProfile.Profile()

best_finished_path = {}
path_counts = {}
def initialize_path_counts():
    global path_counts 
    path_counts.clear()
    initialize_path_counts_for_depth(-1)
    initialize_path_counts_for_depth(0)

def initialize_path_counts_for_depth(depth): 
    global path_counts 
    path_counts[depth] = {
        'open': 0,
        'completed': 0,
        'current_starts': {}
    }

def open_path(depth, cnt=1):
    global path_counts
    path_counts[depth]['open'] += cnt
    path_counts[-1]['open'] += cnt

def complete_path(depth, cnt=1):
    global path_counts
    path_counts[depth]['completed'] += cnt
    path_counts[-1]['completed'] += cnt

def print_paths(current_start, reaction_start, depth, paths, path_counts, sr):
    print(f"Paths at point: {current_start / sr:.1f} {reaction_start / sr:.1f} [depth={depth}] [paths found here={len(paths)}]")

    try: 
        for ddd, progress in path_counts.items():
            if progress['open'] > 0:
                print(f"\tDepth {ddd}: [{progress['completed']} / {progress['open']} = {100 * progress['completed'] / progress['open']:.1f}%] [remaining={progress['open'] - progress['completed']}]")
            else:
                print(f"\tDepth {ddd}: [{progress['completed']} / {progress['open']}")

    except: 
        print("Could not print") 



def find_pathways(basics, options, current_path=None, current_path_checkpoint_scores=None, current_start=0, reaction_start=0): 
    global best_finished_path
    global path_counts

    # initializing
    if current_path is None: 
        current_path = []
        current_path_checkpoint_scores = {}


    base_audio = basics.get('base_audio')
    reaction_audio = basics.get('reaction_audio')
    sr = basics.get('sr')
    base_audio_mfcc = basics.get('base_audio_mfcc')
    base_audio_vol_diff = basics.get('song_percentile_loudness')
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    reaction_audio_vol_diff = basics.get('reaction_percentile_loudness')
    hop_length = basics.get('hop_length')

    step = first_n_samples = options.get('first_n_samples')

    depth = len(current_path)

    if profile_pathways:
        global profiler
        if depth == 0:
            profiler.enable()

    if depth not in path_counts:
        initialize_path_counts_for_depth(depth)

 
    if should_prune_path(basics, options, current_path, current_path_checkpoint_scores, best_finished_path, current_start, reaction_start, path_counts):
        return [None]
    

    # print(f"Starting pathway {current_path} {current_start} {reaction_start}")
    try: 


        current_end = min(current_start + step, len(base_audio))
        chunk = base_audio[current_start:current_end]
        current_chunk_size = current_end - current_start


        #######
        # Finished with this path!
        matched_all_of_base = current_start >= len(base_audio) - 1
        reaction_audio_depleted = reaction_start >= len(reaction_audio) - 1 or current_start > len(base_audio) - int(.5 * sr)
        if matched_all_of_base or reaction_audio_depleted:

            # Handle case where reaction video finishes before base video
            if reaction_audio_depleted:
                length_remaining = len(base_audio) - current_start - 1
                filler_segment = (reaction_start, reaction_start + length_remaining, current_start, current_start + length_remaining, True)
                current_path.append(  filler_segment   )
                # print(f"Reaction video finished before end of base video. Backfilling with base video ({length_remaining / sr}s).")
                # print(f"Backfilling {current_path} {current_start} {reaction_start}")

            score = path_score(current_path, basics)
            if 'score' not in best_finished_path or best_finished_path['score'][0] < score[0]:
                best_finished_path.update({
                    "path": current_path,
                    "score": score,
                    "partials": {}
                    })
                print(f"**** New best score is {best_finished_path['score']}")


            return [current_path]
        ###############


        alignment_bounds = options.get('alignment_bounds')
        if alignment_bounds is not None:
            upper_bound = get_bound(alignment_bounds, current_start, len(reaction_audio))

        candidate_starts = find_next_segment_start_candidates(
                                basics=basics, 
                                open_chunk=reaction_audio[reaction_start:], 
                                open_chunk_mfcc=reaction_audio_mfcc[:, round(reaction_start / hop_length):], 
                                open_chunk_vol_diff=reaction_audio_vol_diff[round(reaction_start / hop_length):],
                                closed_chunk=chunk, 
                                closed_chunk_mfcc= base_audio_mfcc[:, round(current_start / hop_length):round(current_end / hop_length) ],
                                closed_chunk_vol_diff=base_audio_vol_diff[round(current_start / hop_length):round(current_end / hop_length)],
                                current_chunk_size=current_chunk_size, 
                                peak_tolerance=options.get('peak_tolerance'),
                                open_start=reaction_start, 
                                closed_start=current_start, 
                                distance=first_n_samples, 
                                prune_for_continuity=True,
                                prune_types=prune_types,
                                upper_bound=upper_bound, 
                                filter_for_similarity=depth > 0, 
                                print_candidates=depth==0,
                                current_path=current_path  )

        # if depth == 0:
        #     candidate_starts = [candidate_starts[0], candidate_starts[1]]
        #     # candidate_starts = [0]

        #########
        # segment start pruned
        if candidate_starts == -1:
            return [None]

        #########
        # Could not find match in remainder of reaction video
        if candidate_starts is None:
            current_start += int(sr * .25)
            if (len(base_audio) - step - current_start) / sr < .5:
                # print(f"Could not find match in remainder of reaction video!!! Stopping.")
                # print(f"No match in remainder {current_path} {current_start} {reaction_start}")

                length_remaining = len(base_audio) - current_start - 1
                if length_remaining > 0:
                    filler_segment = (reaction_start, reaction_start + length_remaining, current_start, current_start + length_remaining, True)
                    current_path.append(  filler_segment   )

                return [current_path]
            else:
                # print(f"Could not find match in remainder of reaction video!!! Skipping forward.")
                open_path(depth)
                result = find_pathways(basics, options, current_path, copy.deepcopy(current_path_checkpoint_scores), current_start, reaction_start)
                complete_path(depth)
                return result
        #########



        paths = []
        open_path(depth, len(candidate_starts))

        for ci, candidate_segment_start in enumerate(candidate_starts):
            if depth == 0:
                print(f"STARTING DEPTH ZERO from {candidate_segment_start / sr}")


            segment, next_start, next_reaction_start, scores = scope_segment(basics, options, current_start, reaction_start, candidate_segment_start, current_chunk_size, prune_types)

            my_path = copy.deepcopy(current_path)
            if segment:
                my_path.append(segment)
            elif next_reaction_start < len(reaction_audio) - 1: 
                print("did not find segment", next_start, next_reaction_start)

            paths_this_direction = 0
            continued_paths = find_pathways(basics, options, my_path, copy.deepcopy(current_path_checkpoint_scores), next_start, next_reaction_start)
            for full_path in continued_paths:
                if full_path is not None:
                    paths_this_direction += 1
                    paths.append(full_path)

            complete_path(depth)


            if depth < 2 or path_counts[-1]['open'] % 10000 == 5000:
                print_paths(current_start, reaction_start, depth, paths, path_counts, sr)
                print_prune_data(basics)

            if depth == 0: 
                print("**************")
                print(f"Len of paths from {candidate_segment_start / sr} is {paths_this_direction}")
                # initialize_path_counts() # give each starting point a fair shot

            if profile_pathways and path_counts[-1]['completed'] % 1000 == 50:
                profiler.disable()
                stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
                stats.print_stats()
                profiler.enable()

        return paths

    except Exception as e: 
        print("Got error down this path")
        traceback.print_exc()
        print(e)
        return [None]





def cross_expander_aligner(base_audio, reaction_audio, sr, options, ground_truth=None):

    step_size = options.get('step_size')
    min_segment_length_in_seconds = options.get('min_segment_length_in_seconds')

    # Convert seconds to samples
    n_samples = int(step_size * sr)
    first_n_samples = int(min_segment_length_in_seconds * sr)

    hop_length = 256
    n_mfcc = 20

    base_audio_mfcc = librosa.feature.mfcc(y=base_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    reaction_audio_mfcc = librosa.feature.mfcc(y=reaction_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    song_percentile_loudness = audio_percentile_loudness(base_audio, loudness_window_size=100, percentile_window_size=1000, std_dev_percentile=None, hop_length=hop_length)
    reaction_percentile_loudness = audio_percentile_loudness(reaction_audio, loudness_window_size=100, percentile_window_size=1000, std_dev_percentile=None, hop_length=hop_length)

    basics = {
        "base_audio": base_audio,
        "base_audio_mfcc": base_audio_mfcc,
        "reaction_audio": reaction_audio,
        "reaction_audio_mfcc": reaction_audio_mfcc,
        "song_percentile_loudness": song_percentile_loudness,
        "reaction_percentile_loudness": reaction_percentile_loudness,
        "sr": sr,
        "hop_length": hop_length,
        "ground_truth": ground_truth
    }

    basics["checkpoints"] = initialize_checkpoints(basics)

    options['n_samples'] = n_samples
    options['first_n_samples'] = first_n_samples


    initialize_segment_start_cache()
    initialize_segment_end_cache()

    initialize_path_score()
    initialize_path_counts()
    initialize_path_pruning()

    best_finished_path.clear()


    options['alignment_bounds'] = create_reaction_alignment_bounds(basics, first_n_samples)

    paths = find_pathways(basics, options)

    path = find_best_path(paths, basics)

    sequences = compress_segments(path, sr=sr)

    reaction_sample_rate = Decimal(sr)
    final_sequences =          [ ( Decimal(s[0]) / reaction_sample_rate, Decimal(s[1]) / reaction_sample_rate, Decimal(s[2]) / reaction_sample_rate, Decimal(s[3]) / reaction_sample_rate, s[4]) for s in sequences ]
    
    if ground_truth:
        compute_precision_recall(sequences, ground_truth, tolerance=1.5)

    return final_sequences



def create_aligned_reaction_video(song:dict, react_video_ext, output_file: str, react_video, base_video, base_audio_data, base_audio_path, options, extend_by = 0):

    gt = song.get('ground_truth', {}).get(os.path.basename(react_video) )
    # if not gt: 
    #     return

    options.setdefault("step_size", 1)
    options.setdefault("min_segment_length_in_seconds", 3)
    options.setdefault("reverse_search_bound", options['min_segment_length_in_seconds'])
    options.setdefault("segment_end_backoff", 20000)
    options.setdefault("peak_tolerance", .5)
    options.setdefault("expansion_tolerance", .7)



    # Extract the reaction audio
    reaction_audio_data, reaction_sample_rate, reaction_audio_path = extract_audio(react_video)

    alignment_metadata_file = os.path.splitext(output_file)[0] + '.pckl'
    if not os.path.exists(alignment_metadata_file):

        # Determine the number of decimal places to try avoiding frame boundary errors given python rounding issues
        fr = Decimal(universal_frame_rate())
        precision = Decimal(1) / fr
        precision_str = str(precision)
        getcontext().prec = len(precision_str.split('.')[-1])


        print(f"\n*******{options}")
        final_sequences = cross_expander_aligner(base_audio_data, reaction_audio_data, sr=reaction_sample_rate, options=options, ground_truth=gt)
        

        print("\nsequences:")

        for sequence in final_sequences:
            print(f"\t{'*' if sequence[4] else ''}base: {float(sequence[2])}-{float(sequence[3])}  reaction: {float(sequence[0])}-{float(sequence[1])}")

        if options['output_alignment_metadata']:
            output_alignment_metadata(alignment_metadata_file, final_sequences)
    else: 
        final_sequences = read_alignment_metadata(alignment_metadata_file)

    if not os.path.exists(output_file) and options["output_alignment_video"]:
        # Trim and align the reaction video
        trim_and_concat_video(react_video, final_sequences, base_video, output_file, react_video_ext, extend_by = extend_by)
    return output_file


def output_alignment_metadata(output_file, final_sequences):
    temp_dir, _ = os.path.splitext(output_file)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    with open(output_file, 'wb') as f:
        pickle.dump(final_sequences, f)

def read_alignment_metadata(input_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data


def compress_segments(match_segments, sr):
    compressed_subsequences = []

    idx = 0 
    segment_groups = []
    current_group = []
    current_filler = match_segments[0][4]
    for current_start, current_end, current_base_start, current_base_end, filler in match_segments:
        if filler != current_filler:
            if len(current_group) > 0:
                segment_groups.append(current_group)
                current_group = []
            segment_groups.append([(current_start, current_end, current_base_start, current_base_end, filler)])
            current_filler = filler
        else: 
            current_group.append((current_start, current_end, current_base_start, current_base_end, filler))

    if len(current_group) > 0:
        segment_groups.append(current_group)


    for group in segment_groups:

        if len(group) == 1:
            compressed_subsequences.append(group[0])
            continue

        current_start, current_end, current_base_start, current_base_end, filler = group[0]
        for i, (start, end, base_start, base_end, filler) in enumerate(group[1:]):

            if start == current_end:
                # This subsequence is continuous with the current one, extend it
                # print("***COMBINING SEGMENT", current_end, start, (start - current_end), (start - current_end) / sr   )
                current_end = end
                current_base_end = base_end

                result = (current_base_end - current_base_start) / sr * universal_frame_rate()
                # print(f"is new segment whole? Is {current_base_end - current_base_start} [{(result)}] divisible by frame rate? {is_close(result, round(result))} ")
            else:
                # This subsequence is not continuous, add the current one to the list and start a new one
                compressed_segment = (current_start, current_end, current_base_start, current_base_end, filler)
                if not filler:
                    print(current_end, current_start, current_base_end, current_base_start)
                    # assert( is_close(current_end - current_start, current_base_end - current_base_start) )
                compressed_subsequences.append( compressed_segment )
                current_start, current_end, current_base_start, current_base_end, _ = start, end, base_start, base_end, filler

        # Add the last subsequence
        compressed_subsequences.append((current_start, current_end, current_base_start, current_base_end, filler))
        # print(f"{end - start} vs {base_end - base_start}"      )
        if not filler:
            if not is_close(current_end - current_start, current_base_end - current_base_start):
                print("NOT CLOSE!!!! Possible error", current_start, current_end - current_start, current_base_end - current_base_start)

    # compressed_subsequences = match_segments
    return compressed_subsequences

