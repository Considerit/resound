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
import matplotlib.pyplot as plt

from typing import List, Tuple
import traceback
import random

import cProfile
import pstats

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
                                print_candidates=depth==0  )

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

            paths_this_direction = []

            segment, next_start, next_reaction_start, scores = scope_segment(basics, options, current_start, reaction_start, candidate_segment_start, current_chunk_size, prune_types)

            my_path = copy.deepcopy(current_path)
            if segment:
                my_path.append(segment)
            elif next_reaction_start < len(reaction_audio) - 1: 
                print("did not find segment", next_start, next_reaction_start)


            continued_paths = find_pathways(basics, options, my_path, copy.deepcopy(current_path_checkpoint_scores), next_start, next_reaction_start)
            for full_path in continued_paths:
                if full_path is not None:
                    paths_this_direction.append(full_path)

            complete_path(depth)

            # Don't add duplicate paths
            for new_path in paths_this_direction:
                duplicate = False
                for other_path in paths:
                    if other_path == new_path:
                        print('SKIPPING DUPLICATE PATH', other_path, new_path)
                        duplicate = True
                        break
                if not duplicate:
                    # if depth == 0:
                    #     print(f"Adding path {new_path}")
                    paths.append(new_path)

            if depth < 2 or path_counts[-1]['open'] % 10000 == 5000:
                print_paths(current_start, reaction_start, depth, paths, path_counts, sr)
                print_prune_data(basics)

            if depth == 0: 
                print("**************")
                print(f"Len of paths from {candidate_segment_start / sr} is {len(paths_this_direction)}")
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





def cross_expander_aligner(base_audio, reaction_audio, sr, options):

    step_size = options.get('step_size')
    min_segment_length_in_seconds = options.get('min_segment_length_in_seconds')

    # Convert seconds to samples
    n_samples = int(step_size * sr)
    first_n_samples = int(min_segment_length_in_seconds * sr)

    hop_length = 256

    base_audio_mfcc = librosa.feature.mfcc(y=base_audio, sr=sr, n_mfcc=20, hop_length=hop_length)
    reaction_audio_mfcc = librosa.feature.mfcc(y=reaction_audio, sr=sr, n_mfcc=20, hop_length=hop_length)

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
        "hop_length": hop_length
    }

    basics["checkpoints"] = initialize_checkpoints(basics)

    options['n_samples'] = n_samples
    options['first_n_samples'] = first_n_samples

    options['alignment_bounds'] = create_reaction_alignment_bounds(basics, first_n_samples)


    initialize_path_score()
    initialize_segment_start_cache()
    initialize_path_pruning()
    initialize_segment_end_cache()
    initialize_path_counts()

    best_finished_path.clear()

    paths = find_pathways(basics, options)

    path = find_best_path(paths, basics)

    return path


