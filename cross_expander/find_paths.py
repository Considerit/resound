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
import gc
import librosa
import copy
import random
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
import numpy as np


from typing import List, Tuple
import traceback


from cross_expander.create_trimmed_video import trim_and_concat_video

from utilities import extract_audio, compute_precision_recall, universal_frame_rate, is_close, print_memory_consumption, on_press_key, print_profiling
from utilities import save_object_to_file, read_object_from_file

from cross_expander.pruning_search import should_prune_path, initialize_path_pruning, prune_types, initialize_checkpoints, print_prune_data, find_last_checkpoint_crossed

from cross_expander.find_segment_start import find_next_segment_start_candidates, initialize_segment_start_cache
from cross_expander.find_segment_end import scope_segment, initialize_segment_end_cache
from cross_expander.scoring_and_similarity import path_score, find_best_path, initialize_path_score, print_path, calculate_partial_score
from cross_expander.scoring_and_similarity import initialize_segment_tracking, truncate_path, append_or_extend_segment
from cross_expander.bounds import create_reaction_alignment_bounds, get_bound




from utilities.audio_processing import audio_percentile_loudness


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
    if depth not in path_counts:
        initialize_path_counts_for_depth(depth)    
    path_counts[depth]['completed'] += cnt
    path_counts[-1]['completed'] += cnt

def print_paths(current_start, reaction_start, depth, path_counts, sr):
    print(f"Paths at point: {current_start / sr:.1f} {reaction_start / sr:.1f} [depth={depth}]")

    try: 
        for ddd, progress in path_counts.items():
            if progress['open'] > 0:
                print(f"\tDepth {ddd}: [{progress['completed']} / {progress['open']} = {100 * progress['completed'] / progress['open']:.1f}%] [remaining={progress['open'] - progress['completed']}]")
            else:
                print(f"\tDepth {ddd}: [{progress['completed']} / {progress['open']}")

    except: 
        print("Could not print") 






print_when_possible = False
def print_status():
    global print_when_possible
    print_when_possible = True

on_press_key('π', print_status) # option-p



skip_to_next_branch = False
# def skip_branch():
#     global skip_to_next_branch
#     skip_to_next_branch = True
#     print('skipping branch!')


# on_press_key('ø', skip_branch) # option-o

    



def branching_search(basics, options, current_path=None, current_path_checkpoint_scores=None, current_start=0, reaction_start=0, continuations=None, greedy_branching=False, recursive=True, probe_to_time=False, peak_tolerance=None): 
    global best_finished_path
    global path_counts
    global skip_to_next_branch



    if (probe_to_time and current_start >= probe_to_time):
        continuations.append([current_path, current_path_checkpoint_scores, current_start, reaction_start])
        return [None]

    if peak_tolerance is None:
        peak_tolerance = options.get('peak_tolerance')


    print_profiling()

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

    # print(f"\t\tDepth={depth} Reaction Start={reaction_start / sr} Current_start={current_start / sr}")

    if depth not in path_counts:
        initialize_path_counts_for_depth(depth)

    if skip_to_next_branch:
        if depth > 1:
            prune_types['manual_branch_prune'] += 1
            print('manual_branch_prune')
            return [None]
        else:
            skip_to_next_branch = False
 

    check_if_print_info(basics, current_start, reaction_start, depth, path_counts, sr)

    if should_prune_path(basics, options, current_path, current_path_checkpoint_scores, best_finished_path, current_start, reaction_start, path_counts):
        
        return [None]
    
    try: 


        current_end = min(current_start + step, len(base_audio))
        chunk = base_audio[current_start:current_end]
        current_chunk_size = current_end - current_start


        #######
        # Finished with this path!
        matched_all_of_base = current_start >= len(base_audio) - 1
        reaction_audio_depleted = reaction_start >= len(reaction_audio) - 1 or current_start > len(base_audio) - int(.5 * sr)
        if matched_all_of_base or reaction_audio_depleted:
            # print('finished!')
            # Handle case where reaction video finishes before base video
            if reaction_audio_depleted:
                length_remaining = len(base_audio) - current_start - 1
                filler_segment = (reaction_start, reaction_start + length_remaining, current_start, current_start + length_remaining, True)
                append_or_extend_segment(current_path, filler_segment)

                # print(f"Reaction video finished before end of base video. Backfilling with base video ({length_remaining / sr}s).")
                # print(f"Backfilling {current_path} {current_start} {reaction_start}")

            return [current_path]
        ###############


        alignment_bounds = options.get('alignment_bounds')
        if alignment_bounds is not None:
            upper_bound = get_bound(alignment_bounds, current_start, len(reaction_audio))


        candidate_starts = find_next_segment_start_candidates(
                                basics=basics, 
                                open_chunk=reaction_audio[reaction_start:],                   
                                open_chunk_mfcc=reaction_audio_mfcc[:, max(0, round(reaction_start / hop_length)):], 
                                open_chunk_vol_diff=reaction_audio_vol_diff[max(0, round(reaction_start / hop_length)):], 
                                closed_chunk=chunk,
                                closed_chunk_mfcc= base_audio_mfcc[:, round(current_start / hop_length):round(current_end / hop_length) ],
                                closed_chunk_vol_diff=base_audio_vol_diff[round(current_start / hop_length):round(current_end / hop_length)],
                                current_chunk_size=current_chunk_size, 
                                peak_tolerance=peak_tolerance,
                                open_start=reaction_start, 
                                closed_start=current_start, 
                                distance=first_n_samples, 
                                prune_for_continuity=True,
                                prune_types=prune_types,
                                upper_bound=upper_bound, 
                                filter_for_similarity=True, #depth > 0, 
                                current_path=current_path
                            )

        #########
        # segment start pruned
        if candidate_starts == -1:
            # print("\t\t\treturning none")
            return [None]

        if candidate_starts:
            candidate_starts = [c for c in candidate_starts if c >= 0]


        # Could not find match in remainder of reaction video
        if candidate_starts is None or len(candidate_starts) == 0:
            increment = int(sr * .25)
            current_start += increment

            if (len(base_audio) - step - current_start) / sr < .5:
                # print(f"Could not find match in remainder of reaction video!!! Stopping.")
                # print(f"No match in remainder {current_path} {current_start} {reaction_start}")

                length_remaining = len(base_audio) - current_start - 1
                if length_remaining > 0:
                    filler_segment = (reaction_start, reaction_start + length_remaining, current_start, current_start + length_remaining, True)
                    append_or_extend_segment(current_path, filler_segment)

                return [current_path]
            elif ( current_start / len(base_audio) > .75 ):
                # print(f"Could not find match in remainder of reaction video!!! Skipping forward.")
                open_path(depth)

                filler_segment = (reaction_start - increment, reaction_start, current_start - increment, current_start, True)
                append_or_extend_segment(current_path, filler_segment)

                result = branching_search(basics, options, current_path, copy.deepcopy(current_path_checkpoint_scores), current_start, reaction_start, continuations=continuations, greedy_branching=greedy_branching, recursive=recursive, probe_to_time=probe_to_time, peak_tolerance=peak_tolerance)
                complete_path(depth)
                return result
            else: # if this occurs earlier in the reaction, it almost always results from a poor match
                return [None]
        #########

        # for candidate in candidate_starts:
        #     print(f"\t\t\tcandidate start={(reaction_start + candidate) / sr}")

        paths = []
        open_path(depth, len(candidate_starts))

        for ci, candidate_segment_start in enumerate(candidate_starts):
            if depth == 0:
                print(f"STARTING DEPTH ZERO from {candidate_segment_start / sr}")

            my_path = copy.deepcopy(current_path)

            scoping_needed = True
            next_start = current_start
            next_reaction_start = reaction_start

            while scoping_needed:
                segment, next_start, next_reaction_start = scope_segment(basics, options, next_start, next_reaction_start, candidate_segment_start, current_chunk_size, prune_types)
                if segment:
                    back_fill_needed = segment[-1]
                    append_or_extend_segment(my_path, segment)
                elif next_reaction_start < len(reaction_audio) - 1: 
                    back_fill_needed = True
                    print("did not find segment", next_start/sr, next_reaction_start/sr)
                    increment = int(sr * .25)
                    segment = (reaction_start - increment, reaction_start, current_start, current_start + increment, True)
                    append_or_extend_segment(my_path, segment)

                scoping_needed = back_fill_needed

            filler = segment[-1]
            if (not recursive and filler and len(my_path) < 10) or (recursive and (not greedy_branching or ci < greedy_branching) and (not probe_to_time or next_start < probe_to_time)):

                # print('recursing', filler, my_path)
                continued_paths = branching_search(basics, options,
                    current_path=my_path, 
                    current_path_checkpoint_scores=copy.deepcopy(current_path_checkpoint_scores), 
                    current_start=next_start, 
                    reaction_start=next_reaction_start, 
                    continuations=continuations, 
                    greedy_branching=greedy_branching, 
                    recursive=recursive, 
                    probe_to_time=probe_to_time,
                    peak_tolerance=peak_tolerance)

                for full_path in continued_paths:
                    if full_path is not None:
                        paths.append(full_path)

                complete_path(depth)
            else: 
                continuations.append([my_path, copy.deepcopy(current_path_checkpoint_scores), next_start, next_reaction_start])



        return paths

    except Exception as e: 
        print("Got error down this path")
        traceback.print_exc()
        print(e)
        return [None]



def check_if_print_info(basics, current_start, reaction_start, depth, path_counts, sr):
    global best_finished_path
    global print_when_possible

    if print_when_possible: 
        print_paths(current_start, reaction_start, depth, path_counts, sr)
        print_prune_data(basics)
        if 'score' in best_finished_path:
            print(f"**** Best score is {best_finished_path['score']}")
            print_path(best_finished_path["path"], basics)
        print_when_possible = False


def update_paths(basics, paths, new_paths):
    global best_finished_path
    for new_path in new_paths: 
        if new_path is not None:
            score = path_score(new_path, basics)
            if 'score' not in best_finished_path or best_finished_path['score'][0] < score[0]:
                old_best_path = best_finished_path.get('path', None)
                best_finished_path.update({
                    "path": new_path,
                    "score": score,
                    "partials": {}
                    })
                print(f"**** New best score is {best_finished_path['score']}")
                print_path(new_path, basics)

                if old_best_path:
                    print("Comparative checkpoint scores:")
                    checkpoints = basics.get('checkpoints')
                    sr = basics.get('sr')
                    for ts in checkpoints: 
                        _, old_best_score = calculate_partial_score(old_best_path, ts, basics)
                        _, new_best_score = calculate_partial_score(best_finished_path['path'], ts, basics)
                        print(f"\t{ts / sr}: {new_best_score[0] / old_best_score[0]}")
                else: 
                    print("...No old best score to compare")

            paths.append(new_path)

def get_score_ratio_at_checkpoint(candidate, best_candidate, scoring_function, checkpoint, basics):
    path, checkpoint_scores, current_start, reaction_start = candidate        
    best_path, best_checkpoint_scores, best_current_start, best_reaction_start = best_candidate
        
    for p,check_scores in [(path, checkpoint_scores), (best_path, best_checkpoint_scores)]:
        if checkpoint not in check_scores:
            __, score = calculate_partial_score(p, checkpoint, basics)
            check_scores[checkpoint] = score

    score = checkpoint_scores[checkpoint]
    best_score = best_checkpoint_scores[checkpoint]

    return scoring_function(score) / scoring_function(best_score)

def get_score_for_tight_bound(score):
    return score[2] * score[3]

def get_score_for_overall_comparison(score):
    return score[2]



def align_by_checkpoint_probe(basics, options):
    global path_counts
    global best_finished_path
    global print_when_possible

    checkpoints = basics.get('checkpoints')
    base_audio = basics.get('base_audio')
    song_length = len(base_audio)

    sr = basics.get('sr')

    checkpoint_threshold_tight = .85
    checkpoint_threshold_light = .45
    checkpoint_threshold_fill = .8
    checkpoint_threshold_location = .99

    low_score_queue = []
    paths = []

    print("Getting starting points")
    starting_points = []
    new_paths = branching_search(basics, options, current_path=[], current_path_checkpoint_scores={}, current_start=0, reaction_start=0, continuations=starting_points, recursive=False, peak_tolerance=options.get('peak_tolerance')*.75)
    update_paths(basics, paths, new_paths)

    checkpoints = [c for c in checkpoints]
    checkpoints.append(None)

    for idx,checkpoint_ts in enumerate(checkpoints): 
        past_the_checkpoint = []

        if checkpoint_ts is not None:
            print(f"Processing checkpoint {checkpoint_ts / sr} ({idx / len(checkpoints)*100}%)")
        else: 
            print(f"To the end! ({idx / len(checkpoints)})")


        starting_points.sort(key=lambda x: x[3])
        for idx2, starting_point in enumerate(starting_points):
            path, checkpoint_scores, current_start, reaction_start = starting_point
            print(f"\tStarting now at {reaction_start / sr} ({current_start / sr}) [{100 * idx2 / len(starting_points)}%]...{len(past_the_checkpoint)} continuations found so far")

            if (checkpoint_ts and current_start >= checkpoint_ts):
                past_the_checkpoint.append([path, checkpoint_scores, current_start, reaction_start])
            else: 
                continuations = []
                new_paths = branching_search(basics, options, current_path=path, current_path_checkpoint_scores=checkpoint_scores, current_start=current_start, reaction_start=reaction_start, continuations=continuations, recursive=True, probe_to_time=checkpoint_ts)
                update_paths(basics, paths, new_paths)
                past_the_checkpoint.extend(continuations)
                # if abs( reaction_start / sr - 623.5551473922902 ) < .5:
                #     print(f"\t\t Interest! {current_start / sr} {reaction_start / sr}") 
                #     for path, checkpoint_scores, current_start, reaction_start in continuations:
                #         print(f"\t\t\t {current_start / sr}  {reaction_start / sr}", path[-1])


        if checkpoint_ts is None:
            # should be finished!
            if (len(past_the_checkpoint) > 0):
                print("ERRRORRRRR!!!! We have paths remaining at the end.")

        else: 

            if (len(past_the_checkpoint) == 0):
                print("ERRRORRRRR!!!! We didn't find any paths!")

            percent_through = 100 * checkpoint_ts / len(base_audio)

            best_score_past_checkpoint = 0 
            best_path_past_checkpoint = None
            best_past_the_checkpoint = []
            best_candidate_past_the_checkpoint = None
            reaction_ends = {}
            all_by_current_location = {}


            all_by_score = []


            best_scores_to_point = {}

            for x, candidate in enumerate(past_the_checkpoint):
                path, checkpoint_scores, current_start, reaction_start = candidate

                if checkpoint_ts not in checkpoint_scores:
                    reaction_end, score_at_checkpoint = calculate_partial_score(path, checkpoint_ts, basics)
                    reaction_ends[x] = reaction_end
                    checkpoint_scores[checkpoint_ts] = score_at_checkpoint
                else: 
                    if x not in reaction_ends:
                        result = truncate_path(path, checkpoint_ts)
                        if result is None:
                            continue
                        reaction_end, _ = result
                        reaction_ends[x] = reaction_end
                    reaction_end = reaction_ends[x]

                if reaction_end not in best_scores_to_point:
                    for y, candidate2 in enumerate(past_the_checkpoint):
                        path2, checkpoint_scores2, current_start2, reaction_start2 = candidate2

                        if checkpoint_ts not in checkpoint_scores2:
                            reaction_end2, score_at_checkpoint2 = calculate_partial_score(path2, checkpoint_ts, basics)
                            reaction_ends[y] = reaction_end2
                            checkpoint_scores2[checkpoint_ts] = score_at_checkpoint2
                        else: 
                            if y not in reaction_ends:
                                result = truncate_path(path, checkpoint_ts)
                                if result is None:
                                    continue
                                reaction_end2, _ = result
                                reaction_ends[y] = reaction_end2
                            reaction_end2 = reaction_ends[y]                    
                            score_at_checkpoint2 = checkpoint_scores2[checkpoint_ts]


                        if reaction_end2 <= reaction_end and get_score_for_tight_bound(score_at_checkpoint2) > best_scores_to_point.get(reaction_end, [0])[0]:
                            best_scores_to_point[reaction_end] = [get_score_for_tight_bound(score_at_checkpoint2), path2, candidate2]

                        if get_score_for_overall_comparison(score_at_checkpoint2) > best_score_past_checkpoint:
                            best_score_past_checkpoint = get_score_for_overall_comparison(score_at_checkpoint2)
                            best_path_past_checkpoint = path2
                            best_candidate_past_the_checkpoint = candidate2

                best_score_to_point, best_path_to_point, best_candidate_to_point = best_scores_to_point[reaction_end]

                score_at_checkpoint = checkpoint_scores[checkpoint_ts]
                

                depth = len(path)
                complete_path(depth)



                passes_tight = checkpoint_threshold_tight < get_score_ratio_at_checkpoint(candidate, best_candidate_to_point, get_score_for_tight_bound, checkpoint_ts, basics) 
                passes_mfcc = checkpoint_threshold_light < get_score_ratio_at_checkpoint(candidate, best_candidate_past_the_checkpoint, get_score_for_overall_comparison, checkpoint_ts, basics)
                passes_fill = checkpoint_threshold_fill < score_at_checkpoint[3] or percent_through < 20


                if passes_tight and passes_mfcc and passes_fill:

                    best_past_the_checkpoint.append(candidate)
                    location = f"{int(1000 * current_start / sr)} {int(1000 * reaction_start / sr)}"
                    if location not in all_by_current_location:
                        all_by_current_location[location] = []
                    all_by_current_location[location].append( [candidate, score_at_checkpoint]  )

                ideal_filter = get_score_for_tight_bound(score_at_checkpoint) / best_score_to_point > .75 and .4 < score_at_checkpoint[2] / best_score_past_checkpoint and (.8 < score_at_checkpoint[3] or percent_through < 20)
                all_by_score.append([get_score_for_overall_comparison(score_at_checkpoint), score_at_checkpoint[3], ideal_filter, path])



            filtered_best_past_the_checkpoint = []

            for location, candidates in all_by_current_location.items():
                if len(candidates) < 2:
                    filtered_best_past_the_checkpoint.append(candidates[0][0])
                else:
                    kept = 0
                    best_at_location = 0
                    for candidate, score in candidates:
                        if score[0] > best_at_location:
                            best_at_location = score[0]

                    for candidate, score in candidates:
                        if score[0] / best_at_location > checkpoint_threshold_location:
                            filtered_best_past_the_checkpoint.append(candidate)
                            kept += 1
                    # print(f"Location filter: removed {len(candidates) - kept} of {len(candidates)}    kept: {kept / len(candidates)}")



            best_past_the_checkpoint = filtered_best_past_the_checkpoint


            if len(past_the_checkpoint) > 0:
                print(f"\tFinished checkpoint {checkpoint_ts}. Kept {len(best_past_the_checkpoint)} of {len(past_the_checkpoint)} ({100 * len(best_past_the_checkpoint) / len(past_the_checkpoint)}%)")
                print(f"best path at checkpoint\t", print_path(best_path_past_checkpoint, basics))
            if False and len(past_the_checkpoint) > 1000:
                if len(past_the_checkpoint) > 1:
                    plot_candidates(all_by_score)

            starting_points = best_past_the_checkpoint



    if 'score' in best_finished_path:
        print(f"**** Best score is {best_finished_path['score']}")
        print_path(best_finished_path["path"], basics)

    print_prune_data(basics)


    return [p for p in paths if p]



def plot_candidates(data):
    # Separate the data into different lists for easier plotting
    scores = [item[0] for item in data]
    depths = [item[1] for item in data]
    selected = [item[2] for item in data]

    # Create the plot
    plt.figure()

    # Loop through the data to plot points color-coded by 'selected'
    for i in range(len(scores)):
        if selected[i]:
            plt.scatter(depths[i], scores[i], color='green')
        else:
            plt.scatter(depths[i], scores[i], color='red')

    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Add labels and title
    plt.xlabel('FILL')
    plt.ylabel('MFCC')
    plt.title('MFCC vs FILL')

    # Show the plot
    plt.show()


use_align_by_greed = False
def find_alignments(basics, options):
    global path_counts
    global best_finished_path
    global print_when_possible

    initialize_segment_start_cache()
    initialize_segment_end_cache()
    initialize_path_score()
    initialize_path_counts()
    initialize_path_pruning()
    initialize_segment_tracking()


    best_finished_path.clear()


    saved_bounds = os.path.splitext(options['output_file'])[0] + '-bounds.pckl'
    if not os.path.exists(saved_bounds):
        options['alignment_bounds'] = create_reaction_alignment_bounds(basics, options['first_n_samples'])
        save_object_to_file(saved_bounds, options['alignment_bounds'])
    else: 
        options['alignment_bounds'] = read_object_from_file(saved_bounds)


    if use_align_by_greed:
        paths = align_by_greed(basics, options)
    else: 
        paths = align_by_checkpoint_probe(basics, options)

    return paths





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

    if ground_truth: 
        ground_truth = [ (int(s * sr), int(e * sr)) for (s,e) in ground_truth]

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


    paths = find_alignments(basics, options)

    path = find_best_path(paths, basics)

    sequences = compress_segments(basics, path)

    reaction_sample_rate = Decimal(sr)
    final_sequences =          [ ( Decimal(s[0]) / reaction_sample_rate, Decimal(s[1]) / reaction_sample_rate, Decimal(s[2]) / reaction_sample_rate, Decimal(s[3]) / reaction_sample_rate, s[4]) for s in sequences ]
    
    if ground_truth:
        compute_precision_recall(sequences, ground_truth, tolerance=1.5)

    return final_sequences



def create_aligned_reaction_video(song:dict, react_video_ext, output_file: str, react_video, base_video, base_audio_data, base_audio_path, options, extend_by = 0):

    gt = song.get('ground_truth', {}).get(os.path.splitext(os.path.basename(react_video))[0] )
    # if not gt: 
    #     return

    options.setdefault("step_size", 1)
    options.setdefault("min_segment_length_in_seconds", 3)
    options.setdefault("reverse_search_bound", options['min_segment_length_in_seconds'])
    options.setdefault("peak_tolerance", .5)
    options.setdefault("expansion_tolerance", .85)
    options['output_file'] = output_file



    # Extract the reaction audio
    reaction_audio_data, reaction_sample_rate, reaction_audio_path = extract_audio(react_video)

    if options['output_alignment_metadata']:
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

            save_object_to_file(alignment_metadata_file, final_sequences)
        else: 
            final_sequences = read_object_from_file(alignment_metadata_file)

    if not os.path.exists(output_file) and options["output_alignment_video"]:
        # Trim and align the reaction video
        trim_and_concat_video(react_video, final_sequences, base_video, output_file, react_video_ext, extend_by = extend_by, use_fill = song.get('include_base_video', True))
    return output_file




def compress_segments(basics, match_segments):
    compressed_subsequences = []

    sr = basics.get('sr')
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
            if start - current_end <= 1:
                # This subsequence is continuous with the current one, extend it
                # print("***COMBINING SEGMENT", current_end, start, (start - current_end), (start - current_end) / sr   )
                current_end = end
                current_base_end = base_end
            else:
                # This subsequence is not continuous, add the current one to the list and start a new one
                compressed_segment = (current_start, current_end, current_base_start, current_base_end, filler)
                # if not filler:
                #     print('not contiguous', current_end, start, current_base_end, base_start)
                #     # assert( is_close(current_end - current_start, current_base_end - current_base_start) )
                compressed_subsequences.append( compressed_segment )
                current_start, current_end, current_base_start, current_base_end, _ = start, end, base_start, base_end, filler

        # Add the last subsequence
        compressed_subsequences.append((current_start, current_end, current_base_start, current_base_end, filler))
        # print(f"{end - start} vs {base_end - base_start}"      )
        # if not filler:
        #     if not is_close(current_end - current_start, current_base_end - current_base_start):
        #         print("NOT CLOSE!!!! Possible error", current_start, current_end - current_start, current_base_end - current_base_start)

    # compressed_subsequences = match_segments
    return compressed_subsequences

