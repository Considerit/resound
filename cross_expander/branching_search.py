import traceback
import copy
import os

from utilities import conversion_audio_sample_rate as sr
from utilities import conf, print_profiling

from cross_expander.find_segment_start import find_segment_starts
from cross_expander.pruning_search import should_prune_path, prune_types
from cross_expander.find_segment_end import find_segment_end, check_for_start_adjustment
from cross_expander.bounds import get_bound

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

def branching_search(reaction, current_path=None, current_path_checkpoint_scores=None, current_start=0, reaction_start=0, continuations=None, greedy_branching=False, recursive=True, probe_to_time=False, peak_tolerance=None): 
    global best_finished_path

    if (probe_to_time and current_start >= probe_to_time):
        continuations.append([current_path, current_path_checkpoint_scores, current_start, reaction_start])
        return [None]

    if should_prune_path(reaction, current_path, current_start, reaction_start):
        # print('\t\tPRUNED')
        return [None]

    if peak_tolerance is None:
        peak_tolerance = conf.get('peak_tolerance')

    print_profiling()

    base_audio = conf.get('base_audio_data')
    reaction_audio = reaction.get('reaction_audio_data')

    
    base_audio_mfcc = conf.get('base_audio_mfcc')
    base_audio_vol_diff = conf.get('song_percentile_loudness')
    reaction_audio_mfcc = reaction.get('reaction_audio_mfcc')
    reaction_audio_vol_diff = reaction.get('reaction_percentile_loudness')
    hop_length = conf.get('hop_length')

    depth = len(current_path)

    


    # print(f"\t\tDepth={depth} Reaction Start={reaction_start / sr} Current_start={current_start / sr}")

    
    try: 

        step = first_n_samples = conf.get('first_n_samples')
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


        alignment_bounds = reaction['alignment_bounds']
        if alignment_bounds is not None:
            upper_bound = get_bound(alignment_bounds, current_start, len(reaction_audio))

        candidate_starts = find_segment_starts(
                                reaction=reaction, 
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
                                distance= sr, # first_n_samples, 
                                prune_for_continuity=True,
                                full_search=True,
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
                filler_segment = (reaction_start - increment, reaction_start, current_start - increment, current_start, True)
                append_or_extend_segment(current_path, filler_segment)

                result = branching_search(reaction, current_path, copy.copy(current_path_checkpoint_scores), current_start, reaction_start, continuations=continuations, greedy_branching=greedy_branching, recursive=recursive, probe_to_time=probe_to_time, peak_tolerance=peak_tolerance)

                return result
            else: # if this occurs earlier in the reaction, it almost always results from a poor match
                return [None]
        #########

        if depth > 0 and 0 not in candidate_starts:
            candidate_starts.append(0)  # always consider the continuation


        # Create two unique paths if there is a candidate starting point adjustment.
        starting_points = []
        for candidate_segment_start in candidate_starts:
            # print(f"\t\t\tcandidate start={(reaction_start + candidate_segment_start) / sr}")
            starting_points.append( (candidate_segment_start, current_start, copy.deepcopy(current_path))  )

            adjusted_start = check_for_start_adjustment(reaction, current_start, reaction_start, candidate_segment_start, current_chunk_size)
            if adjusted_start is not None and adjusted_start > 0: #adjusted_start > sr / 100:

                my_adjusted_path = copy.deepcopy(current_path)

                filler_segment = [reaction_start - adjusted_start, reaction_start, current_start, current_start + adjusted_start, True]
                append_or_extend_segment(my_adjusted_path, filler_segment)
                starting_points.append( (candidate_segment_start + adjusted_start, current_start + adjusted_start, my_adjusted_path)    )



        paths = []

        for ci, (candidate_segment_start, next_start, my_path) in enumerate(starting_points):

            if depth == 0:
                print(f"STARTING DEPTH ZERO from {candidate_segment_start / sr}")

            find_end = True
            next_reaction_start = reaction_start

            while find_end:
                segment, next_start, next_reaction_start = find_segment_end(reaction, next_start, next_reaction_start, candidate_segment_start, current_chunk_size, prune_types)
                if segment:
                    backfill_was_needed = segment[-1]
                    append_or_extend_segment(my_path, segment)
                elif next_reaction_start < len(reaction_audio) - 1: 
                    backfill_was_needed = True
                    print("did not find segment", next_start/sr, next_reaction_start/sr)
                    increment = int(sr * .25)
                    segment = (reaction_start - increment, reaction_start, current_start, current_start + increment, True)
                    append_or_extend_segment(my_path, segment)

                find_end = backfill_was_needed

            filler = segment[-1]
            if (not recursive and filler and len(my_path) < 10) or (recursive and (not greedy_branching or ci < greedy_branching) and (not probe_to_time or next_start < probe_to_time)):

                # print('recursing', filler, my_path)
                continued_paths = branching_search(reaction,
                    current_path=my_path, 
                    current_path_checkpoint_scores=copy.copy(current_path_checkpoint_scores),
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

            else: 
                continuations.append([my_path, copy.copy(current_path_checkpoint_scores), next_start, next_reaction_start])



        return paths

    except Exception as e: 
        print("Got error down this path")
        traceback.print_exc()
        print(e)
        return [None]


def append_or_extend_segment(my_path, segment):
    if len(my_path) == 0:
        my_path.append(segment)
        return
    (previous_reaction_start, previous_reaction_end, previous_start, previous_end, previous_is_filler) = my_path[-1]
    (new_reaction_start, new_reaction_end, new_start, new_end, new_is_filler) = segment
    if new_reaction_start - previous_reaction_end > 1 or new_is_filler or new_is_filler != previous_is_filler:
        my_path.append(segment)
    else: 
        my_path[-1][1] = new_reaction_end
        my_path[-1][3] = new_end 

