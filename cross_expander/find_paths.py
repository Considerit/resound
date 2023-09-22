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
import random
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from prettytable import PrettyTable


from cross_expander.create_trimmed_video import trim_and_concat_video

from utilities import compute_precision_recall, universal_frame_rate, is_close, print_profiling
from utilities import conf, save_object_to_file, read_object_from_file
from utilities import conversion_audio_sample_rate as sr

from cross_expander.pruning_search import initialize_path_pruning, initialize_checkpoints, print_prune_data

from cross_expander.find_segment_start import initialize_segment_start_cache
from cross_expander.find_segment_end import initialize_segment_end_cache
from cross_expander.scoring_and_similarity import path_score, find_best_path, initialize_path_score, print_path, calculate_partial_score
from cross_expander.scoring_and_similarity import initialize_segment_tracking, truncate_path
from cross_expander.bounds import create_reaction_alignment_bounds
from cross_expander.branching_search import branching_search


from utilities.audio_processing import audio_percentile_loudness








best_finished_path = {}



def update_paths(reaction, paths, new_paths):
    global best_finished_path
    for new_path in new_paths: 
        if new_path is not None:
            score = path_score(new_path, reaction)
            if 'score' not in best_finished_path or best_finished_path['score'][0] < score[0]:
                # old_best_path = best_finished_path.get('path', None)
                best_finished_path.update({
                    "path": new_path,
                    "score": score,
                    "partials": {}
                    })
                print(f"**** New best score is {best_finished_path['score']}")
                print_path(new_path, reaction)

            paths.append(new_path)

def get_score_ratio_at_checkpoint(reaction, score, best_score, scoring_function, checkpoint):
    if scoring_function(best_score) == 0:
        return 1

    return scoring_function(score) / scoring_function(best_score)


# Threshold have to increase as we get farther into the video, because pathways often share a big portion of 
# their pathways. Suboptimum pathway choices 80% of the way through aren't pruned because they're benefiting 
# from 80% good score from the shared optimal pathway. In this method, we look at the differential in score
# from when two path's scores last diverged.

def get_score_ratio_at_checkpoint_diff(checkpoints_reversed, reaction, candidate, best_candidate, scoring_function, checkpoint):
    path, checkpoint_scores, current_start, reaction_start = candidate        
    best_path, best_checkpoint_scores, best_current_start, best_reaction_start = best_candidate

    past_checkpoint_to_compare = None
    for past_checkpoint in checkpoints_reversed:
        if (checkpoint - past_checkpoint) > 6 * sr and abs(1 - scoring_function(checkpoint_scores[past_checkpoint]) / scoring_function(best_checkpoint_scores[past_checkpoint])) < .01:
            past_checkpoint_to_compare = past_checkpoint
            break

    if past_checkpoint_to_compare is None:
        # didn't find an equal score, so we can't use this method
        return 1

    score = scoring_function(checkpoint_scores[checkpoint])
    best_score = scoring_function(best_checkpoint_scores[checkpoint])

    previous_score = scoring_function(checkpoint_scores[past_checkpoint_to_compare])
    previous_best_score = scoring_function(best_checkpoint_scores[past_checkpoint_to_compare])

    if best_score == 0:
        return 1

    diff = score - previous_score
    best_diff = best_score - previous_best_score


    if diff >= best_diff:
        return 1

    if abs(best_diff) < 1:
        diff += 1
        best_diff += 1

    if best_diff < 0 and diff < 0:
        return best_diff / diff
    elif best_diff > 0 and diff > 0:
        return diff / best_diff
    elif diff < 0:
        return abs(diff - best_diff) /  (-2 * diff + best_diff)

    else: # best_diff < 0, can't happen
        assert(False)

    if best_diff < 0:

        return best_diff / diff
    else:
        return diff / best_diff




def get_score_for_tight_bound(score):
    return score[2] * score[3]

def get_score_for_overall_comparison(score):
    return score[2]

def get_score_for_location(score):
    return score[0]

def check_path_quality(path):

    # Poor quality: 
    #  - reaction separation greater than .1s and less than 1s and no filler
    #  - three+ non filler segments less than 2.5 seconds with spacing between
    #  - first segment is less than 3s, followed by 60s+ of space...
    #      ...or first two segments are separated by .1s and average < 6s
    short_segments = 0
    short_separation = 0

    for i, (reaction_start, reaction_end, current_start, current_end, filler) in enumerate(path): 
        if filler:
            continue

        if i > 0: 
            space_before = reaction_start - path[i-1][1]
            filler_before = path[i-1][4]
        else:
            filler_before = False
            space_before = 0

        if i < len(path) - 1:
            space_after = path[i+1][0] - reaction_end
            filler_after = path[i+1][4]
        else:
            filler_after = False
            space_after = 0 

        if i == 0 and len(path) > 1 and not filler_after and space_after > 60 * sr and (reaction_end - reaction_start) < 3 * sr:
            return False

        if i == 0 and len(path) > 2 and not filler_after and space_after > .1 * sr and path[i+2][0] - path[i+1][1] > .1 * sr and ((reaction_end - reaction_start) + (path[1][1] - path[1][0])) < 6 * sr:
            return False

        if (not filler_before and .1 * sr < space_before < sr) or (not filler_after and .1 * sr < space_after < sr):
            return False

        if (reaction_end - reaction_start) < 2.5 * sr:

            # we're not going to consider the last segment, because we might add to it the next time it branches
            if (i == 0 or space_before > sr / 10 ) and i < len(path) - 1 and space_after > sr / 10:
                short_segments += 1
                if short_segments >= 3:
                    return False

    return True







def align_by_checkpoint_probe(reaction):
    global best_finished_path
    global print_when_possible

    checkpoints = conf.get('checkpoints')
    base_audio = conf.get('base_audio_data')
    song_length = len(base_audio)


    low_score_queue = []
    paths = []


    print_profiling()

    print("Getting starting points")
    starting_points = []
    new_paths = branching_search(reaction, 
                                 current_path=[], 
                                 current_path_checkpoint_scores={}, 
                                 current_start=0, 
                                 reaction_start=0, 
                                 continuations=starting_points, 
                                 recursive=False, 
                                 peak_tolerance=conf.get('peak_tolerance')*.25)
                                            # if changing the initial peak tolerance, check with suicide/that'snotacting

    update_paths(reaction, paths, new_paths)

    checkpoints = [c for c in checkpoints]
    checkpoints.append(None)


    prunes_all = {
        'tight': 0,
        'light': 0,
        'location': 0,
        'diff_tight': 0,
        'diff_light': 0,
        'location_diff': 0,
        'quality': 0
    }

    for idx,checkpoint_ts in enumerate(checkpoints): 
        past_the_checkpoint = []

        if checkpoint_ts is not None:
            print(f"\nProcessing checkpoint {round(checkpoint_ts / sr)}s ({idx / len(checkpoints)*100:.1f}%) of {reaction['channel']} / {conf.get('song_key')}")
        else: 
            print(f"To the end! ({idx / len(checkpoints)})")


        starting_points.sort(key=lambda x: x[3])
        for idx2, starting_point in enumerate(starting_points):
            path, checkpoint_scores, current_start, reaction_start = starting_point

            print(f"\t[{idx / len(checkpoints)*100:.1f}% / {100 * idx2 / len(starting_points):.1f}%] Reaction: {reaction_start / sr:.8f} Base: {current_start / sr:.4f} ... {len(past_the_checkpoint)} continuations", end="\r")

            if (checkpoint_ts and current_start >= checkpoint_ts):
                past_the_checkpoint.append([path, checkpoint_scores, current_start, reaction_start])

            else: 
                continuations = []
                new_paths = branching_search(reaction, current_path=path, current_path_checkpoint_scores=checkpoint_scores, current_start=current_start, reaction_start=reaction_start, continuations=continuations, recursive=True, probe_to_time=checkpoint_ts)
                update_paths(reaction, paths, new_paths)
                past_the_checkpoint.extend(continuations)
                # if abs( reaction_start / sr - 623.5551473922902 ) < .5:
                #     print(f"\t\t Interest! {current_start / sr} {reaction_start / sr}") 
                #     for path, checkpoint_scores, current_start, reaction_start in continuations:
                #         print(f"\t\t\t {current_start / sr}  {reaction_start / sr}", path[-1])

        print("")

        plotting = False and len(past_the_checkpoint) > 1000

        if checkpoint_ts is None:
            # should be finished!
            if (len(past_the_checkpoint) > 0):
                print("ERRRORRRRR!!!! We have paths remaining at the end.")

        else: 

            if (len(past_the_checkpoint) == 0):
                print("ERRRORRRRR!!!! We didn't find any paths!")

            percent_through = 100 * checkpoint_ts / len(base_audio)

            prunes = {}
            for k,v in prunes_all.items():
                prunes[k] = 0 


            checkpoint_threshold_tight = .9  # Black Pegasus / Suicide can't go to .8 when light threshold = .5 (ok, that's only true w/o the path quality filters)
            checkpoint_threshold_light = .5    # > .15 doesn't work for Black Pegasus / Suicide in conjunction with high tight threshold (>.9)
            checkpoint_threshold_location = .95  
            location_rounding = 10


            best_score_past_checkpoint = None
            best_value_past_checkpoint = 0 
            best_path_past_checkpoint = None
            best_candidate_past_the_checkpoint = None


            best_score_to_point = None
            best_value_to_point = 0 
            best_path_to_point = None
            best_candidate_to_point = None                 

            all_by_current_location = {}
            all_by_score = []


            checkpoints_reversed = [c for c in checkpoints if c and c <= checkpoint_ts][::-1]

            # Score candidates and get the reaction_end at the checkpoint
            candidates = []
            for x, candidate in enumerate(past_the_checkpoint):
                path = candidate[0]


                passes_path_quality = check_path_quality(path)

                if passes_path_quality: 
                    score_at_each_checkpoint = candidate[1]
                    reaction_end, score_at_checkpoint = calculate_partial_score(path, checkpoint_ts, reaction)
                    candidates.append([candidate, reaction_end, score_at_checkpoint])
                    score_at_each_checkpoint[checkpoint_ts] = score_at_checkpoint

                    value = get_score_for_overall_comparison(score_at_checkpoint)

                    if value > best_value_past_checkpoint: 
                        best_value_past_checkpoint = value
                        best_score_past_checkpoint = score_at_checkpoint
                        best_path_past_checkpoint = path
                        best_candidate_past_the_checkpoint = candidate
                else: 
                    prunes['quality'] += 1

            candidates.sort( key=lambda x: (x[1], -x[2][0]) )  # sort by reaction_end, with a tie breaker of -overall score. 
                                                               # The negative is so that the best score comes first, for 
                                                               # pruning purposes.

            for expanded_candidate in candidates:
                candidate, reaction_end, score_at_checkpoint = expanded_candidate
                path, score_at_each_checkpoint, current_start, reaction_start = candidate

                value = get_score_for_tight_bound(score_at_checkpoint)
                if  value > best_value_to_point:
                    best_value_to_point = value
                    best_score_to_point = score_at_checkpoint
                    best_path_to_point = path
                    best_candidate_to_point = candidate                


                # Don't prune out paths until we're at the checkpoint just before current_end of the latest path. 
                # These are very low cost options to keep, and it gives them a chance to shine and not prematurely pruned.
                passes_time = idx < len(checkpoints) - 1 and checkpoints[idx + 1] and path[-1][3] and path[-1][3] > checkpoints[idx + 1]

                passes_tight = passes_time or checkpoint_threshold_tight <= get_score_ratio_at_checkpoint(reaction, score_at_checkpoint, best_score_to_point, get_score_for_tight_bound, checkpoint_ts) 
                passes_mfcc = passes_time or checkpoint_threshold_light <= get_score_ratio_at_checkpoint(reaction, score_at_checkpoint, best_score_past_checkpoint, get_score_for_overall_comparison, checkpoint_ts)

                if not passes_tight: 
                    prunes['tight'] += 1
                if not passes_mfcc:
                    prunes['light'] += 1

                if passes_tight and passes_mfcc:

                    passes_tight_diff = passes_time or checkpoint_threshold_tight <= get_score_ratio_at_checkpoint_diff(checkpoints_reversed, reaction, candidate, best_candidate_to_point, get_score_for_tight_bound, checkpoint_ts) 
                    passes_mfcc_diff = passes_time or checkpoint_threshold_light <= get_score_ratio_at_checkpoint_diff(checkpoints_reversed, reaction, candidate, best_candidate_past_the_checkpoint, get_score_for_overall_comparison, checkpoint_ts)

                    if not passes_tight_diff: 
                        prunes['diff_tight'] += 1
                    if not passes_mfcc_diff:
                        prunes['diff_light'] += 1

                    if passes_tight_diff and passes_mfcc_diff:
                        location = f"{int(location_rounding * current_start / sr)} {int(location_rounding * reaction_start / sr)}"
                        if location not in all_by_current_location:
                            all_by_current_location[location] = []
                        all_by_current_location[location].append( expanded_candidate  )

                if plotting:
                    ideal_filter = passes_time and passes_tight and passes_mfcc and passes_tight_diff and passes_mfcc_diff
                    all_by_score.append([get_score_for_overall_comparison(score_at_checkpoint), score_at_checkpoint[3], ideal_filter, path])



            best_past_the_checkpoint = []

            for location, candidates in all_by_current_location.items():
                if len(candidates) < 2:
                    best_past_the_checkpoint.append(candidates[0][0])
                else:
                    kept = 0
                    best_at_location = 0
                    best_score_at_location = None
                    best_candidate_at_location = None

                    for candidate, reaction_end, score in candidates:
                        if get_score_for_location(score) > best_at_location:
                            best_at_location = get_score_for_location(score)
                            best_candidate_at_location = candidate
                            best_score_at_location = score

                    for candidate, reaction_end, score in candidates:
                        relative_current_end = candidate[0][-1][3]
                        passes_time = idx < len(checkpoints) - 1 and checkpoints[idx + 1] and relative_current_end and relative_current_end > checkpoints[idx + 1]
                        passes_location = get_score_for_location(score) / best_at_location >= checkpoint_threshold_location
                        if passes_time or passes_location:
                            passes_location_diff = checkpoint_threshold_location <= get_score_ratio_at_checkpoint_diff(checkpoints_reversed, reaction, candidate, best_candidate_at_location, get_score_for_location, checkpoint_ts) 
                            
                            if passes_location_diff:
                                best_past_the_checkpoint.append(candidate)
                            else: 
                                prunes['location_diff'] += 1
                        else: 
                            prunes['location'] += 1


            if len(past_the_checkpoint) > 0:
                print(f"\tKept {len(best_past_the_checkpoint)} of {len(past_the_checkpoint)} ({100 * len(best_past_the_checkpoint) / len(past_the_checkpoint):.1f}%)")
                print(f"\tBest path at checkpoint:")
                print_path(best_path_past_checkpoint, reaction)



                # if idx > 0 and len(best_past_the_checkpoint) > 1000:
                #     print("RANDOM")
                #     for p in random.sample(best_past_the_checkpoint, 25):
                #         print_path(p[0], reaction)


                print("")

                x = PrettyTable()
                x.border = False
                x.align = "r"
                x.field_names = ['\t', 'Prune type', 'Checkpoint', 'Overall']
                for k,v in prunes.items():
                    prunes_all[k] += v   
                    x.add_row(['\t', k, v, prunes_all[k]])  

                    # print(f"\tPrune {k}: {v} [{prunes_all[k]}]")
                print(x)






            if plotting:
                if len(past_the_checkpoint) > 1:
                    plot_candidates(all_by_score)

            starting_points = best_past_the_checkpoint



    if 'score' in best_finished_path:
        print(f"**** Best score is {best_finished_path['score']}")
        print_path(best_finished_path["path"], reaction)

    print_prune_data()


    return [p for p in paths if p]



def plot_candidates(data):
    
    # Separate the data into different lists for easier plotting
    scores = [item[0] for item in data]
    # depths = [item[1] for item in data]
    depths = [item[3][0][0] / sr for item in data]   # plotting scores of paths that start from a particular reaction_start 
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

def find_alignments(reaction):
    global best_finished_path
    global print_when_possible

    initialize_segment_start_cache()
    initialize_segment_end_cache()
    initialize_path_score()
    initialize_path_pruning()
    initialize_segment_tracking()


    best_finished_path.clear()


    saved_bounds = os.path.splitext(reaction.get('aligned_path'))[0] + '-bounds.pckl'
    if not os.path.exists(saved_bounds):
        conf['alignment_bounds'] = create_reaction_alignment_bounds(reaction, conf['first_n_samples'])
        save_object_to_file(saved_bounds, conf['alignment_bounds'])
    else: 
        conf['alignment_bounds'] = read_object_from_file(saved_bounds)


    paths = align_by_checkpoint_probe(reaction)

    return paths





def cross_expander_aligner(reaction):
    global conf

    step_size = conf.get('step_size')
    min_segment_length_in_seconds = conf.get('min_segment_length_in_seconds')

    # Convert seconds to samples
    n_samples = int(step_size * sr)
    first_n_samples = int(min_segment_length_in_seconds * sr)


    conf['checkpoints'] = initialize_checkpoints()
    conf['n_samples'] = n_samples
    conf['first_n_samples'] = first_n_samples

    paths = find_alignments(reaction)

    path = find_best_path(reaction, paths)

    sequences = compress_segments(path)

    ground_truth = reaction.get('ground_truth')
    if ground_truth:
        compute_precision_recall(sequences, ground_truth, tolerance=1.5)

    return sequences



def create_aligned_reaction_video(reaction, extend_by = 0):
    global conf

    output_file = reaction.get('aligned_path')


    conf.setdefault("step_size", 1)
    conf.setdefault("min_segment_length_in_seconds", 3)
    conf.setdefault("reverse_search_bound", conf['min_segment_length_in_seconds'])
    conf.setdefault("peak_tolerance", .5)
    conf.setdefault("expansion_tolerance", .85)

    if conf['create_alignment'] or conf['alignment_test']:
        alignment_metadata_file = os.path.splitext(output_file)[0] + '.pckl'
        score_metadata_file = os.path.splitext(output_file)[0] + '-score.pckl'

        if not os.path.exists(alignment_metadata_file):
            conf['load_reaction'](reaction['channel'])
            

            # Determine the number of decimal places to try avoiding frame boundary errors given python rounding issues
            fr = Decimal(universal_frame_rate())
            precision = Decimal(1) / fr
            precision_str = str(precision)
            getcontext().prec = len(precision_str.split('.')[-1])

            best_path = cross_expander_aligner(reaction)
            

            # print("\nsequences:")

            # for segment in best_path:
            #     print(f"\t{'*' if segment[4] else ''}base: {float(segment[2])}-{float(segment[3])}  reaction: {float(segment[0])}-{float(segment[1])}")

            if conf['save_alignment_metadata']:
                save_object_to_file(alignment_metadata_file, best_path)
                save_object_to_file(score_metadata_file, path_score(best_path, reaction))
        else: 
            best_path = read_object_from_file(alignment_metadata_file)


        reaction['best_path'] = best_path

        if not os.path.exists(score_metadata_file):
            print("SAVING SCORE", score_metadata_file)
            conf['load_reaction'](reaction['channel'])
            save_object_to_file(score_metadata_file, path_score(best_path, reaction))
        
        reaction['best_path_score'] = read_object_from_file(score_metadata_file)

        if not os.path.exists(output_file) and conf["output_alignment_video"]:
            react_video = reaction.get('video_path')
            base_video = conf.get('base_video_path')
            conf['load_reaction'](reaction['channel'])

            reaction_sample_rate = Decimal(sr)
            best_path_converted = [ ( Decimal(s[0]) / reaction_sample_rate, Decimal(s[1]) / reaction_sample_rate, Decimal(s[2]) / reaction_sample_rate, Decimal(s[3]) / reaction_sample_rate, s[4]) for s in reaction['best_path'] ]

            trim_and_concat_video(react_video, best_path_converted, base_video, output_file, extend_by = extend_by, use_fill = conf.get('include_base_video', True))
        

    return output_file




def compress_segments(match_segments):
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

