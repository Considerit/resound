import math, random
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import correlate, find_peaks
from cross_expander.scoring_and_similarity import mfcc_similarity, relative_volume_similarity
from cross_expander.pruning_search import check_for_prune_at_segment_start


from utilities import on_press_key



def correct_peak_index(peak_index, chunk_len):
    # return max(0,peak_index)
    return max(0, peak_index - (chunk_len - 1))

seg_start_cache = {}
seg_start_cache_effectiveness = {}
paths_at_segment_start = {}
def initialize_segment_start_cache():
    global seg_start_cache
    global seg_start_cache_effectiveness


    seg_start_cache.clear()
    seg_start_cache_effectiveness["hits"] = 0
    seg_start_cache_effectiveness["misses"] = 0

    paths_at_segment_start.clear()



show_plots = False
def toggle_plots():
    global show_plots
    show_plots = not show_plots

on_press_key('˚', toggle_plots) # option-k


# For debugging:
# If ground truth is defined, we try to filter down to only the paths
# that include the ground truth. It isn't perfect, but can really help
# to figure out if the system at least generates the ground truth path.
# Even if it doesn't select it.
force_ground_truth = False

def find_next_segment_start_candidates(basics, open_chunk, open_chunk_mfcc, open_chunk_vol_diff, closed_chunk, closed_chunk_mfcc, closed_chunk_vol_diff, current_chunk_size, peak_tolerance, open_start, closed_start, distance, prune_for_continuity=False, prune_types=None, upper_bound=None, filter_for_similarity=True, current_path=None, use_mfcc_correlation=False):
    global seg_start_cache
    global seg_start_cache_effectiveness
    global paths_at_segment_start


    sr = basics.get('sr')
    hop_length = basics.get('hop_length')


    if force_ground_truth: 
        gt = basics.get('ground_truth')
        if gt and use_mfcc_correlation: 
            gt_current_start = 0
            alright = False
            for idx, (start,end) in enumerate(gt):
                if closed_start >= gt_current_start - (idx + 1) * sr and closed_start <= gt_current_start + (end-start) + (idx + 1) * sr:
                    if open_start < end:
                        # print("ALRIGHT!", open_start / sr, end / sr, closed_start / sr, (gt_current_start / sr - (idx + 1)), (gt_current_start + (end-start))/sr + (idx + 1))
                        alright = True
                gt_current_start += end - start
            if not alright:
                # print('petty prune')
                return -1 


    key = f"{open_start} {closed_start} {len(open_chunk)} {len(closed_chunk)} {upper_bound} {peak_tolerance} {filter_for_similarity} {prune_for_continuity} {hop_length}"
    
    if current_path is not None:
        if key not in paths_at_segment_start:
            paths_at_segment_start[key] = [[list(current_path), None]]
        else: 
            if check_for_prune_at_segment_start(basics, paths_at_segment_start[key], current_path, closed_start):
                seg_start_cache_effectiveness["hits"] += 1        
                return -1


    if key not in seg_start_cache:

        # print(key)
        if upper_bound is not None:
            prev = len(open_chunk)
            open_chunk = open_chunk[:int(upper_bound - open_start + 2 * current_chunk_size)]
            open_chunk_mfcc = open_chunk_mfcc[:, :int((upper_bound - open_start + 2 * current_chunk_size) / hop_length)]

            # print(f"\tConstraining open chunk from size {prev} to {len(open_chunk)}  [{len(open_chunk) / prev * 100}% of original]")

        assert len(open_chunk) > 0
        assert len(closed_chunk) > 0

        correlation = correlate(open_chunk, closed_chunk)
        cross_max = np.max(correlation)
        # Find peaks
        peak_indices, _ = find_peaks(correlation, height=cross_max*peak_tolerance, distance=distance)
        peak_indices = peak_indices.tolist()

        if use_mfcc_correlation:
            mfcc_correlations = [correlate(open_chunk_mfcc[i, :], closed_chunk_mfcc[i, :]) for i in range(open_chunk_mfcc.shape[0])]        
            aggregate_mfcc_correlation = np.mean(mfcc_correlations, axis=0)
            normalized_aggregate_mfcc_correlation = aggregate_mfcc_correlation - np.mean(aggregate_mfcc_correlation[100:len(aggregate_mfcc_correlation)-100])
            # Find the position of maximum correlation
            max_mfcc_correlation = np.max(normalized_aggregate_mfcc_correlation)
            peak_mfcc_indices, _ = find_peaks(normalized_aggregate_mfcc_correlation, height=max_mfcc_correlation*peak_tolerance, distance=distance / hop_length)
            for candidate in peak_mfcc_indices:
                candidate_adjusted = int(candidate * hop_length)


                # mfcc isn't exact, so let's use cross correlation on the regular audio data to get something more precise
                start_adjusted = max(0,int(candidate_adjusted - len(closed_chunk) * 2))
                end_adjusted = min(len(open_chunk), int(candidate_adjusted + len(closed_chunk) * 2))
                centered_open_chunk = open_chunk[start_adjusted:end_adjusted]
                assert len(centered_open_chunk) > 0
                adjusted_correlation = correlate(centered_open_chunk, closed_chunk)

                # print("Adjusted by", candidate_adjusted - (start_adjusted + int(np.argmax(adjusted_correlation))), (candidate_adjusted - (start_adjusted + int(np.argmax(adjusted_correlation)))) / distance)

                candidate_adjusted_adjusted = start_adjusted + int(np.argmax(adjusted_correlation))

                index = correct_peak_index(candidate_adjusted_adjusted, current_chunk_size)
                # print("MFCC addition: from ", (candidate_adjusted + open_start) / sr, ' to ', (index + open_start) / sr, ' or ', (candidate_adjusted_adjusted + open_start) / sr)

                sufficiently_unique = candidate_adjusted_adjusted not in peak_indices
                for ind in peak_indices:
                    if abs(ind - candidate_adjusted_adjusted) < distance:
                        sufficiently_unique = False
                        if prune_types:
                            prune_types['mfcc_correlate_overlap'] += 1
                        break

                if sufficiently_unique and candidate_adjusted < len(open_chunk):

                    peak_indices.append(candidate_adjusted_adjusted)


        peak_indices = sorted( peak_indices )


        if len(peak_indices) == 0:
            # print(f"No peaks found for {closed_start} [{closed_start / sr} sec] / {open_start} [{open_start / sr} sec] {np.max(correlation)}")
            seg_start_cache[key] = None
            return None

        assert( len(closed_chunk) == current_chunk_size)

        scores = []
        max_mfcc_score = 0
        max_correlation_score = 0
        max_relative_volume_score = 0
        max_mfcc_correlation_score = 0
        for candidate in peak_indices:
            candidate_index = correct_peak_index(candidate, current_chunk_size) 

            if upper_bound is not None and not math.isinf(upper_bound) and upper_bound < candidate_index + open_start:
                continue

            open_chunk_here = open_chunk[candidate_index:candidate_index + current_chunk_size]

            if len(open_chunk_here) != current_chunk_size:
                # print(f"Skipping because we couldn't make a chunk of size {current_chunk_size} [{current_chunk_size / sr}] starting at {candidate_index} [{candidate_index / sr}] from chunk of length {len(open_chunk) / sr}")
                continue 

            open_chunk_here_vol_diff = open_chunk_vol_diff[round(candidate_index / hop_length): round((candidate_index + current_chunk_size) / hop_length)       ]

            if use_mfcc_correlation:
                mfcc_correlation_score = aggregate_mfcc_correlation[round(candidate_index / hop_length)]
            else: 
                mfcc_correlation_score = 0
                max_mfcc_correlation_score = 9999

            open_chunk_here_mfcc = open_chunk_mfcc[:,      round(candidate_index / hop_length): round((candidate_index + current_chunk_size) / hop_length)       ]
            mfcc_score = mfcc_similarity(sr, mfcc1=open_chunk_here_mfcc, mfcc2=closed_chunk_mfcc)  

            relative_volume_score = relative_volume_similarity(sr, vol_diff1=open_chunk_here_vol_diff, vol_diff2=closed_chunk_vol_diff)

            scores.append( (candidate_index, mfcc_score, relative_volume_score, correlation[candidate], mfcc_correlation_score) ) 
            
            if correlation[candidate] > max_correlation_score:
                max_correlation_score = correlation[candidate]

            if mfcc_score > max_mfcc_score:
                max_mfcc_score = mfcc_score

            if relative_volume_score > max_relative_volume_score:
                max_relative_volume_score = relative_volume_score

            if mfcc_correlation_score > max_mfcc_correlation_score:
                max_mfcc_correlation_score = mfcc_correlation_score



        if len(scores) == 0:
            seg_start_cache[key] = None
            return None

        if not filter_for_similarity:
            candidates = scores
        else:
            candidates = []
            max_score = 0

            continuity_found = False
            continuity_score = 0
            continuity = None

            for candidate in scores:
                (candidate_index, mfcc_score, rel_vol_score, correlation_score, mfcc_correlation_score) = candidate

                good_by_correlation = correlation_score >= max_correlation_score * (peak_tolerance + (1 - peak_tolerance) * .75)
                good_by_mfcc = mfcc_score >= max_mfcc_score * (peak_tolerance + (1 - peak_tolerance) * .75)
                good_by_rel_vol = rel_vol_score >= max_relative_volume_score * (peak_tolerance + (1 - peak_tolerance) * .75) 
                # good_by_mfcc_correlation = mfcc_correlation_score >= max_mfcc_correlation_score * (peak_tolerance + (1 - peak_tolerance) * .5)  

                if good_by_mfcc or good_by_rel_vol or good_by_correlation: # or good_by_mfcc_correlation: 
                    candidates.append(candidate)

                #score = correlation_score / max_correlation_score + mfcc_score / max_mfcc_score + rel_vol_score / max_relative_volume_score
                score = correlation_score / max_correlation_score
                if score > max_score:
                    max_score = score

                if candidate_index < 2:
                    continuity_found = True
                    continuity_score = score
                    continuity = candidate                

            # Sometimes there is a rhythm that causes scope_segment to frequently drop out of it, and then find_next_segment_start_candidates
            # returns the next part, with some others. This can cause bad branching. So we'll just return the continuation if we 
            # measure it as the best next segment.
            if prune_for_continuity and continuity_found and continuity_score > .98 * max_score:
                # print("continuity prune")
                prune_types['continuity'] += len(candidates) - 1

                candidates = [continuity[0]]
                seg_start_cache[key] = candidates
                seg_start_cache_effectiveness["misses"] += 1                
                return candidates


        # candidates_by_time = [c[0] for c in candidates]
        candidates.sort(key=lambda x: .5 * x[1] / max_mfcc_score + .125 * x[2] / max_relative_volume_score + .125 * x[3] / max_correlation_score, reverse=True)

        # Helps us examine how the system is perceiving candidate starting locations
        if use_mfcc_correlation:
            points_of_interest = [
                # (abs(closed_start - 146.5 * sr) < 2 * sr and open_start < 600 * sr),
                # (abs(closed_start - 33.6 * sr) < 2 * sr) # and open_start < 400 * sr),
            ]
            has_point_of_interest = False
            for pi in points_of_interest:
                has_point_of_interest = has_point_of_interest or pi

            if has_point_of_interest or show_plots:
                plt.figure(figsize=(21, 6))
                
                plt.subplot(1, 3, 1)
                plt.title("Standard Correlation")
                new_x_values = np.arange(len(correlation)) / sr + open_start / sr
                plt.plot(new_x_values, correlation)
                plt.scatter(new_x_values[peak_indices], correlation[peak_indices], color='blue')
                plt.axhline(y=cross_max * peak_tolerance, color='b', linestyle='--')
                plt.xlabel("Time (s)")
                plt.grid(True)  # Adds grid lines

                plt.subplot(1, 3, 2)
                plt.title("Aggregate MFCC Correlation")
                new_x_values = np.arange(len(normalized_aggregate_mfcc_correlation)) * hop_length / sr + open_start / sr
                plt.plot(new_x_values, normalized_aggregate_mfcc_correlation)
                plt.scatter(new_x_values[peak_mfcc_indices], normalized_aggregate_mfcc_correlation[peak_mfcc_indices], color='red')
                plt.axhline(y=max_mfcc_correlation * (peak_tolerance + (1 - peak_tolerance)/2), color='r', linestyle='--')
                plt.xlabel("Time (s)")
                plt.grid(True)  # Adds grid lines

                plt.subplot(1, 3, 3)
                plt.title(f"selected for current_start {closed_start / sr}")

                # plt.scatter([int(c[0] / sr + open_start / sr) for c in candidates], [.5 * x[1] / max_mfcc_score + .125 * x[2] / max_relative_volume_score + .125 * x[3] / max_correlation_score for x in candidates], color='red')
                # plt.scatter([int(c[0] / sr + open_start / sr) for c in scores if c not in candidates], [.5 * x[1] / max_mfcc_score + .125 * x[2] / max_relative_volume_score + .125 * x[3] / max_correlation_score for x in scores if x not in candidates], color='blue')

                plt.scatter([int(c[0] / sr + open_start / sr) for c in candidates], [x[1] / max_mfcc_score for x in candidates], color='green')
                plt.scatter([int(c[0] / sr + open_start / sr) for c in scores if c not in candidates], [x[1] / max_mfcc_score for x in scores if x not in candidates], color='purple')

                plt.xlabel("Time (s)")
                plt.grid(True)  # Adds grid lines

                
                plt.show()
        
        candidates = [c[0] for c in candidates]

        seg_start_cache[key] = candidates
        seg_start_cache_effectiveness["misses"] += 1
    else:
        candidates = seg_start_cache[key]
        seg_start_cache_effectiveness["hits"] += 1

    if seg_start_cache_effectiveness["hits"] + seg_start_cache_effectiveness["misses"] % 1000 == 500:
        print(seg_start_cache_effectiveness)

    return candidates
    
