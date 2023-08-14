import math, random
import numpy as np

from scipy.signal import correlate, find_peaks
from cross_expander.scoring_and_similarity import mfcc_similarity, relative_volume_similarity



def correct_peak_index(peak_index, chunk_len):
    # return max(0,peak_index)
    return max(0, peak_index - (chunk_len - 1))

seg_start_cache = {}
seg_start_cache_effectiveness = {}
seg_starts = {}
def initialize_segment_start_cache():
    global seg_start_cache
    global seg_start_cache_effectiveness


    seg_start_cache.clear()
    seg_start_cache_effectiveness["hits"] = 0
    seg_start_cache_effectiveness["misses"] = 0

    seg_starts.clear()


def find_next_segment_start_candidates(basics, open_chunk, open_chunk_mfcc, open_chunk_vol_diff, closed_chunk, closed_chunk_mfcc, closed_chunk_vol_diff, current_chunk_size, peak_tolerance, open_start, closed_start, distance, prune_for_continuity=False, prune_types=None, upper_bound=None, print_candidates=False, filter_for_similarity=True):
    global seg_start_cache
    global seg_start_cache_effectiveness

    sr = basics.get('sr')
    hop_length = basics.get('hop_length')

    key = f"{open_start} {closed_start} {len(open_chunk)} {len(closed_chunk)} {upper_bound} {peak_tolerance} {filter_for_similarity} {prune_for_continuity} {hop_length}"
    
    # key2 = f"{open_start} {open_start / sr} {closed_start} {closed_start / sr}"
    # if prune_for_continuity:
    #     if key2 not in seg_starts:
    #         seg_starts[key2] = 0
    #     seg_starts[key2] += 1

    if key not in seg_start_cache:

        # print(key)
        if upper_bound is not None:
            prev = len(open_chunk)
            open_chunk = open_chunk[:int(upper_bound - open_start + 2 * current_chunk_size)]
            # print(f"\tConstraining open chunk from size {prev} to {len(open_chunk)}  [{len(open_chunk) / prev * 100}% of original]")

        correlation = correlate(open_chunk, closed_chunk)
        # Find peaks
        peak_indices, _ = find_peaks(correlation, height=np.max(correlation)*peak_tolerance, distance=distance)
        peak_indices = sorted( peak_indices.tolist() )

        
        # if random.random() < .01:
        #     print(seg_start_cache_effectiveness)



        if len(peak_indices) == 0:
            # print(f"No peaks found for {closed_start} [{closed_start / sr} sec] / {open_start} [{open_start / sr} sec] {np.max(correlation)}")
            seg_start_cache[key] = None
            return None

        assert( len(closed_chunk) == current_chunk_size)

        scores = []
        max_mfcc_score = 0
        max_correlation_score = 0
        max_relative_volume_score = 0
        for candidate in peak_indices:
            candidate_index = correct_peak_index(candidate, current_chunk_size) 

            if upper_bound is not None and not math.isinf(upper_bound) and upper_bound < candidate_index + open_start:
                continue

            open_chunk_here = open_chunk[candidate_index:candidate_index + current_chunk_size]

            if len(open_chunk_here) != current_chunk_size:
                # print(f"Skipping because we couldn't make a chunk of size {current_chunk_size} [{current_chunk_size / sr}] starting at {candidate_index} [{candidate_index / sr}] from chunk of length {len(open_chunk) / sr}")
                continue 

            open_chunk_here_mfcc = open_chunk_mfcc[:,      round(candidate_index / hop_length): round((candidate_index + current_chunk_size) / hop_length)       ]
            open_chunk_here_vol_diff = open_chunk_vol_diff[round(candidate_index / hop_length): round((candidate_index + current_chunk_size) / hop_length)       ]

            mfcc_score = mfcc_similarity(sr, mfcc1=open_chunk_here_mfcc, mfcc2=closed_chunk_mfcc)  

            relative_volume_score = relative_volume_similarity(sr, vol_diff1=open_chunk_here_vol_diff, vol_diff2=closed_chunk_vol_diff)

            scores.append( (candidate_index, mfcc_score, relative_volume_score, correlation[candidate]) ) 
            
            if correlation[candidate] > max_correlation_score:
                max_correlation_score = correlation[candidate]

            if mfcc_score > max_mfcc_score:
                max_mfcc_score = mfcc_score

            if relative_volume_score > max_relative_volume_score:
                max_relative_volume_score = relative_volume_score



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
                (candidate_index, mfcc_score, rel_vol_score, correlation_score) = candidate

                good_by_mfcc = mfcc_score >= max_mfcc_score * (peak_tolerance + .2)
                good_by_rel_vol = rel_vol_score >= max_relative_volume_score * (peak_tolerance + .4)
                # joint_goodness = (mfcc_score / max_mfcc_score + rel_vol_score / max_relative_volume_score) / 2 > peak_tolerance

                # if (good_by_mfcc or good_by_rel_vol) and not joint_goodness:
                #     print(f"[restricted] Start at {(closed_start + candidate_index) / sr:.1f} with correlation={100 * correlation_score / max_correlation_score:.1f}% mfcc_similarity={100 * mfcc_score / max_mfcc_score:.1f}% rel_vol_similarity={100 * relative_volume_score / max_relative_volume_score:.1f}%")

                # good_by_mediocrity = (mfcc_score / max_mfcc_score + correlation_score / max_correlation_score + rel_vol_score / max_relative_volume_score) / 3 > peak_tolerance
                if good_by_mfcc or good_by_rel_vol:
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
                return [continuity[0]]

        candidates_by_time = [c[0] for c in candidates]
        candidates.sort(key=lambda x: x[1] * x[2], reverse=True)
        candidates_by_score = [c[0] for c in candidates]

        # create a list that interlaces every other element of candidates_by_time and candidates_by_score, 
        # starting with candidates_by_score. The resulting list should not have duplicates. Only the
        # first instance of the item should stay in the array. 

        candidates = []
        for score, time in zip(candidates_by_score, candidates_by_time):
            if score not in candidates:
                candidates.append(score)
            if time not in candidates:
                candidates.append(time)


        # if print_candidates:
        #     for candidate_index, mfcc_score, relative_volume_score, correlation_score in candidates:
        #         print(f"Comparing start at {(closed_start + candidate_index) / sr:.1f} with correlation={100 * correlation_score / max_correlation_score:.1f}% mfcc_similarity={100 * mfcc_score / max_mfcc_score:.1f}% rel_vol_similarity={100 * relative_volume_score / max_relative_volume_score:.1f}%")


        seg_start_cache[key] = candidates
        seg_start_cache_effectiveness["misses"] += 1
    else:
        candidates = seg_start_cache[key]
        seg_start_cache_effectiveness["hits"] += 1

    if random.random() < .001:
        print(seg_start_cache_effectiveness)

        # for k,v in seg_starts.items():
        #     print(k,v)

    return candidates
    
