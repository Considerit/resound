import math, random
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import correlate, find_peaks
from cross_expander.scoring_and_similarity import mse_mfcc_similarity, mfcc_cosine_similarity, raw_cosine_similarity


from utilities import conf, on_press_key
from utilities import conversion_audio_sample_rate as sr


def unique(list1):
    present = {}
    new_list = []
    for l in list1:
        if not l[0] in present:
            new_list.append(l)
            present[l[0]] = True
    return new_list




def correct_peak_index(peak_index, chunk_len):
    # return max(0,peak_index)
    return max(0, peak_index - (chunk_len - 1))

seg_start_cache = {}
paths_at_segment_start = {}
full_search_cache = {}

def initialize_segment_start_cache():
    global seg_start_cache

    seg_start_cache.clear()
    full_search_cache.clear()
    paths_at_segment_start.clear()



# For debugging:
# If ground truth is defined, we try to filter down to only the paths
# that include the ground truth. It isn't perfect, but can really help
# to figure out if the system at least generates the ground truth path.
# Even if it doesn't select it.

def find_segment_starts(reaction, open_chunk, open_chunk_mfcc, closed_chunk, closed_chunk_mfcc, current_chunk_size, peak_tolerance, open_start, closed_start, distance, full_search=True, prune_for_continuity=False, prune_types=None, upper_bound=None, filter_for_similarity=True, current_path=None):
    global full_search_start_cache
    global full_search_cache    

    global seg_start_cache
    global paths_at_segment_start

    force_ground_truth = conf.get("force_ground_truth_paths")


    hop_length = conf.get('hop_length')

    if force_ground_truth and full_search: 
        gt = reaction.get('ground_truth')
        if gt: 
            gt_current_start = 0
            alright = False
            for idx, (start,end) in enumerate(gt):
                if closed_start >= gt_current_start - (idx + 1) * sr and closed_start <= gt_current_start + (end-start) + (idx + 1) * sr:
                    if open_start < end + (idx + 1) * sr:
                        # print("ALRIGHT!", open_start / sr, end / sr, closed_start / sr, (gt_current_start / sr - (idx + 1)), (gt_current_start + (end-start))/sr + (idx + 1))
                        alright = True
                gt_current_start += end - start
            if not alright:
                # print('petty prune')
                return -1 


    key = f"{conf.get('song_key')} {open_start} {closed_start} {len(open_chunk)} {len(closed_chunk)} {upper_bound} {peak_tolerance} {filter_for_similarity} {prune_for_continuity} {hop_length}"

    if full_search:
        full_search_key = f"{conf.get('song_key')} {closed_start} {len(closed_chunk)} {upper_bound} {peak_tolerance} {filter_for_similarity} {prune_for_continuity} {hop_length}"
        if full_search_key in full_search_cache:
            prior_open_start, prior_candidates = full_search_cache[full_search_key]
            if prior_open_start <= open_start:
                new_candidates = [c for c in prior_candidates if c >= open_start]
                if prune_types:
                    prune_types['full_search_start_cache'] += 1

                # print(f"HIT at {open_start}! {full_search_start_cache[True]} / {(full_search_start_cache[True] + full_search_start_cache[False])} ({100 * full_search_start_cache[True] / (full_search_start_cache[True] + full_search_start_cache[False]):.1f}%) ({100 * full_search_start_cache['earlier'] / (full_search_start_cache['earlier'] + full_search_start_cache[False]):.1f}%)")
                # print("new", new_candidates)   
                # print("old", modified_candidates)   

                return [c - open_start for c in prior_candidates]


    if key not in seg_start_cache:

        # print(key)
        if upper_bound is not None:
            prev = len(open_chunk)
            open_chunk = open_chunk[:int(upper_bound - open_start + current_chunk_size)]            
            open_chunk_mfcc = open_chunk_mfcc[:, :int((upper_bound - open_start + current_chunk_size) / hop_length)]

            # print(f"\tConstraining open chunk from size {prev} to {len(open_chunk)}  [{len(open_chunk) / prev * 100}% of original]")

        assert len(open_chunk) > 0
        assert len(closed_chunk) > 0

        correlation = correlate(open_chunk, closed_chunk)
        cross_max = np.max(correlation)
        # Find peaks
        peak_indices, _ = find_peaks(correlation, height=cross_max*(peak_tolerance+(1-peak_tolerance) * .25), distance=distance)
        peak_indices2, _ = find_peaks(correlation, height=cross_max*peak_tolerance, distance=3*distance)
        peak_indices = np.unique(np.concatenate((peak_indices, peak_indices2)))


        peak_indices = [ [correct_peak_index(pi, current_chunk_size), correlation[pi], 0] for pi in peak_indices.tolist()  ]



        if full_search:
            mfcc_correlations = [correlate(open_chunk_mfcc[i, :], closed_chunk_mfcc[i, :]) for i in range(open_chunk_mfcc.shape[0])]        
            aggregate_mfcc_correlation = np.mean(mfcc_correlations, axis=0)
            normalized_aggregate_mfcc_correlation = aggregate_mfcc_correlation - np.mean(aggregate_mfcc_correlation[100:len(aggregate_mfcc_correlation)-100])
            # Find the position of maximum correlation
            max_mfcc_correlation = np.max(normalized_aggregate_mfcc_correlation)
            normalized_aggregate_mfcc_correlation = normalized_aggregate_mfcc_correlation / max_mfcc_correlation
            peak_mfcc_indices, _ = find_peaks(normalized_aggregate_mfcc_correlation, height=peak_tolerance, distance=3*distance / hop_length)
            
            for peak in peak_indices:
                peak[2] = normalized_aggregate_mfcc_correlation[round(peak[0] / hop_length)]


            # adjust the peaks with cross correlation, as the mfccs are sampled and lose time information b/c of FFT to frequency domain
            adjusted_peaks = []
            adjusted_mfcc_peaks_for_plot = []
            for candidate in peak_mfcc_indices:
                candidate_adjusted = int(candidate * hop_length)

                # mfcc isn't exact, so let's use cross correlation on the regular audio data to get something more precise
                start_adjusted = max(0,int(candidate_adjusted - current_chunk_size))
                end_adjusted = min(len(open_chunk), int(candidate_adjusted + current_chunk_size))
                centered_open_chunk = open_chunk[start_adjusted:end_adjusted]
                assert len(centered_open_chunk) > 0
                adjusted_correlation = correlate(centered_open_chunk, closed_chunk)

                candidate_adjusted_adjusted = start_adjusted + int(np.argmax(adjusted_correlation))

                index = correct_peak_index(candidate_adjusted_adjusted, current_chunk_size)
                # print("MFCC addition: from ", (candidate_adjusted + open_start) / sr, ' to ', (index + open_start) / sr, ' or ', (candidate_adjusted_adjusted + open_start) / sr)

                adjusted_mfcc_peaks_for_plot.append(int(index / hop_length))
                adjusted_peaks.append( (candidate, candidate_adjusted, candidate_adjusted_adjusted, index)  )

            # remove duplicates within the mfcc correlates that occur after the cross correlation
            filtered_adjusted_peaks = []
            for i, (candidate, candidate_adjusted, candidate_adjusted_adjusted, index) in enumerate(adjusted_peaks):
                sufficiently_unique = True
                for j, (candidate2, candidate_adjusted2, candidate_adjusted_adjusted2, index2) in enumerate(adjusted_peaks):
                    if i != j:
                        if abs(index2 - index) < distance:
                            score1 = normalized_aggregate_mfcc_correlation[candidate]
                            score2 = normalized_aggregate_mfcc_correlation[candidate2]
                            sufficiently_unique = sufficiently_unique and (score1 > score2 or (score1 == score2 and i < j)  )
                if sufficiently_unique:
                    filtered_adjusted_peaks.append((candidate, candidate_adjusted, candidate_adjusted_adjusted, index))

            # now remove duplicates with the basic cross correlation
            new_peaks = []
            for candidate, candidate_adjusted, candidate_adjusted_adjusted, index in filtered_adjusted_peaks:
                sufficiently_unique = True
                for pi in peak_indices:
                    ind, score, mfcc_mse_score = pi
                    if abs(ind - index) < distance:
                        pi[2] = normalized_aggregate_mfcc_correlation[candidate]
                        
                        sufficiently_unique = False
                        if prune_types:
                            prune_types['mfcc_correlate_overlap'] += 1
                        break

                if sufficiently_unique and candidate_adjusted < len(open_chunk):
                    correlation_score = correlation[candidate_adjusted_adjusted]
                    mfcc_correlation_score = normalized_aggregate_mfcc_correlation[candidate]
                    new_peaks.append( [index, correlation_score, mfcc_correlation_score]  )
            
            peak_indices.extend(new_peaks)

        if len(peak_indices) == 0:
            # print(f"No peaks found for {closed_start} [{closed_start / sr} sec] / {open_start} [{open_start / sr} sec] {np.max(correlation)}")
            seg_start_cache[key] = None
            return None

        assert( len(closed_chunk) == current_chunk_size)


        scores = []
        max_mfcc_mse_score = 0
        max_correlation_score = 0
        max_mfcc_correlation_score = 0
        max_mfcc_cosine_score = 0
        max_raw_cosine_score = 0 
        max_composite_score = 0 

        candidates_seen = {}
        for candidate_index, correlation_score, mfcc_correlation_score in peak_indices:
            if candidate_index in candidates_seen: # ensure unique indices
                continue
            candidates_seen[candidate_index] = True

            if upper_bound is not None and not math.isinf(upper_bound) and upper_bound < candidate_index + open_start:
                continue

            open_chunk_here = open_chunk[candidate_index:candidate_index + current_chunk_size]

            if len(open_chunk_here) != current_chunk_size:
                # print(f"Skipping because we couldn't make a chunk of size {current_chunk_size} [{current_chunk_size / sr}] starting at {candidate_index} [{candidate_index / sr}] from chunk of length {len(open_chunk) / sr}")
                continue 

            open_chunk_here_mfcc = open_chunk_mfcc[:,      round(candidate_index / hop_length): round((candidate_index + current_chunk_size) / hop_length)       ]
            mfcc_mse_score = mse_mfcc_similarity(open_chunk_here_mfcc, closed_chunk_mfcc)  

            mfcc_cosine_score = mfcc_cosine_similarity(open_chunk_here_mfcc, closed_chunk_mfcc)  

            raw_cosine_score = raw_cosine_similarity(open_chunk[candidate_index:candidate_index + current_chunk_size], closed_chunk)

            composite_score = raw_cosine_score * mfcc_cosine_score * mfcc_mse_score

            scores.append( (candidate_index, mfcc_mse_score, correlation_score, mfcc_correlation_score, mfcc_cosine_score, raw_cosine_score, composite_score) ) 
            
            if correlation_score > max_correlation_score:
                max_correlation_score = correlation_score

            if mfcc_mse_score > max_mfcc_mse_score:
                max_mfcc_mse_score = mfcc_mse_score

            if mfcc_correlation_score > max_mfcc_correlation_score:
                max_mfcc_correlation_score = mfcc_correlation_score

            if mfcc_cosine_score > max_mfcc_cosine_score:
                max_mfcc_cosine_score = mfcc_cosine_score

            if raw_cosine_score > max_raw_cosine_score:
                max_raw_cosine_score = raw_cosine_score

            if composite_score > max_composite_score:
                max_composite_score = composite_score



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
                (candidate_index, mfcc_mse_score, correlation_score, mfcc_correlation_score, mfcc_cosine_score, raw_cosine_score, composite_score) = candidate

                good_by_correlation = correlation_score >= max_correlation_score * (peak_tolerance + (1 - peak_tolerance) * .75) # and (mfcc_mse_score > max_mfcc_mse_score * peak_tolerance * .75)
                good_by_mfcc_correlation = False #full_search and mfcc_correlation_score >= max_mfcc_correlation_score * (peak_tolerance + (1 - peak_tolerance) * .25) # and (mfcc_mse_score > max_mfcc_mse_score * peak_tolerance or mfcc_cosine_score > max_mfcc_cosine_score * peak_tolerance)

                good_by_mfcc_mse = mfcc_mse_score >= max_mfcc_mse_score * (peak_tolerance + (1 - peak_tolerance) * .25)
                good_by_mfcc_cosine = mfcc_cosine_score >= max_mfcc_cosine_score * (peak_tolerance + (1 - peak_tolerance) * .25)
                # good_by_raw_cosine = raw_cosine_score >= max_raw_cosine_score * (peak_tolerance + (1 - peak_tolerance) * .25)
                good_by_composite_score = composite_score >= max_composite_score * (peak_tolerance + (1 - peak_tolerance) * .25)


                if good_by_mfcc_mse or good_by_correlation or good_by_mfcc_correlation or good_by_mfcc_cosine or good_by_composite_score: #or good_by_raw_cosine: 
                    candidates.append(candidate)

                score = correlation_score / max_correlation_score
                if score > max_score:
                    max_score = score

                if candidate_index < 2:
                    continuity_found = True
                    continuity_score = score
                    continuity = candidate                

            # Sometimes there is a rhythm that causes find_segment_end to frequently drop out of it, and then find_segment_starts
            # returns the next part, with some others. This can cause bad branching. So we'll just return the continuation if we 
            # measure it as the best next segment.
            if prune_for_continuity and continuity_found and continuity_score > .98 * max_score:
                # print("continuity prune")
                prune_types['continuity'] += len(candidates) - 1

                candidates = [continuity[0]]
                seg_start_cache[key] = candidates
                return candidates

        # candidates_by_time = [c[0] for c in candidates]
        # candidates.sort(key=lambda x: .5 * x[1] / max_mfcc_mse_score + .5 * x[3] / max_correlation_score, reverse=True)

        # Helps us examine how the system is perceiving candidate starting locations
        if filter_for_similarity:
            points_of_interest = [
                # (abs(closed_start - 146.9 * sr) < 2 * sr and open_start < 600 * sr),
                # (abs(closed_start - 149 * sr) < 1 * sr and open_start > 620 * sr and open_start < 650 * sr),

                # (abs(closed_start - 33.6 * sr) < 2 * sr) # and open_start < 400 * sr),
                # (abs(closed_start - 35 * sr) < 2 * sr and open_start < 105 * sr),  # ian diazapam; should generate 127
                # (abs(closed_start - 42 * sr) < 2 * sr and open_start < 138 * sr),  # ian diazapam; should generate 220
                # (abs(closed_start - 60 * sr) < 3 * sr and open_start < 240 * sr),  # ian diazapam; should generate 278
                # (abs(closed_start - 72 * sr) < 3 * sr and open_start < 295 * sr),  # ian diazapam; should generate 326
                # (abs(closed_start - 89 * sr) < 4 * sr and open_start < 348 * sr),  # ian diazapam; should generate 403
                # (abs(closed_start - 109 * sr) < 4 * sr and open_start < 428 * sr),  # ian diazapam; should generate 481
                # (abs(closed_start - 148 * sr) < 5 * sr and open_start < 525 * sr),  # ian diazapam; should generate 548
                # (abs(closed_start - 164 * sr) < 5 * sr and open_start < 568 * sr),  # ian diazapam; should generate 597
                # (abs(closed_start - 169 * sr) < 6 * sr and open_start < 607 * sr),  # ian diazapam; should generate 621
                # (abs(closed_start - 194 * sr) < 6 * sr and open_start < 651 * sr),  # ian diazapam; should generate 661

                # (abs(closed_start - 146.9 * sr) < 2 * sr and open_start < 624 * sr), # dicodec genesis; should generate 633.6
                # (abs(closed_start - 148 * sr) < 2 * sr and open_start < 634 * sr), # dicodec genesis; should generate 649


                # (abs(closed_start - 145 * sr) < 2 * sr and open_start < 284 * sr), # thatsnotactingeither suicide; should generate 450

                # (abs(closed_start - 1.2 * sr) < .15 * sr and open_start < 53 * sr), # dan wheeler crutch; should generate continuity 52.4

            ]
            has_point_of_interest = False #full_search
            for pi in points_of_interest:
                has_point_of_interest = has_point_of_interest or pi

            if has_point_of_interest:
                plt.figure(figsize=(14, 10))
                
                plt.subplot(2, 3, 1)
                plt.title("Standard Correlation")
                new_x_values = [ (x + open_start) / sr for x in range(len(correlation))]  #np.arange(len(correlation)) / sr + open_start / sr
                plt.plot(new_x_values, correlation)
                # print(peak_indices)
                plt.scatter([int(c[0] / sr + open_start / sr) for c in peak_indices], [p[1] for p in peak_indices], color='blue')
                plt.scatter([int(c[0] / sr + open_start / sr) for c in scores], [p[3] for p in scores], color='red')

                plt.axhline(y=cross_max * (peak_tolerance + (1 - peak_tolerance) * .5), color='b', linestyle='--')
                plt.xlabel("Time (s)")
                plt.grid(True)  # Adds grid lines

                plt.xlim(left=open_start / sr, right=new_x_values[-1])

                if full_search:
                    plt.subplot(2, 3, 2)
                    plt.title("Aggregate MFCC Correlation")
                    new_x_values = np.arange(len(normalized_aggregate_mfcc_correlation)) * hop_length / sr + open_start / sr
                    plt.plot(new_x_values, normalized_aggregate_mfcc_correlation)
                    plt.scatter(new_x_values[peak_mfcc_indices], [normalized_aggregate_mfcc_correlation[p] for p in peak_mfcc_indices], color='red')
                    plt.scatter(new_x_values[adjusted_mfcc_peaks_for_plot], [normalized_aggregate_mfcc_correlation[p] for p in adjusted_mfcc_peaks_for_plot], color='green')


                    plt.axhline(y=1 * (peak_tolerance + (1 - peak_tolerance) * .25), color='r', linestyle='--')
                    plt.xlabel("Time (s)")
                    plt.grid(True)  # Adds grid lines
                    plt.ylim(bottom=0)
                    plt.xlim(left=open_start / sr, right=new_x_values[-1])



                plt.subplot(2, 3, 3)
                plt.title(f"mfcc mse similarity for current_start {closed_start / sr}")

                plt.scatter([int(c[0] / sr + open_start / sr) for c in scores],     [x[1] / max_mfcc_mse_score for x in scores], color='purple')
                plt.scatter([int(c[0] / sr + open_start / sr) for c in candidates], [x[1] / max_mfcc_mse_score for x in candidates], color='green')

                plt.xlim(left=open_start / sr, right=new_x_values[-1])
                plt.ylim(bottom=0)

                plt.xlabel("Time (s)")
                plt.grid(True)  # Adds grid lines



                plt.subplot(2, 3, 4)
                plt.title(f"mfcc cosine similarity")

                plt.scatter([int(c[0] / sr + open_start / sr) for c in scores],     [x[4] / max_mfcc_cosine_score for x in scores], color='purple')
                plt.scatter([int(c[0] / sr + open_start / sr) for c in candidates], [x[4] / max_mfcc_cosine_score for x in candidates], color='green')

                plt.xlim(left=open_start / sr, right=new_x_values[-1])
                plt.ylim(bottom=0)

                plt.xlabel("Time (s)")
                plt.grid(True)  # Adds grid lines


                plt.subplot(2, 3, 5)
                plt.title(f"raw cosine similarity")

                plt.scatter([int(c[0] / sr + open_start / sr) for c in scores],     [x[5] / max_raw_cosine_score for x in scores], color='purple')
                plt.scatter([int(c[0] / sr + open_start / sr) for c in candidates], [x[5] / max_raw_cosine_score for x in candidates], color='green')

                plt.xlim(left=open_start / sr, right=new_x_values[-1])
                plt.ylim(bottom=0)

                plt.xlabel("Time (s)")
                plt.grid(True)  # Adds grid lines


                
                plt.show()
        
        candidates = [c[0] for c in candidates]

        if force_ground_truth and full_search and open_start == 0: 
            gt = reaction.get('ground_truth')
            if gt: 
                new_candidates = []

                for candidate in candidates:
                    alright = False
                    for idx, (start,end) in enumerate(gt):

                        if start - (idx + 10) < open_start + candidate and open_start + candidate < start + (idx + 10) * sr:
                            alright = True

                    if alright:
                        new_candidates.append(candidate)

                if len(new_candidates) == 0:
                    return -1 
                else:
                    candidates = new_candidates




        candidates.sort()

        if full_search:
            modified_candidates = [c + open_start for c in candidates]
            full_search_cache[full_search_key] = (open_start, modified_candidates)

        seg_start_cache[key] = candidates
    else:
        if prune_types:
            prune_types['exact_search_start_cache'] += 1

        candidates = seg_start_cache[key]

    return candidates
    
