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
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from typing import List, Tuple
import traceback
import soundfile as sf
import random
from utilities import samples_per_frame, universal_frame_rate, is_close
from backchannel_isolator import audio_percentile_loudness

from decimal import Decimal, getcontext


import cProfile
import pstats



def correct_peak_index(peak_index, chunk_len):
    # return max(0,peak_index)
    return max(0, peak_index - (chunk_len - 1))


def mfcc_similarity(sr, audio_chunk1=None, audio_chunk2=None, mfcc1=None, mfcc2=None):

    # Compute MFCCs for each audio chunk
    if mfcc1 is None: 
        mfcc1 = librosa.feature.mfcc(y=audio_chunk1, sr=sr, n_mfcc=20)

    if mfcc2 is None: 
        mfcc2 = librosa.feature.mfcc(y=audio_chunk2, sr=sr, n_mfcc=20)


    # Make sure the MFCCs are the same shape
    len1 = mfcc1.shape[1]
    len2 = mfcc2.shape[1]

    if len1 != len2:
        # if abs(len1 - len2) > 1:
        #     print("MFCC SHAPES NOT EQUAL", len1, len2)
        #     # assert(abs(len1 - len2) == 1)

        if len2 > len1:
            mfcc2 = mfcc2[:, :len1]
        else:
            mfcc1 = mfcc1[:, :len2]

        # max_len = max(len1, len2)
        # if len1 < max_len:
        #     padding = max_len - len1
        #     mfcc1 = np.pad(mfcc1, pad_width=((0, 0), (0, padding)), mode='constant')
        # elif len2 < max_len:
        #     padding = max_len - len2
        #     mfcc2 = np.pad(mfcc2, pad_width=((0, 0), (0, padding)), mode='constant')

    # Compute mean squared error between MFCCs
    mse = np.mean((mfcc1 - mfcc2)**2)

    similarity = len(mfcc1) / (1 + mse)
    return similarity


def relative_volume_similarity(sr, audio_chunk1=None, audio_chunk2=None, vol_diff1=None, vol_diff2=None, hop_length=1):

    # Compute MFCCs for each audio chunk
    if vol_diff1 is None: 
        vol_diff1 = audio_percentile_loudness(audio_chunk1, loudness_window_size=100, percentile_window_size=100, std_dev_percentile=None, hop_length=hop_length)

    if vol_diff2 is None: 
        vol_diff2 = audio_percentile_loudness(audio_chunk2, loudness_window_size=100, percentile_window_size=100, std_dev_percentile=None, hop_length=hop_length)

    # Make sure of same shape
    len1 = vol_diff1.shape[0]
    len2 = vol_diff2.shape[0]

    if len1 != len2:
        # if abs(len1 - len2) > 1:
        #     print("VOL DIFF SHAPES NOT EQUAL", len1, len2)
        #     # assert(abs(len1 - len2) == 1)

        if len2 > len1:
            vol_diff2 = vol_diff2[:len1]
        else:
            vol_diff1 = vol_diff1[:len2]



    # Calculate the absolute difference between the two volume differences
    absolute_difference = np.abs(vol_diff1 - vol_diff2)

    # Calculate the sum of absolute_difference
    difference_magnitude = np.sum(absolute_difference)


    # Max difference is for volume inversion, where abs(vol_diff1 - vol_diff2) is always 100 (%)
    max_difference = 100 * len(vol_diff1)
    return max_difference / difference_magnitude




# I have two audio files: a base audio file that contains something like a song and a reaction audio file that contains 
# someone reacting to that base audio file. It includes the base audio, and more. 

# I am trying to create an aligned version of the reaction audio. To help with this I am create a rough bounding of where 
# certain parts of the base audio file aligns with the reaction audio file. This will help pruning a tree exploring 
# potential alignment pathways.  

# I'd like you to help me write a function create_reaction_alignment_bounds. It takes in the reaction audio and the base 
# audio. Here's what it should do: 
#   - Select n equally spaced timestamps from the base audio, not including zero or the end of the base audio.
#   - For each timestamp, select two adjacent 2 sec clips on either side of the timestamp. We use two clips to 
#     minimize the chance that the reactor has paused in the middle of both clips. 
#   - For each clip, find the latest match in the reaction audio to that clip, using the function 
#     find_next_segment_start_candidates. Don't bother using parameters, I'll fill that in. Assume you get 
#     a list of candidate matching indexes of the clip in the reaction audio. Take the greatest of these matching 
#     indexes, between the two clips. This will be the bound value for this timestamp. 
#   - Now ensure the integrity of these timestamp bounds. Make sure that every earlier timestamp bound a is 
#   less than every later timestamp bound b by at least (t(b) - t(a)) where t is the time difference between 
#   b and a in the base audio. If the earlier timestamp bound doesn't satisfy this constraint, set the earlier 
#   bound a to v(b) - (t(b) - t(a)), where v is the value of the bound's latest match in the reaction audio. To 
#   accomplish the integrity checking, walk backwards from the last timestamp.

def create_reaction_alignment_bounds(basics, first_n_samples, n_timestamps = None, peak_tolerance=.5):
    # profiler = cProfile.Profile()
    # profiler.enable()

    base_audio = basics.get('base_audio')
    reaction_audio = basics.get('reaction_audio')
    sr = basics.get('sr')
    hop_length = basics.get('hop_length')
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    reaction_audio_vol_diff = basics.get('reaction_percentile_loudness')
    base_audio_mfcc = basics.get('base_audio_mfcc')
    base_audio_vol_diff = basics.get('song_percentile_loudness')

    clip_length = int(2 * sr)
    base_length_sec = len(base_audio) / sr  # Length of the base audio in seconds

    seconds_per_checkpoint = 24

    if n_timestamps is None:
        n_timestamps = round(base_length_sec / seconds_per_checkpoint) 
    
    timestamps = [i * base_length_sec / (n_timestamps + 1) for i in range(1, n_timestamps + 1)]
    timestamps_samples = [int(t * sr) for t in timestamps]
    
    # Initialize the list of bounds
    bounds = []

    print(f"Creating alignment bounds at {timestamps}")
    
    # For each timestamp
    for i,ts in enumerate(timestamps_samples):
        # Define the segments on either side of the timestamp

        segment_times = [
          (max(0, ts - clip_length), ts),
          (max(0, ts - int(clip_length / 2)), min(len(base_audio), ts + int(clip_length / 2) )),
          (ts, min(len(base_audio), ts + clip_length))
        ]

        segments = [  (s, e, base_audio[s:e], base_audio_mfcc[:, round(s / hop_length): round(e / hop_length)], base_audio_vol_diff[round(s / hop_length): round(e / hop_length) ]) for s,e in segment_times  ]


        # for j, segment in enumerate(segments):
        #     filename = f"segment_{i}_{j}.wav"
        #     sf.write(filename, segment, sr)
        
        # Initialize the list of max indices for the segments
        max_indices = []
        
        print(f"ts: {ts / sr}")
        # For each segment
        for start, end, chunk, chunk_mfcc, chunk_vol_diff in segments:
            # Find the candidate indices for the start of the matching segment in the reaction audio

            candidates = find_next_segment_start_candidates(
                                    basics=basics, 
                                    open_chunk=reaction_audio[start:], 
                                    open_chunk_mfcc=reaction_audio_mfcc[:, round(start / hop_length):],
                                    open_chunk_vol_diff=reaction_audio_vol_diff[round(start / hop_length):],
                                    closed_chunk=chunk, 
                                    closed_chunk_mfcc= chunk_mfcc,
                                    closed_chunk_vol_diff=chunk_vol_diff,
                                    current_chunk_size=clip_length, 
                                    peak_tolerance=peak_tolerance, 
                                    open_start=start,
                                    closed_start=start, 
                                    distance=first_n_samples, 
                                    filter_for_similarity=True, 
                                    print_candidates=True  )



            print(f"\tCandidates: {candidates}  {max(candidates)}")

            # Find the maximum candidate index
            max_indices.append(ts + max(candidates) + clip_length * 2)
        
        # Add the maximum of the max indices to the bounds
        bounds.append(max(max_indices))

        timestamps_samples[i] -= clip_length
    
    # Now, ensure the integrity of the bounds
    for i in range(len(bounds) - 2, -1, -1):  # Start from the second last element and go backward
        # If the current bound doesn't satisfy the integrity condition
        if bounds[i] >= bounds[i+1] - (timestamps_samples[i+1] - timestamps_samples[i]):
            # Update the current bound
            bounds[i] = bounds[i+1] - (timestamps_samples[i+1] - timestamps_samples[i])

    alignment_bounds = list(zip(timestamps_samples, bounds))
    print(f"The alignment bounds:")
    for base_ts, last_reaction_match in alignment_bounds:
        print(f"\t{base_ts / sr}  <=  {last_reaction_match / sr}")


    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
    # stats.print_stats()

    return alignment_bounds


def in_bounds(bound, base_start, reaction_start):
    return reaction_start <= bound

def get_bound(alignment_bounds, base_start, reaction_end): 
    for base_ts, last_reaction_match in alignment_bounds:
        if base_start < base_ts:
            return last_reaction_match
    return reaction_end


def get_initial_checkpoints(basics): 
    base_audio = basics.get('base_audio')
    sr = basics.get('sr')

    samples_per_checkpoint = 10 * sr 

    timestamps = []
    s = samples_per_checkpoint
    while s < len(base_audio):
        if s / basics.get('sr') >= 30:
            timestamps.append(s)
        s += samples_per_checkpoint

    return timestamps


def add_new_checkpoint(checkpoints, current_start, paths_by_checkpoint, basics):
    if current_start in checkpoints or current_start / basics.get('sr') < 30: 
        return

    checkpoints.append(current_start)
    checkpoints.sort()

    idx = checkpoints.index(current_start)
    if idx < len(checkpoints) - 1:
        reference_checkpoint = checkpoints[idx + 1]

        if reference_checkpoint not in paths_by_checkpoint:
            paths_by_checkpoint[reference_checkpoint] = {'prunes_here': 0, 'paths': []}

        for rs, scr, current_path in paths_by_checkpoint[reference_checkpoint]["paths"]:
            partial_score = calculate_partial_score(current_path, current_start, basics)
            if partial_score is None:
                continue
            if current_start not in paths_by_checkpoint:
                paths_by_checkpoint[current_start] = {'prunes_here': 0, 'paths': []}

            paths_by_checkpoint[current_start]['paths'].append( (partial_score[0], partial_score[1], list(current_path))  )



def get_chunk_score(basics, reaction_start, reaction_end, current_start, current_end):

    base_audio = basics.get('base_audio')
    reaction_audio = basics.get('reaction_audio')

    base_audio_mfcc = basics.get('base_audio_mfcc')
    base_audio_vol_diff = basics.get('song_percentile_loudness')
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    reaction_audio_vol_diff = basics.get('reaction_percentile_loudness')
    sr = basics.get('sr')
    hop_length = basics.get('hop_length')


    chunk = base_audio[current_start:current_end]
    reaction_chunk = reaction_audio[reaction_start:reaction_end]            

    mfcc_react_chunk = reaction_audio_mfcc[:, round(reaction_start / hop_length):round(reaction_end / hop_length)]
    mfcc_song_chunk =      base_audio_mfcc[:, round(current_start / hop_length):round(current_end / hop_length)]

    # voldiff_react_chunk = reaction_audio_vol_diff[round(reaction_start / hop_length):round(reaction_end / hop_length)]
    # voldiff_song_chunk =      base_audio_vol_diff[round(current_start / hop_length):round(current_end / hop_length)]
    
    mfcc_score = mfcc_similarity(sr, mfcc1=mfcc_song_chunk, mfcc2=mfcc_react_chunk)
    # rel_volume_alignment = relative_volume_similarity(sr, vol_diff1=voldiff_song_chunk, vol_diff2=voldiff_react_chunk)

    # alignment = math.log(1 + 100 * mfcc_score) * math.log(1 + rel_volume_alignment)
    alignment = mfcc_score
    return alignment

def find_correlation_end(current_start, reaction_start, basics, options, step, scores = [], reaction_end=None, end_at=None, max_score=0, cache={}):
    expansion_tolerance = options.get('expansion_tolerance')
    backoff = options.get('segment_end_backoff')

    base_audio = basics.get('base_audio')
    base_audio_mfcc = basics.get('base_audio_mfcc')
    reaction_audio = basics.get('reaction_audio')
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    sr = basics.get('sr')


    if reaction_end == None:
        current_end = current_start + step
        reaction_end = reaction_start + step
    else: 
        current_end = current_start + reaction_end - reaction_start 

    if end_at == None: 
        end_at = len(reaction_audio)

    assert( reaction_end - reaction_start == current_end - current_start   )


    ##########
    # Progressively expand the size of the segment, until the score falls below expansion_tolerance * max score for this segment...

    segment_scores = []
    next_end_at = end_at or len(reaction_audio)
    while current_end < len(base_audio) and reaction_end < end_at: 

        key = f"{current_start}:{current_end}"

        if key not in cache:
            cache[key] = get_chunk_score(basics, reaction_start, reaction_end, current_start, current_end) 

        chunk_score = cache[key]

        if chunk_score > max_score:
            max_score = chunk_score

        # print(f"Expanding {chunk_score} [{max_score}]: base {(current_start) / sr} - {(current_end) / sr} <==> react {(reaction_start) / sr} - {(reaction_end)/sr} with step {step / sr}")

        segment_scores.append( (reaction_start, reaction_end, chunk_score, current_start, current_end) )

        assert( reaction_end - reaction_start == current_end - current_start   )
        if chunk_score < expansion_tolerance * max_score:
            next_end_at = reaction_end
            break

        current_end += step
        reaction_end += step

    #############
    # Now we're going to shrink the end of the segment based on the slope of the scores, trying to work backwards to 
    # a reasonable pause or rewind breakpoint

    scores += [(start, end, chunk_score) for (start, end, chunk_score, _, _) in segment_scores]

    slopes = []
    window = 2
    for i, score in enumerate(segment_scores):
        prev = 0

        for p in range(i - window, i):
            if p >= 0:
                prev += segment_scores[p][2]

        prev /= window

        after = 0
        for p in range(i, i + window):
            if p < len(segment_scores):
                after += segment_scores[p][2]
            else: 
                after += segment_scores[len(segment_scores) - 1][2]
        after /= window

        slope = (after - prev) / (2 * window * step / sr)

        # print(f"{(reaction_start + i * step) / sr} : {slope}     {range(i - window - 1, i)}   {range(i + 1, i + window + 1)}  {prev} - {after}")
        slopes.append( (slope, score)  )


    slopes.reverse()
    break_point = None
    last_neg_slopes = None
    neg_slope_avg = 0

    # pop off any stray ending positive slopes
    for i, (slope, (_,_,_,_,_)) in enumerate(slopes):
        if (slope < 0):
            slopes = slopes[i:]
            break

    for i, (slope, (_,_,_,_,_)) in enumerate(slopes):
        if slope < 0: 
            neg_slope_avg += slope
        if i == len(slopes) - 1 or slope >= 0:
             last_neg_slopes = slopes[0:i]
             break


    if not last_neg_slopes or len(last_neg_slopes) == 0:
        assert( current_end - current_start == reaction_end - reaction_start)        
        return (current_end, reaction_end, sorted(scores, key=lambda score: score[1]))

    last_neg_slopes.reverse()

    # print("last_neg_slopes", last_neg_slopes)

    neg_slope_avg /= len(last_neg_slopes)

    for i, (slope, (_, reaction_end,_,_,_)) in enumerate(last_neg_slopes):
        if slope <= neg_slope_avg:
            next_end_at = max(reaction_start, min(end_at, reaction_end + step * window))
            break_point = max(reaction_start, reaction_end - step * window)

            while i > 0: 
                i -= 1
                if last_neg_slopes[i][0] <= slope / (2 * window):

                    next_end_at = max(reaction_start, min(end_at, last_neg_slopes[i][1][1] + step * window))
                    break_point = max(reaction_start, last_neg_slopes[i][1][1] - step * window)

                else: 
                    break

            break


    if step > 250:

        # print(f"RECURSING AROUND {break_point / sr} - {next_end_at / sr} with neg_slope_avg of {neg_slope_avg}")

        return find_correlation_end(current_start, reaction_start, basics, options, step = step // 2, scores = scores, reaction_end = break_point, end_at = next_end_at, max_score = max_score, cache = cache )
    else: 
        reaction_end = break_point
        current_end = current_start + reaction_end - reaction_start

        if backoff > 0:

            if reaction_end - reaction_start >= 2 * backoff:
                reaction_end -= backoff
                current_end -= backoff
            else: 
                decrement = int((reaction_end - reaction_start) / 2)
                if reaction_end - reaction_start >= 2 * decrement:
                    reaction_end -= decrement
                    current_end -= decrement

        # seek to a frame boundary
        while (current_end - current_start) % samples_per_frame() > 0 and reaction_end < len(reaction_audio) and current_end < len(base_audio):
            current_end += 1
            reaction_end += 1

        result = Decimal(current_end - current_start) / Decimal(sr) * Decimal(universal_frame_rate())
        # print(f"by samples = {(current_end - current_start) % samples_per_frame()}  by rounding {result}")

        assert((current_end - current_start) % samples_per_frame() == 0)
        assert( is_close(result, round(result) )  )

        return (current_end, reaction_end, sorted(scores, key=lambda score: score[1]))

import math


seg_start_cache = {}
seg_start_cache_effectiveness = {"hits": 0, "misses": 0}
def find_next_segment_start_candidates(basics, open_chunk, open_chunk_mfcc, open_chunk_vol_diff, closed_chunk, closed_chunk_mfcc, closed_chunk_vol_diff, current_chunk_size, peak_tolerance, open_start, closed_start, distance, prune_for_continuity=False, prune_counts=None, upper_bound=None, print_candidates=False, filter_for_similarity=True):
    global seg_start_cache
    global seg_start_cache_effectiveness

    sr = basics.get('sr')
    hop_length = basics.get('hop_length')

    key = f"{open_start} {closed_start} {len(open_chunk)} {len(closed_chunk)} {upper_bound} {peak_tolerance} {filter_for_similarity} {prune_for_continuity} {hop_length}"
    
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

        seg_start_cache_effectiveness["misses"] += 1
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
                prune_counts['continuity'] += len(candidates) - 1
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
    else:
        candidates = seg_start_cache[key]
        seg_start_cache_effectiveness["hits"] += 1



    return candidates
    



#####################################################################
# Given a starting index into the reaction, find a good ending index
def scope_segment(basics, current_start, reaction_start, candidate_segment_start, current_chunk_size, options):

    reverse_search_bound = options.get('reverse_search_bound')
    peak_tolerance = options.get('peak_tolerance')
    n_samples = options.get('n_samples')
    first_n_samples = options.get('first_n_samples')

    base_audio = basics.get('base_audio')
    reaction_audio = basics.get('reaction_audio')
    base_audio_mfcc = basics.get('base_audio_mfcc')
    base_audio_vol_diff = basics.get('song_percentile_loudness')
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    reaction_audio_vol_diff = basics.get('reaction_percentile_loudness')
    hop_length = basics.get('hop_length')
    sr = basics.get('sr')


    #####################
    # Sometimes we're off in our segment start because, e.g. a reactor doesn't start from the beginning. So our very first match 
    # is really low quality, which can cause problems like never popping out of the first segment. 
    # To address this, we do a reverse match for a segment start: find the best start of reaction_chunk 
    # in base_audio[current_start:]. The match should be at the beginning. If it isn't, then we're off base. 
    # We have missing base audio. We can then try to backfill that by matching the missing segment with a 
    # smaller minimum segment size (size of the missing chunk), so that we 
    # (1) recover that base audio and (2) are aligned for subsequent sequences.  

    open_end = min(current_start+current_chunk_size+int(reverse_search_bound * sr), len(base_audio))
    reverse_chunk_size = min(current_chunk_size, open_end - current_start)

    candidate_reaction_chunk_start = reaction_start + candidate_segment_start
    candidate_reaction_chunk_end = reaction_start + candidate_segment_start + reverse_chunk_size
    candidate_reaction_chunk = reaction_audio[candidate_reaction_chunk_start:candidate_reaction_chunk_end]
    if reverse_chunk_size > len(candidate_reaction_chunk):
        reverse_chunk_size = len(candidate_reaction_chunk)
        open_end = min(open_end, current_start + reverse_chunk_size)


    open_base_chunk = base_audio[current_start:open_end]

    # print(f'\nDoing reverse index search  reaction_start={reaction_start+candidate_segment_start}  current_start={current_start}  {reverse_chunk_size} {len(candidate_reaction_chunk)} {len(open_base_chunk)}')

    reverse_index = find_next_segment_start_candidates(
                        basics = basics, 
                        open_chunk=open_base_chunk, 
                        open_chunk_mfcc=base_audio_mfcc[:, round(current_start / hop_length):round(open_end / hop_length)], 
                        open_chunk_vol_diff=base_audio_vol_diff[round(current_start / hop_length):round(open_end / hop_length)], 
                        closed_chunk=candidate_reaction_chunk, 
                        closed_chunk_mfcc=reaction_audio_mfcc[:,      round(candidate_reaction_chunk_start / hop_length):round(candidate_reaction_chunk_end / hop_length) ], 
                        closed_chunk_vol_diff=reaction_audio_vol_diff[round(candidate_reaction_chunk_start / hop_length):round(candidate_reaction_chunk_end / hop_length) ],                         
                        current_chunk_size=reverse_chunk_size, 
                        peak_tolerance=peak_tolerance, 
                        open_start=current_start, 
                        closed_start=reaction_start + candidate_segment_start, 
                        distance=first_n_samples)



    if reverse_index and len(reverse_index) > 0:
        reverse_index = reverse_index[0]
    # print('reverse index:', reverse_index / sr)

    if reverse_index and reverse_index > 0: 
        # print(f"Better match for base segment found later in reaction: using filler from base video from {current_start / sr} to {(current_start + reverse_index) / sr} with {(reaction_start - reverse_index) / sr} to {(reaction_start) / sr}")
        
        # seek to a frame boundary
        while (reverse_index - current_start) % samples_per_frame() > 0 and reverse_index < len(reaction_audio) and current_start + reverse_index < len(base_audio):
            reverse_index += 1

        segment = (reaction_start - reverse_index, reaction_start, current_start, current_start + reverse_index, True)

        current_start += reverse_index

        return (segment, current_start, reaction_start, [])

    #########################################

    reaction_start += candidate_segment_start # the first acceptable match in the reaction video
    # print(f"Start segment {current_start / sr}-{(current_start + n_samples) / sr} at {reaction_start / sr}")


    candidate_current_end, candidate_reaction_end, scores = find_correlation_end(current_start, reaction_start, basics, options, step = n_samples, cache={})
    # print(candidate_current_end - current_start, candidate_reaction_end - reaction_start)        


    if candidate_current_end >= len(base_audio): 
        candidate_current_end = len(base_audio) - 1
        candidate_reaction_end = reaction_start + (candidate_current_end - current_start)
        # print("Went past base end, adjusting")
    if candidate_reaction_end >= len(reaction_audio):
        candidate_reaction_end = len(reaction_audio) - 1
        candidate_current_end = current_start + (candidate_reaction_end - reaction_start)
        # print("Went past reaction end, adjusting")

    current_end = candidate_current_end
    reaction_end = candidate_reaction_end

    if reaction_start == reaction_end:
        # print(f"### Sequence of zero length at {reaction_start} {current_start}, skipping forward")
        return (None, current_start, reaction_start + samples_per_frame(), [])

    # print(f"*** Completing match ({reaction_start / sr} [{reaction_start}], {reaction_end / sr} [{reaction_end}]), ({current_start / sr} [{current_start}], {current_end / sr} [{current_end}])\n")                

    assert( is_close(current_end - current_start, reaction_end - reaction_start) )
    segment = (reaction_start, reaction_end, current_start, current_end, False)


    # print("scores", scores)
    # all_scores.append(scores)


    return (segment, current_end, reaction_end, scores)



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

def calculate_partial_score(current_path, checkpoint_ts, basics):

    modified_path = []
    adjusted_reaction_end = None

    for segment in current_path:
        (reaction_start, reaction_end, current_start, current_end, filler) = segment
        if current_end >= checkpoint_ts:
            if current_start > checkpoint_ts:
                break

            to_trim = current_end - checkpoint_ts
            current_end = checkpoint_ts
            reaction_end -= to_trim
            modified_path.append( (reaction_start, reaction_end, current_start, current_end, filler) )
            adjusted_reaction_end = reaction_end
            break
        else:
            modified_path.append(segment)
            adjusted_reaction_end = reaction_end

    if adjusted_reaction_end is None: 
        print(f"WEIRD! Could not calculate partial score for {checkpoint_ts} {current_path}")
        return None

    # truncate current path back to current_ts
    score = path_score(modified_path, basics, relative_to = checkpoint_ts) 

    return adjusted_reaction_end, score


def check_if_prune_at_nearest_checkpoint(current_path, current_path_checkpoint_scores, paths_by_checkpoint, best_finished_path, current_start, basics):
    base_audio = basics.get('base_audio')
    sr = basics.get('sr')
    new_checkpoint_every_n_prunes = 5
    checkpoints = basics.get('checkpoints')
    len_audio = len(base_audio)

    if len(current_path) == 0:
        print("Weird zero length path!")
        return 'checkpoint'


    # go through all checkpoints we've passed
    for its, current_ts in enumerate(checkpoints):
        if current_start < current_ts: 
            break

        if current_ts in current_path_checkpoint_scores and current_path_checkpoint_scores[current_ts]:
            continue

        partial_score = calculate_partial_score(current_path, current_ts, basics)
        if partial_score is None:
            return 'checkpoint'

        adjusted_reaction_end, current_score = partial_score

        current_path_checkpoint_scores[current_ts] = True #[current_ts, current_score, adjusted_reaction_end]

        if current_ts not in paths_by_checkpoint:
            paths_by_checkpoint[current_ts] = {'prunes_here': 0, 'paths': []}

        prunes_here = paths_by_checkpoint[current_ts]['prunes_here']


        if 'score' in best_finished_path:
            if current_ts not in best_finished_path['partials']:
                _, best_at_checkpoint = calculate_partial_score(best_finished_path['path'], current_ts, basics)
                best_finished_path['partials'][current_ts] = best_at_checkpoint
            best_at_checkpoint = best_finished_path['partials'][current_ts]

            if current_ts > len_audio / 2:
                confidence = .99
            else: 
                confidence = .5 + .499 * (  current_ts / (len_audio / 2) )

            if random.random() < .001:
                print(f"Best score is {best_finished_path['score']}, at {current_ts / sr:.1f} comparing:")
                print(f"{best_at_checkpoint}")
                print(f"{current_score}")
                print(f"({100 * current_score[2] * current_score[3] * current_score[3] / (best_at_checkpoint[2] * best_at_checkpoint[3] * best_at_checkpoint[3]):.1f}%) ")

            if confidence * best_at_checkpoint[2] * best_at_checkpoint[3] * best_at_checkpoint[3] > current_score[2] * current_score[3] * current_score[3]: 
                paths_by_checkpoint[current_ts]['prunes_here'] += 1
                return 'best_score'

        ts_thresh_contrib = min( current_ts / (3 * 60 * sr), .1)
        prunes_thresh_contrib = min( .04 * prunes_here / 50, .04 )

        prune_threshold = .85 + ts_thresh_contrib + prunes_thresh_contrib


        full_comp_score = None
        for comp_reaction_end, comp_score, ppath in paths_by_checkpoint[current_ts]['paths']:
            # print(f"\t{comp_reaction_end} <= {adjusted_reaction_end}?")
            if comp_reaction_end <= adjusted_reaction_end:
                # (cs1,cs2,cs3) = comp_score
                # (s1,s2,s3) = current_score

                # m1 = max(cs1,s1); m2 = max(cs2,s2); m3 = max(cs3,s3)

                # full_comp_score = cs1 / m1 + cs2 / m2 + cs3 / m3
                # full_score      =  s1 / m1 +  s2 / m2 +  s3 / m3
                # print(f"\t\t{full_score} < {.9 * full_comp_score}?")

                if current_score[0] < prune_threshold * comp_score[0]:
                    paths_by_checkpoint[current_ts]['prunes_here'] += 1 # increment prunes at this checkpoint
                    # print(f"\tCheckpoint Prune at {current_ts / sr}: {current_score[0]} compared to {comp_score[0]}. Prunes here: {paths_by_checkpoint[current_ts]['prunes_here']} @ thresh {prune_threshold}")
                    
                    return 'checkpoint'

        paths_by_checkpoint[current_ts]['paths'].append( (adjusted_reaction_end, current_score, list(current_path)) )


        # print("no prune", paths_by_checkpoint[current_ts])

            
    if random.random() < .01:
        prune_path_prunes(paths_by_checkpoint)

    return False

def prune_path_prunes(paths_by_checkpoint):
    # print("##############")
    # print("Cleaning out prune paths")
    # print('##############')

    for ts, prune_data in paths_by_checkpoint.items():
        paths = prune_data["paths"]
        new_paths = []

        paths.sort(key=lambda x: x[0])

        original_path_length = path_length = len(paths)
        new_path_length = -1

        while path_length != new_path_length:
            path_length = len(paths)
            paths[:] = [path for i,path in enumerate(paths) if i == 0 or path[1] > paths[i-1][1]]
            new_path_length = len(paths)

        # print(f"\t{ts}: from {original_path_length} to {new_path_length}")




prune_types = {}


profile_pathways = False
if profile_pathways:
    profiler = cProfile.Profile()

def find_pathways(basics, options, current_path=None, current_path_checkpoint_scores=None, paths_by_checkpoint=None, paths_from_current_start=None, current_start=0, reaction_start=0, segment_scope_path=None, path_counts=None, best_finished_path=None): 
    
    # initializing
    if current_path is None: 
        current_path = []
        current_path_checkpoint_scores = {}
        paths_by_checkpoint = {}
        paths_from_current_start = {}
        segment_scope_path = {}
        path_counts = {-1: {'open': 0, 'completed': 0}}
        best_finished_path = {}


    if profile_pathways:
        global profiler

    base_audio = basics.get('base_audio')
    reaction_audio = basics.get('reaction_audio')
    sr = basics.get('sr')
    base_audio_mfcc = basics.get('base_audio_mfcc')
    base_audio_vol_diff = basics.get('song_percentile_loudness')
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    reaction_audio_vol_diff = basics.get('reaction_percentile_loudness')
    hop_length = basics.get('hop_length')
    checkpoints = basics.get('checkpoints')

    step = first_n_samples = options.get('first_n_samples')

    depth = len(current_path)

    if depth == 0 and profile_pathways:
        profiler.enable()


    if depth not in path_counts:
        path_counts[depth] = {
            'open': 0,
            'completed': 0,
            'current_starts': {}
        }

    max_visited = 0 
    for ddd, cnt in path_counts.items():
        if cnt['open'] > max_visited:
            max_visited = cnt['open']

    total_visited = path_counts[-1]['open']



    if current_start not in path_counts[depth]['current_starts']:
        path_counts[depth]['current_starts'][current_start] = 0
    path_counts[depth]['current_starts'][current_start] += 1

    # if random.random() < .001:
    #     depths = list(path_counts.keys())
    #     depths.sort()
    #     print("***********************")
    #     print("Current_starts by depth")
    #     for ddepth in depths:
    #         print(f"\t{ddepth}:")
    #         starts = list(path_counts[ddepth]['current_starts'].keys())
    #         starts.sort()
    #         for sstart in starts:
    #             cnt = path_counts[ddepth]['current_starts'][sstart]
    #             print(f"\t\t{sstart} [{sstart / sr:.1f}]: {path_counts[ddepth]['current_starts'][sstart]}")

 

    
    if depth > 3:
        prior_current_start = current_path[-1][2]
        prior_prior_current_start = current_path[-2][2]
        if prior_current_start in path_counts[depth - 1]['current_starts'] and path_counts[depth - 1]['current_starts'][prior_current_start] > 1:  # path_counts[depth]['completed'] > 250:
            check_depth = depth - 1
            while check_depth > 3 and depth - check_depth < 10:
                new_checkpoint = current_path[check_depth - depth][2]
                add_new_checkpoint(checkpoints, new_checkpoint, paths_by_checkpoint, basics)
                check_depth -= 1


        if prior_prior_current_start in path_counts[depth - 2]['current_starts'] and max_visited > 2000 and path_counts[depth - 2]['current_starts'][prior_prior_current_start] > 5:  # path_counts[depth]['completed'] > 250:
            prune_types["combinatorial"] += 1
            return [None]

    # aggressive prune based on scores after having passed a checkpoint 
    if depth > 0:
        should_prune = check_if_prune_at_nearest_checkpoint(current_path, current_path_checkpoint_scores, paths_by_checkpoint, best_finished_path, current_start, basics)
        if should_prune and max_visited > 100000:
            prune_types[should_prune] += 1

            if random.random() < .0001:
                for k,v in prune_types.items():
                    print(f"\t{k}: {v}")

                for tss in basics.get('checkpoints'):
                    if tss in paths_by_checkpoint:
                        prunes = paths_by_checkpoint[tss]['prunes_here']
                    else:
                        prunes = "<nil>"
                    print(f"\t\t{tss / sr}: {prunes}")

            return [None]

    # specific prune based on exact match of current start
    if depth > 0:
        score = None
        if current_start not in paths_from_current_start: 
            paths_from_current_start[current_start] = []
        else: 
            score = path_score(current_path, basics, relative_to = current_start) 
            for i, (comp_reaction_start, comp_path, comp_score) in enumerate(paths_from_current_start[current_start]):
                if comp_reaction_start <= reaction_start:
                    if comp_score is None:
                        comp_score = path_score(comp_path, basics, relative_to = current_start)
                        paths_from_current_start[current_start][i][2] = comp_score


                    # (cs1,cs2,cs3) = comp_score
                    # (s1,s2,s3) = score

                    # m1 = max(cs1,s1); m2 = max(cs2,s2); m3 = max(cs3,s3)

                    # full_comp_score = cs1 / m1 + cs2 / m2 + cs3 / m3
                    # full_score      =  s1 / m1 +  s2 / m2 +  s3 / m3

                    ts_thresh_contrib = min( current_start / (2 * 60 * sr), .09)
                    prune_threshold = .9 + ts_thresh_contrib

                    if score[0] < prune_threshold * comp_score[0] and max_visited > 1000000:
                        print(f"\tExact Prune! {comp_reaction_start} {comp_score[0]}  >  {reaction_start} {score[0]} @ threshold {prune_threshold}")
                        prune_types['exact'] += 1
                        return [None]      
                    # else: 
                    #     print(f"    unpruned... {comp_reaction_start} {full_comp_score}  ~  {reaction_start} {full_score}")      
        paths_from_current_start[current_start].append( [reaction_start, copy.deepcopy(current_path), score] )


    if depth > 100:
        print(f"\tPath Length Prune!")
        prune_types['length'] += 1
        return [None]

    alignment_bounds = options.get('alignment_bounds')
    if alignment_bounds is not None:
        upper_bound = get_bound(alignment_bounds, current_start, len(reaction_audio))
        if not in_bounds(upper_bound, current_start, reaction_start):
            # print(f'\tBounds Prune! {current_start / sr} {reaction_start / sr} not in bounds (upper_bound={upper_bound}!')
            prune_types['bounds'] += 1
            return [None]
    else:
        upper_bound = None






    # print(f"Starting pathway {current_path} {current_start} {reaction_start}")
    try: 
        # Get the base audio chunk
        current_end = min(current_start + step, len(base_audio))
        chunk = base_audio[current_start:current_end]
        current_chunk_size = current_end - current_start


        # Finished!
        matched_all_of_base = current_start >= len(base_audio) - 1
        reaction_audio_finished = reaction_start >= len(reaction_audio) - 1 or current_start > len(base_audio) - int(.5 * sr)
        if matched_all_of_base or reaction_audio_finished:

            # Handle case where reaction video finishes before base video
            if reaction_audio_finished:
                length_remaining = len(base_audio) - current_start - 1
                filler_segment = (reaction_start, reaction_start + length_remaining, current_start, current_start + length_remaining, True)
                current_path.append(  filler_segment   )
                # print(f"Reaction video finished before end of base video. Backfilling with base video ({length_remaining / sr}s).")
                # print(f"Backfilling {current_path} {current_start} {reaction_start}")

            score = path_score(current_path, basics)
            if 'score' not in best_finished_path or best_finished_path['score'][0] < score[0]:
                best_finished_path["path"] = current_path
                best_finished_path["score"] = score
                best_finished_path["partials"] = {}
                print(f"**** New best score is {best_finished_path['score']}")


            return [current_path]




        # print(f'\nFinding segment start  reaction_start={reaction_start} (of {len(reaction_audio)})  current_start={current_start} (of {len(base_audio)}) upper_bound={upper_bound}')
        # candidate_paths = find_next_best_segment_start(reaction_audio[reaction_start:], chunk, current_chunk_size, peak_tolerance, reaction_start, current_start, sr, distance=first_n_samples)

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
                                prune_counts=prune_types,
                                upper_bound=upper_bound, 
                                filter_for_similarity=depth > 0, 
                                print_candidates=depth==0  )

        # print(f'\tCandidate increments: {candidate_starts}')


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
                path_counts[depth]['open'] += 1
                path_counts[-1]['open'] += 1
                result = find_pathways(basics, options, current_path, copy.deepcopy(current_path_checkpoint_scores), paths_by_checkpoint, paths_from_current_start, current_start, reaction_start, segment_scope_path, path_counts, best_finished_path)
                path_counts[depth]['completed'] += 1
                path_counts[-1]['completed'] += 1
                return result

        # candidate_starts = [candidate_starts] # temporary as we test this approach with just a single branch in the tree

        paths = []
        path_counts[depth]['open'] += len(candidate_starts)
        path_counts[-1]['open'] += len(candidate_starts)

        for ci, candidate_segment_start in enumerate(candidate_starts):
            if depth == 0:
                print(f"STARTING DEPTH ZERO from {candidate_segment_start / sr}")

            paths_this_direction = []

            scope_key = f'({current_start}, {reaction_start + candidate_segment_start}, {current_chunk_size})'
            if scope_key not in segment_scope_path:
                segment_scope_path[scope_key] = scope_segment(basics, current_start, reaction_start, candidate_segment_start, current_chunk_size, options)
            else: 
                prune_types['scope_cached'] += 1

            segment, next_start, next_reaction_start, scores = segment_scope_path[scope_key]

            my_path = copy.deepcopy(current_path)
            if segment:
                my_path.append(segment)
            elif next_reaction_start < len(reaction_audio) - 1: 
                print("did not find segment", next_start, next_reaction_start)


            if next_reaction_start >= len(reaction_audio) - 1:
                # end this path
                paths_this_direction.append(my_path)
            else:
                continued_paths = find_pathways(basics, options, my_path, copy.deepcopy(current_path_checkpoint_scores), paths_by_checkpoint, paths_from_current_start, next_start, next_reaction_start, segment_scope_path, path_counts, best_finished_path)
                for full_path in continued_paths:
                    if full_path is not None:
                        paths_this_direction.append(full_path)

            path_counts[depth]['completed'] += 1
            path_counts[-1]['completed'] += 1

            if depth == 0:
                print("**************")
                print(f"Len of paths from {candidate_segment_start / sr} is {len(paths_this_direction)}")

                # profiler.disable()
                # stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
                # stats.print_stats()


            # Don't add duplicate paths
            for new_path in paths_this_direction:
                duplicate = False
                for other_path in paths:
                    if other_path == new_path:
                        print('SKIPPING DUPLICATE PATH')
                        duplicate = True
                        break
                if not duplicate:
                    # if depth == 0:
                    #     print(f"Adding path {new_path}")
                    paths.append(new_path)




            # print(f"Got paths: {current_path} {current_start} {reaction_start} {paths}")


            if depth < 2 or random.random() < .0001:
                print(f"Paths at point: {current_start / sr:.1f} {reaction_start / sr:.1f} [depth={depth}] [paths found here={len(paths)}]")

                try: 
                    for ddd, progress in path_counts.items():
                        if progress['open'] > 0:
                            print(f"\tDepth {ddd}: [{progress['completed']} / {progress['open']} = {100 * progress['completed'] / progress['open']:.1f}%] [remaining={progress['open'] - progress['completed']}]")
                        else:
                            print(f"\tDepth {ddd}: [{progress['completed']} / {progress['open']}")

                except: 
                    print("Could not print")

            if depth < 2: 
                for k,v in prune_types.items():
                    print(f"\t{k}: {v}")


                for tss in basics.get('checkpoints'):
                    if tss in paths_by_checkpoint:
                        prunes = paths_by_checkpoint[tss]['prunes_here']
                    else:
                        prunes = "<nil>"
                    print(f"\t\t{tss / sr}: {prunes}")

            if depth == 0: 
                print("*************\nCLEARING PATH COUNTS")
                path_counts = {-1: {'open': 0, 'completed': 0}, 0: {'open': 0, 'completed': 0}} # give each starting point a fair shot



            if profile_pathways and path_counts[depth]['completed'] % 1000 == 50:
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




def create_reaction_mfcc_from_path(path, basics):

    # these mfcc variables are the result of calls to librosa.feature.mfcc, thus
    # they have a shape of (num_mfcc, length of audio track). The length of the 
    # reaction_audio_mfcc track is greater than the length of base_audio_mfcc track. 
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    base_audio_mfcc = basics.get('base_audio_mfcc')
    hop_length = basics.get('hop_length')

    total_length = 0
    for reaction_start, reaction_end, current_start, current_end, is_filler in path:
        if not is_filler:
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)
            total_length += reaction_end - reaction_start
        else:
            current_start = round(current_start / hop_length)
            current_end = round(current_end / hop_length)
            total_length += current_end - current_start



    # the resulting combined mfcc (path_mfcc) should have the same number of mfccs 
    # as the input mfccs, and be the length of the reaction_audio_for_path. 
    path_mfcc = np.zeros((base_audio_mfcc.shape[0], total_length))


    # Now we're going to fill up the path_mfcc incrementally, taking from either
    # base_audio_mfcc or reaction_audio_mfcc
    start = 0
    for reaction_start, reaction_end, current_start, current_end, is_filler in path:

        if not is_filler:
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)

            length = reaction_end - reaction_start
            segment = reaction_audio_mfcc[:,reaction_start:reaction_end]
        else:
            current_start = round(current_start / hop_length)
            current_end = round(current_end / hop_length)

            length = current_end - current_start
            segment = base_audio_mfcc[:,current_start:current_end]

        if length > 0:
            path_mfcc[:,start:start+length] = segment
            start += length

    return path_mfcc

def create_reaction_vol_diff_from_path(path, basics):

    # these mfcc variables are the result of calls to librosa.feature.mfcc, thus
    # they have a shape of (num_mfcc, length of audio track). The length of the 
    # reaction_audio_mfcc track is greater than the length of base_audio_mfcc track. 
    reaction_audio_mfcc = basics.get('reaction_percentile_loudness')
    base_audio_mfcc = basics.get('song_percentile_loudness')
    hop_length = basics.get('hop_length')

    total_length = 0
    for reaction_start, reaction_end, current_start, current_end, is_filler in path:
        if not is_filler:
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)
            total_length += reaction_end - reaction_start
        else:
            current_start = round(current_start / hop_length)
            current_end = round(current_end / hop_length)
            total_length += current_end - current_start



    # the resulting combined mfcc (path_mfcc) should have the same number of mfccs 
    # as the input mfccs, and be the length of the reaction_audio_for_path. 
    path_mfcc = np.zeros(total_length)


    # Now we're going to fill up the path_mfcc incrementally, taking from either
    # base_audio_mfcc or reaction_audio_mfcc
    start = 0
    for reaction_start, reaction_end, current_start, current_end, is_filler in path:


        if not is_filler:
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)

            length = reaction_end - reaction_start
            segment = reaction_audio_mfcc[reaction_start:reaction_end]
        else:
            current_start = round(current_start / hop_length)
            current_end = round(current_end / hop_length)

            length = current_end - current_start
            segment = base_audio_mfcc[current_start:current_end]

        if length > 0:
            path_mfcc[start:start+length] = segment
            start += length

    return path_mfcc




path_score_cache = {}
path_score_cache_perf = {}
def path_score(path, basics, relative_to=None): 
    global path_score_cache
    global path_score_cache_perf

    base_audio = basics.get('base_audio')
    base_audio_mfcc = basics.get('base_audio_mfcc')
    base_audio_vol_diff = basics.get('song_percentile_loudness')
    hop_length = basics.get('hop_length')
    reaction_audio = basics.get('reaction_audio')
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    sr = basics.get('sr')

    key = f"{len(base_audio)} {len(reaction_audio)} {str(path)} {relative_to}"
    if key in path_score_cache:
        path_score_cache_perf['hits'] += 1
        return path_score_cache[key]

    path_score_cache_perf['misses'] += 1

    # print(f"path score cache hits/misses = {path_score_cache_perf['hits']} / {path_score_cache_perf['misses']}")

    if relative_to is None:
        relative_to = len(base_audio)



    duration = 0
    fill = 0

    temporal_center = 0
    total_length = 0

    for reaction_start, reaction_end, current_start, current_end, is_filler in path:
        if reaction_start < 0:
            reaction_end += -1 * reaction_start
            reaction_start = 0
        if current_start < 0:
            current_end += -1 * current_start
            current_start = 0

        if not is_filler:
            duration += (reaction_end - reaction_start)
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)
            total_length += reaction_end - reaction_start            

        else:
            fill += current_end - current_start
            current_start = round(current_start / hop_length)
            current_end = round(current_end / hop_length)
            total_length += current_end - current_start


        segment_weight = (reaction_end - reaction_start) / relative_to
        segment_time_center = (reaction_start + (reaction_end - reaction_start) / 2)
        temporal_center += segment_weight * segment_time_center

    # Derivation for below:
    #   earliness = |R| / temporal_center
    #   best possible earliness =  |R| / (|B| / 2) (when the first part of the reaction is matched with the full base audio)
    #   normalized earliness = earliness / best_possible_earliness = |B| / (|R| * temporal_center)
    #   ...but I'm going to change the normalization to match being in the middle of the reaction, because sometimes
    #      the reactions are really really long.
    #   middle_earliness = |R| / ( |R| / 2  ) = 2
    #   normalized earliness = earliness / middle_earliness
    # normalized_earliness_score = len(base_audio) / (len(reaction_audio) * temporal_center)
    # fill_score = 1 / (1 + abs(duration - len(base_audio)) / sr)

    earliness = len(reaction_audio) / temporal_center
    earliness = math.log(1 + earliness)
    fill_score = duration / (duration + fill)


    path_mfcc = create_reaction_mfcc_from_path(path, basics)
    path_vol_diff = create_reaction_vol_diff_from_path(path, basics)

    mfcc_alignment = mfcc_similarity(sr, mfcc1=base_audio_mfcc[:,:total_length], mfcc2=path_mfcc)
    rel_vol_alignment = relative_volume_similarity(sr, vol_diff1=base_audio_vol_diff[:total_length], vol_diff2=path_vol_diff)

    alignment = math.log(1 + 100 * mfcc_alignment) * math.log(1 + rel_vol_alignment)

    duration_score = (duration + fill) / relative_to

    total_score = duration_score * duration_score * fill_score * fill_score * earliness * alignment
    if duration == 0:
        total_score = alignment = 0

    path_score_cache[key] = [total_score, earliness, alignment, fill_score]
    return path_score_cache[key]


def find_best_path(candidate_paths, basics):
    sr = basics.get('sr')

    assert( len(candidate_paths) > 0 )
    
    paths_with_scores = []

    for path in candidate_paths:
        paths_with_scores.append([path, path_score(path, basics)])

    best_score = 0 
    best_early_completion_score = 0
    best_similarity = 0 
    best_duration = 0
    for path, scores in paths_with_scores:
        if scores[0] > best_score:
            best_score = scores[0]
        if scores[1] > best_early_completion_score:
            best_early_completion_score = scores[1]
        if scores[2] > best_similarity:
            best_similarity = scores[2]
        if scores[3] > best_duration:
            best_duration = scores[3]


    for path, scores in paths_with_scores:
        total_score = scores[0] / best_score
        completion_score = scores[1] / best_early_completion_score
        similarity_score = scores[2] / best_similarity
        fill_score = scores[3] / best_duration

        scores[0] = total_score
        scores[1] = completion_score
        scores[2] = similarity_score
        scores[3] = fill_score

        # score = (fill_score + completion_score + similarity_score) / 3
        # scores.insert(0, score)

    paths_by_score = sorted(paths_with_scores, key=lambda x: x[1][0], reverse=True)

    print("Paths by score:")
    for path,scores in paths_by_score:
        print(f"\tScore={scores[0]}  EarlyThrough={scores[1]}  Similarity={scores[2]} Duration={scores[3]}")
        for sequence in path:
            print(f"\t\t{'*' if sequence[4] else ''}base: {float(sequence[2])/sr:.1f}-{float(sequence[3])/sr:.1f}  reaction: {float(sequence[0])/sr:.1f}-{float(sequence[1])/sr:.1f} ")


    return paths_by_score[0][0]

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

    basics["checkpoints"] = get_initial_checkpoints(basics)

    options['n_samples'] = n_samples
    options['first_n_samples'] = first_n_samples

    options['alignment_bounds'] = create_reaction_alignment_bounds(basics, first_n_samples)


    global path_score_cache
    global path_score_cache_perf
    global seg_start_cache
    global seg_start_cache_effectiveness

    path_score_cache.clear()
    path_score_cache_perf["hits"] = 0
    path_score_cache_perf["misses"] = 0

    seg_start_cache.clear()
    seg_start_cache_effectiveness["hits"] = 0
    seg_start_cache_effectiveness["misses"] = 0


    global prune_types
    prunes = {
        "checkpoint": 0,
        "best_score": 0,
        "exact": 0,
        "bounds": 0,
        "length": 0,
        "cached": 0,
        "scope_cached": 0,
        "combinatorial": 0,
        "continuity": 0
    }
    for prune_type in prunes:
        prune_types[prune_type] = 0



    paths = find_pathways(basics, options)

    # print('FOUND PATHS!', paths)


    path = find_best_path(paths, basics)

    # scores will now be stored associated with each path
    # plot_curves(all_scores)

    # print("segs", match_segments)

    # for current_start, current_end, current_base_start, current_base_end, filler in match_segments:
    #     print((current_end - current_start) / sr, (current_base_end - current_base_start) / sr, filler)


    return path


def plot_curves(curves):
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Set the colors you want to cycle through
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # For each curve
    for i, curve in enumerate(curves):
        # Select the color for this curve
        color = colors[i % len(colors)]

        times = [end / 44100 for _,end,_ in curve]
        scores = [score for _,_,score in curve]
            
        # Plot the curve in the selected color
        plt.plot(times, scores, color=color)

    # Set labels
    plt.xlabel('Time')
    plt.ylabel('Score')

    # Display the plot
    plt.show()


