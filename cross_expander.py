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

from utilities import samples_per_frame, universal_frame_rate, is_close, create_audio_extraction
from face_finder import detect_faces, create_reactor_view
from backchannel_isolator import process_reactor_audio

from decimal import Decimal, getcontext



def correct_peak_index(peak_index, chunk_len):
    # return max(0,peak_index)
    return max(0, peak_index - (chunk_len - 1))


def mfcc_similarity(audio_chunk1, audio_chunk2, sr, mfcc1=None, mfcc2=None):

    # Compute MFCCs for each audio chunk
    if mfcc1 is None: 
        mfcc1 = librosa.feature.mfcc(y=audio_chunk1, sr=sr, n_mfcc=20)

    if mfcc2 is None: 
        mfcc2 = librosa.feature.mfcc(y=audio_chunk2, sr=sr, n_mfcc=20)


    # Make sure the MFCCs are the same shape
    len1 = mfcc1.shape[1]
    len2 = mfcc2.shape[1]
    max_len = max(len1, len2)
    if len1 < max_len:
        padding = max_len - len1
        mfcc1 = np.pad(mfcc1, pad_width=((0, 0), (0, padding)), mode='constant')
    elif len2 < max_len:
        padding = max_len - len2
        mfcc2 = np.pad(mfcc2, pad_width=((0, 0), (0, padding)), mode='constant')

    # Compute mean squared error between MFCCs
    mse = np.mean((mfcc1 - mfcc2)**2)

    similarity = 1 / (1 + mse)
    return 10000 * similarity





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

def create_reaction_alignment_bounds(base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, sr, first_n_samples, n_timestamps = 6, peak_tolerance=.6):
    clip_length = int(1.5 * sr)
    base_length_sec = len(base_audio) / sr  # Length of the base audio in seconds
    
    timestamps = [i * base_length_sec / (n_timestamps + 1) for i in range(1, n_timestamps + 1)]
    timestamps_samples = [int(t * sr) for t in timestamps]
    
    # Initialize the list of bounds
    bounds = []

    print(f"Creating alignment bounds at {timestamps}")
    
    # For each timestamp
    for i,ts in enumerate(timestamps_samples):
        # Define the segments on either side of the timestamp
        segments = [base_audio[max(0, ts - clip_length):ts], base_audio[ts:min(len(base_audio), ts + clip_length)]]

        # for j, segment in enumerate(segments):
        #     filename = f"segment_{i}_{j}.wav"
        #     sf.write(filename, segment, sr)
        
        # Initialize the list of max indices for the segments
        max_indices = []
        
        print(f"ts: {ts / sr}")
        # For each segment
        for chunk in segments:
            # Find the candidate indices for the start of the matching segment in the reaction audio
            candidates = find_next_segment_start_candidates(reaction_audio[ts:], chunk, clip_length, peak_tolerance, ts, ts, sr, distance=first_n_samples)

            print(f"\tCandidates: {candidates}  {max(candidates)}")

            # Find the maximum candidate index
            max_indices.append(ts + max(candidates))
        
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
    return alignment_bounds


def in_bounds(bound, base_start, reaction_start):
    return reaction_start <= bound

def get_bound(alignment_bounds, base_start, reaction_end): 
    for base_ts, last_reaction_match in alignment_bounds:
        if base_start < base_ts:
            return last_reaction_match
    return reaction_end







def find_correlation_end(current_start, reaction_start, base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, sr, expansion_tolerance, step, scores = [], reaction_end=None, end_at=None, max_score=0, backoff=0, cache={}):
    hop_length = 512

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
            chunk = base_audio[current_start:current_end]
            reaction_chunk = reaction_audio[reaction_start:reaction_end]
            chunk_score = mfcc_similarity(reaction_chunk, chunk, sr, mfcc1=reaction_audio_mfcc[:, round(reaction_start / hop_length):round(reaction_end / hop_length)], mfcc2=base_audio_mfcc[:, round(current_start / hop_length):round(current_end / hop_length)])
            cache[key] = chunk_score

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

        return find_correlation_end(current_start, reaction_start, base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, sr, expansion_tolerance, step = step // 2, scores = scores, reaction_end = break_point, end_at = next_end_at, max_score = max_score, backoff = backoff, cache = cache )
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

def find_next_segment_start_candidates(open_chunk, closed_chunk, current_chunk_size, peak_tolerance, open_start, closed_start, sr, distance, upper_bound=None):

    if upper_bound is not None:
        prev = len(open_chunk)
        open_chunk = open_chunk[:int(upper_bound - open_start + 2 * current_chunk_size)]
        print(f"\tConstraining open chunk from size {prev} to {len(open_chunk)}  [{len(open_chunk) / prev * 100}% of original]")

    # Perform cross correlation
    correlation = correlate(open_chunk, closed_chunk)

    # Find peaks
    peak_indices, _ = find_peaks(correlation, height=np.max(correlation)*peak_tolerance, distance=distance)
    peak_indices = sorted( peak_indices.tolist() )

    if len(peak_indices) == 0:
        print(f"No peaks found for {closed_start} [{closed_start / sr} sec] / {open_start} [{open_start / sr} sec] {np.max(correlation)}")
        return None

    # first, find the correct cluster of matches, then seek to the best starting point
    # best_index = correct_peak_index(peak_indices[0], current_chunk_size)

    assert( len(closed_chunk) == current_chunk_size)

    scores = []
    max_score = 0
    max_index = None
    for candidate in peak_indices:
        candidate_index = correct_peak_index(candidate, current_chunk_size) 

        if upper_bound is not None and not math.isinf(upper_bound) and upper_bound < candidate_index + open_start:
            continue

        chunk_score = mfcc_similarity(open_chunk[candidate_index:candidate_index + current_chunk_size], closed_chunk, sr)    
        scores.append( (candidate_index, chunk_score) ) 
        if chunk_score > max_score:
            max_score = chunk_score
            max_index = candidate_index
        # print(f"Comparing start at {(closed_start + candidate_index) / sr} with score {correlation[candidate]}  [{chunk_score}]")

    candidates = [ candidate_index for candidate_index, chunk_score in scores if chunk_score >= max_score * peak_tolerance ]
    return candidates

    first = candidates[0]
    if first == max_index:
        return [first]
    else:
        return [first, max_index]


def find_next_best_segment_start(open_chunk, closed_chunk, current_chunk_size, peak_tolerance, closed_start, open_start, sr, distance):
    candidates = find_next_segment_start_candidates(open_chunk, closed_chunk, current_chunk_size, peak_tolerance, open_start, closed_start, sr, distance)
    if not candidates: 
      return None
    return candidates[0]
    



#####################################################################
# Given a starting index into the reaction, find a good ending index
def scope_segment(base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, current_start, reaction_start, candidate_segment_start, current_chunk_size, n_samples, first_n_samples, reverse_search_bound, peak_tolerance, expansion_tolerance, segment_end_backoff, sr): 


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
    candidate_reaction_chunk = reaction_audio[reaction_start+candidate_segment_start:reaction_start+candidate_segment_start+reverse_chunk_size]
    if reverse_chunk_size > len(candidate_reaction_chunk):
        reverse_chunk_size = len(candidate_reaction_chunk)
        open_end = min(open_end, current_start + reverse_chunk_size)


    open_base_chunk = base_audio[current_start:open_end]

    # print(f'\nDoing reverse index search  reaction_start={reaction_start+candidate_segment_start}  current_start={current_start}  {reverse_chunk_size} {len(candidate_reaction_chunk)} {len(open_base_chunk)}')
    reverse_index = find_next_best_segment_start(open_base_chunk, candidate_reaction_chunk, reverse_chunk_size, peak_tolerance, current_start, reaction_start + candidate_segment_start, sr, distance=first_n_samples)
    # print('reverse index:', reverse_index / sr)

    if reverse_index > 0: 
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


    candidate_current_end, candidate_reaction_end, scores = find_correlation_end(current_start, reaction_start, base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, sr, expansion_tolerance, step = n_samples, backoff = segment_end_backoff, cache={})
    # print(candidate_current_end - current_start, candidate_reaction_end - reaction_start)        


    if candidate_current_end >= len(base_audio): 
        candidate_current_end = len(base_audio) - 1
        candidate_reaction_end = reaction_start + (candidate_current_end - current_start)
        print("Went past base end, adjusting")
    if candidate_reaction_end >= len(reaction_audio):
        candidate_reaction_end = len(reaction_audio) - 1
        candidate_current_end = current_start + (candidate_reaction_end - reaction_start)
        print("Went past reaction end, adjusting")

    current_end = candidate_current_end
    reaction_end = candidate_reaction_end

    if reaction_start == reaction_end:
        print(f"### Sequence of zero length at {reaction_start} {current_start}, skipping forward")
        return (None, current_start, reaction_start + samples_per_frame(), [])

    print(f"*** Completing match ({reaction_start / sr} [{reaction_start}], {reaction_end / sr} [{reaction_end}]), ({current_start / sr} [{current_start}], {current_end / sr} [{current_end}])\n")                

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
def find_pathways(current_path, base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, current_start, reaction_start, step, n_samples, first_n_samples, reverse_search_bound, peak_tolerance, expansion_tolerance, segment_end_backoff, sr, path_cache, path_counts, alignment_bounds=None): 

    if alignment_bounds is not None:
        upper_bound = get_bound(alignment_bounds, current_start, len(reaction_audio))
        if not in_bounds(upper_bound, current_start, reaction_start):
            print(f'Pruning path! {current_start / sr} {reaction_start / sr} not in bounds (upper_bound={upper_bound}!')
            return [None]
    else:
        upper_bound = None

    # print(f"Starting pathway {current_path} {current_start} {reaction_start}")
    try: 
        # Get the base audio chunk
        current_end = min(current_start + step, len(base_audio))
        chunk = base_audio[current_start:current_end]
        current_chunk_size = current_end - current_start

        if current_start >= len(base_audio) - 1:
            length_remaining = len(base_audio) - current_start - 1
            if length_remaining > 0:
                filler_segment = (reaction_start, reaction_start + length_remaining, current_start, current_start + length_remaining, True)
                current_path.append(  filler_segment   )

            return [current_path]


        #################
        # Handle case where reaction video finishes before base video
        if reaction_start >= len(reaction_audio) - 1 or current_start > len(base_audio) - int(.5 * sr):
            length_remaining = len(base_audio) - current_start - 1
            filler_segment = (reaction_start, reaction_start + length_remaining, current_start, current_start + length_remaining, True)
            current_path.append(  filler_segment   )
            print(f"Reaction video finished before end of base video. Backfilling with base video ({length_remaining / sr}s).")

            print(f"Backfilling {current_path} {current_start} {reaction_start}")

            return [current_path]
        ########


        print(f'\nFinding segment start  reaction_start={reaction_start} (of {len(reaction_audio)})  current_start={current_start} (of {len(base_audio)}) upper_bound={upper_bound}')
        # candidate_paths = find_next_best_segment_start(reaction_audio[reaction_start:], chunk, current_chunk_size, peak_tolerance, reaction_start, current_start, sr, distance=first_n_samples)
        candidate_starts = find_next_segment_start_candidates(reaction_audio[reaction_start:], chunk, current_chunk_size, peak_tolerance, reaction_start, current_start, sr, distance=first_n_samples, upper_bound=upper_bound)
        # print(f'\tCandidate increments: {candidate_starts}')

        if candidate_starts is None:
            current_start += 1
            if (len(base_audio) - step - current_start) / sr < .5:
                print(f"Could not find match in remainder of reaction video!!! Stopping.")
                print(f"No match in remainder {current_path} {current_start} {reaction_start}")

                length_remaining = len(base_audio) - current_start - 1
                if length_remaining > 0:
                    filler_segment = (reaction_start, reaction_start + length_remaining, current_start, current_start + length_remaining, True)
                    current_path.append(  filler_segment   )

                return [current_path]
            else:
                print(f"Could not find match in remainder of reaction video!!! Skipping sample.")
                path_counts['open'] += 1
                result = find_pathways(current_path, base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, current_start, reaction_start, first_n_samples, n_samples, first_n_samples, reverse_search_bound, peak_tolerance, expansion_tolerance, segment_end_backoff, sr, path_cache, path_counts, alignment_bounds)
                path_counts['completed'] += 1
                return result

        # candidate_starts = [candidate_starts] # temporary as we test this approach with just a single branch in the tree

        paths = []
        path_counts['open'] += len(candidate_starts)

        for candidate_segment_start in candidate_starts:

            key = f'({current_start}, {reaction_start + candidate_segment_start})'
            if key not in path_cache:
                # print(f'CACHE MISS: {key}')

                paths_this_direction = []

                segment, next_start, next_reaction_start, scores = scope_segment(base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, current_start, reaction_start, candidate_segment_start, current_chunk_size, n_samples, first_n_samples, reverse_search_bound, peak_tolerance, expansion_tolerance, segment_end_backoff, sr)

                my_path = list(current_path)
                if segment:
                    my_path.append(segment)

                if next_reaction_start >= len(reaction_audio) - 1:
                    # end this path
                    paths_this_direction.append(my_path)
                else:
                    continued_paths = find_pathways(my_path, base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, next_start, next_reaction_start, first_n_samples, n_samples, first_n_samples, reverse_search_bound, peak_tolerance, expansion_tolerance, segment_end_backoff, sr, path_cache, path_counts, alignment_bounds)
                    for full_path in continued_paths:
                        if full_path is not None:
                            paths_this_direction.append(full_path)

                path_cache[key] = paths_this_direction
            else: 
                # print(f'CACHE HIT!!! {key}')
                paths_this_direction = path_cache[key]

            path_counts['completed'] += 1

            # Don't add duplicate paths
            for new_path in paths_this_direction:
                duplicate = False
                for other_path in paths:
                    if other_path == new_path:
                        print('SKIPPING DUPLICATE PATH')
                        duplicate = True
                        break
                if not duplicate:
                    # print(f"Adding path {new_path}")
                    paths.append(new_path)





        # print(f"Got paths: {current_path} {current_start} {reaction_start} {paths}")

        print(f"Paths at point: {current_start / sr} {reaction_start / sr} [{len(paths)}] [{path_counts['completed']} / {path_counts['open']} = {path_counts['completed'] / path_counts['open'] * 100}%]")
        # for path in paths:
        #     # print(f"\tScore={scores[0]}  Fill={scores[1]}  Completion={scores[2]}  Similarity={scores[3]}")
        #     scores = path_score(path, base_audio, reaction_audio, sr)

        #     print(f"\tSimilarity={scores[2]}")            
        #     for sequence in path:
        #         print(f"\t{'*' if sequence[4] else ''}base: {float(sequence[2])}-{float(sequence[3])}  reaction: {float(sequence[0])}-{float(sequence[1])} ")

        return paths

    except Exception as e: 
        print("Got error down this path")
        traceback.print_exc()
        print(e)
        return [None]


def path_score(path, base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, sr): 
    total_fill = 1
    duration = 0

    total_reaction_span = 0
    for reaction_start, reaction_end, current_start, current_end, is_filler in path:
        if is_filler:
            total_fill += current_end - current_start
        else: 
            duration += (reaction_end - reaction_start)

        total_reaction_span += (reaction_end - reaction_start) * (reaction_start + (reaction_end - reaction_start) / 2)

    early_completion_score = 1 / total_reaction_span
    duration_score = 1 / (1 + abs(duration - len(base_audio)) / sr)

    reaction_audio_for_path = create_audio_extraction(reaction_audio, base_audio, path)

    alignment = mfcc_similarity(base_audio, reaction_audio_for_path, sr, mfcc1=base_audio_mfcc)

    return [duration_score, early_completion_score, alignment]


def find_best_path(candidate_paths, base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, sr):

    paths_with_scores = []

    for path in candidate_paths:
        paths_with_scores.append([path, path_score(path, base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, sr)])

    best_fill_score = 0 
    best_early_completion_score = 0
    best_similarity = 0 
    for path, scores in paths_with_scores:
        if scores[0] > best_fill_score:
            best_fill_score = scores[0]
        if scores[1] > best_early_completion_score:
            best_early_completion_score = scores[1]
        if scores[2] > best_similarity:
            best_similarity = scores[2]


    for path, scores in paths_with_scores:
        fill_score = scores[0] / best_fill_score
        completion_score = scores[1] / best_early_completion_score
        similarity_score = scores[2] / best_similarity

        scores[0] = fill_score
        scores[1] = completion_score
        scores[2] = similarity_score

        score = (fill_score + completion_score + similarity_score) / 3
        scores.insert(0, score)

    paths_by_score = sorted(paths_with_scores, key=lambda x: x[1][0], reverse=True)

    print("Paths by score:")
    for path,scores in paths_by_score:
        print(f"\tScore={scores[0]}  Duration={scores[1]}  Completion={scores[2]}  Similarity={scores[3]}")
        for sequence in path:
            print(f"\t\t{'*' if sequence[4] else ''}base: {float(sequence[2])}-{float(sequence[3])}  reaction: {float(sequence[0])}-{float(sequence[1])} ")


    return paths_by_score[0][0]

def cross_expander_aligner(base_audio, reaction_audio, step_size, min_segment_length_in_seconds, first_match_length_multiplier, reverse_search_bound, peak_tolerance, expansion_tolerance, segment_end_backoff, sr):
    # Convert seconds to samples
    n_samples = int(step_size * sr)
    first_n_samples = int(min_segment_length_in_seconds * sr)

    step = int(first_n_samples * first_match_length_multiplier)

    base_audio_mfcc = librosa.feature.mfcc(y=base_audio, sr=sr, n_mfcc=20)
    reaction_audio_mfcc = librosa.feature.mfcc(y=reaction_audio, sr=sr, n_mfcc=20)


    alignment_bounds = create_reaction_alignment_bounds(base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, sr, first_n_samples)
    # alignment_bounds = None

    paths = find_pathways([], base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, 0, 0, step, n_samples, first_n_samples, reverse_search_bound, peak_tolerance, expansion_tolerance, segment_end_backoff, sr, {}, {'open': 0, 'completed': 0, 'pruned': 0}, alignment_bounds)

    print('FOUND PATHS!', paths)


    path = find_best_path(paths, base_audio, base_audio_mfcc, reaction_audio, reaction_audio_mfcc, sr)

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


