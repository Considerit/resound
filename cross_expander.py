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
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from typing import List, Tuple

from utilities import trim_and_concat_video, prepare_files, extract_audio, compute_precision_recall, samples_per_frame, universal_frame_rate, download_and_parse_reactions, is_close
from face_finder import detect_faces, create_reactor_view
from backchannel_isolator import process_reactor_audio

from decimal import Decimal, getcontext



def correct_peak_index(peak_index, chunk_len):
    # return max(0,peak_index)
    return max(0, peak_index - (chunk_len - 1))


def mfcc_similarity(audio_chunk1, audio_chunk2, sr):

    # Compute MFCCs for each audio chunk
    mfcc1 = librosa.feature.mfcc(y=audio_chunk1, sr=sr, n_mfcc=20)
    mfcc2 = librosa.feature.mfcc(y=audio_chunk2, sr=sr, n_mfcc=20)



    # Make sure the MFCCs are the same shape
    min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_len]
    mfcc2 = mfcc2[:, :min_len]

    # Compute mean squared error between MFCCs
    mse = np.mean((mfcc1 - mfcc2)**2)

    similarity = 1 / (1 + mse)
    return 10000 * similarity





def find_correlation_end(current_start, reaction_start, base_audio, reaction_audio, sr, expansion_tolerance, step, scores = [], reaction_end=None, end_at=None, max_score=0, backoff=0, cache={}):
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

            chunk_score = mfcc_similarity(reaction_chunk, chunk, sr)
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

        return find_correlation_end(current_start, reaction_start, base_audio, reaction_audio, sr, expansion_tolerance, step = step // 2, scores = scores, reaction_end = break_point, end_at = next_end_at, max_score = max_score, backoff = backoff, cache = cache )
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


def find_next_segment_start_candidates(open_chunk, closed_chunk, current_chunk_size, peak_tolerance, closed_start, open_start, sr, distance):

    # Perform cross correlation
    correlation = correlate(open_chunk, closed_chunk)

    # Find peaks
    peak_indices, _ = find_peaks(correlation, height=np.max(correlation)*peak_tolerance, distance=distance)
    peak_indices = sorted( peak_indices.tolist() )

    if len(peak_indices) == 0:
        print(f"No peaks found for {closed_start} [{closed_start / sr} sec] / {open_start} [{open_start / sr} sec] {np.max(correlation)}")
        return None

    # first, find the correct cluster of matches, then seek to the best starting point
    best_index = correct_peak_index(peak_indices[0], current_chunk_size)

    assert( len(closed_chunk) == current_chunk_size)

    scores = []
    max_score = 0
    for candidate in peak_indices:
        candidate_index = correct_peak_index(candidate, current_chunk_size)        
        chunk_score = mfcc_similarity(open_chunk[candidate_index:candidate_index + current_chunk_size], closed_chunk, sr)    
        scores.append( (candidate_index, chunk_score) ) 
        if chunk_score > max_score:
            max_score = chunk_score
        print(f"Comparing start at {(closed_start + candidate_index) / sr} with score {correlation[candidate]}  [{chunk_score}]")

    candidates = [ candidate_index for candidate_index, chunk_score in scores if chunk_score >= max_score * peak_tolerance ]
    return candidates

def find_next_best_segment_start(open_chunk, closed_chunk, current_chunk_size, peak_tolerance, closed_start, open_start, sr, distance):
    candidates = find_next_segment_start_candidates(open_chunk, closed_chunk, current_chunk_size, peak_tolerance, closed_start, open_start, sr, distance)
    if not candidates: 
      return None
    return candidates[0]
    

def cross_expander_aligner(base_audio, reaction_audio, step_size, min_segment_length_in_seconds, first_match_length_multiplier, reverse_search_bound, peak_tolerance, expansion_tolerance, segment_end_backoff, segment_combination_threshold, sr):
    # Convert seconds to samples
    n_samples = int(step_size * sr)
    first_n_samples = int(min_segment_length_in_seconds * sr)

    match_segments = []

    current_start = 0
    reaction_start = 0

    step = int(first_n_samples * first_match_length_multiplier)

    all_scores = []

    while current_start < len(base_audio) - 1:
        # Each loop represents a segment starting with the chunk

        # Get the base audio chunk
        current_end = min(current_start + step, len(base_audio))
        chunk = base_audio[current_start:current_end]
        current_chunk_size = current_end - current_start



        #################
        # Handle case where reaction video finishes before base video
        if reaction_start >= len(reaction_audio) - 1 or current_start > len(base_audio) - int(.5 * sr):
            length_remaining = len(base_audio) - current_start - 1
            match_segments.append(  (reaction_start, reaction_start + length_remaining, current_start, current_start + length_remaining, True)   )
            print(f"Reaction video finished before end of base video. Backfilling with base video ({length_remaining / sr}s).")
            break
        ########


        print(f'\nFinding segment start  reaction_start={reaction_start} (of {len(reaction_audio)})  current_start={current_start} (of {len(base_audio)})')
        best_index = find_next_best_segment_start(reaction_audio[reaction_start:], chunk, current_chunk_size, peak_tolerance, reaction_start, current_start, sr, distance=first_n_samples)
        if best_index is None:
            current_start += 1
            if (len(base_audio) - step - current_start) / sr < .5:
                print(f"Could not find match in remainder of reaction video!!! Stopping.")
                break
            else:
                print(f"Could not find match in remainder of reaction video!!! Skipping sample.")
                continue


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
        candidate_reaction_chunk = reaction_audio[reaction_start+best_index:reaction_start+best_index+reverse_chunk_size]
        if reverse_chunk_size > len(candidate_reaction_chunk):
            reverse_chunk_size = len(candidate_reaction_chunk)
            open_end = min(open_end, current_start + reverse_chunk_size)


        open_base_chunk = base_audio[current_start:open_end]

        # print(f'\nDoing reverse index search  reaction_start={reaction_start+best_index}  current_start={current_start}  {reverse_chunk_size} {len(candidate_reaction_chunk)} {len(open_base_chunk)}')

        reverse_index = find_next_best_segment_start(open_base_chunk, candidate_reaction_chunk, reverse_chunk_size, peak_tolerance, current_start, reaction_start + best_index, sr, distance=first_n_samples)

        # print('reverse index:', reverse_index / sr)


        if reverse_index > 0: 
            print(f"Better match for base segment found later in reaction: using filler from base video from {current_start / sr} to {(current_start + reverse_index) / sr} with {(reaction_start - reverse_index) / sr} to {(reaction_start) / sr}")
            
            # seek to a frame boundary
            while (reverse_index - current_start) % samples_per_frame() > 0 and reverse_index < len(reaction_audio) and current_start + reverse_index < len(base_audio):
                reverse_index += 1

            match_segments.append((reaction_start - reverse_index, reaction_start, current_start, current_start + reverse_index, True))

            current_start += reverse_index

            continue

        #########################################



        reaction_start += best_index # the first acceptable match in the reaction video
        # print(f"Start segment {current_start / sr}-{(current_start + n_samples) / sr} at {reaction_start / sr}")


        candidate_current_end, candidate_reaction_end, scores = find_correlation_end(current_start, reaction_start, base_audio, reaction_audio, sr, expansion_tolerance, step = n_samples, backoff = segment_end_backoff, cache={})
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
            reaction_start += samples_per_frame()
            continue

        print(f"*** Completing match ({reaction_start / sr} [{reaction_start}], {reaction_end / sr} [{reaction_end}]), ({current_start / sr} [{current_start}], {current_end / sr} [{current_end}])\n")                

        assert( is_close(current_end - current_start, reaction_end - reaction_start) )
        match_segments.append((reaction_start, reaction_end, current_start, current_end, False))


        current_start = current_end
        reaction_start = reaction_end

        if reaction_start >= len(reaction_audio) - 1:
            break

        # print("scores", scores)
        all_scores.append(scores)
        # else: 
        #     reaction_start += int(.5 * sr) # skip half a second and try again

        step = first_n_samples

    # plot_curves(all_scores)


    # print("segs", match_segments)

    # for current_start, current_end, current_base_start, current_base_end, filler in match_segments:
    #     print((current_end - current_start) / sr, (current_base_end - current_base_start) / sr, filler)

    return match_segments



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


