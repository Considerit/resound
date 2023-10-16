from decimal import Decimal

from aligner.scoring_and_similarity import get_chunk_score
from aligner.find_segment_start import find_segment_starts
from aligner.pruning_search import prune_types


from utilities import conf, samples_per_frame, universal_frame_rate, is_close
from utilities import conversion_audio_sample_rate as sr




def find_correlation_end(reaction, current_start, reaction_start, step, reaction_end=None, end_at=None, max_score=0, cache={}):
    expansion_tolerance = conf.get('expansion_tolerance')

    base_audio = conf.get('song_audio_data')
    reaction_audio = reaction.get('reaction_audio_data')


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
            cache[key] = get_chunk_score(reaction, reaction_start, reaction_end, current_start, current_end) 

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
        # print('exiting early', current_start, current_end, reaction_start, reaction_end)       
        return (current_end, reaction_end)

    last_neg_slopes.reverse()

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

    if break_point is None:
        break_point = len(reaction_audio)

    if step > 250:

        # print(f"RECURSING AROUND {break_point / sr} - {next_end_at / sr} with neg_slope_avg of {neg_slope_avg}")

        return find_correlation_end(reaction, current_start, reaction_start, step = step // 2, reaction_end = break_point, end_at = next_end_at, max_score = max_score, cache = cache )
    else: 
        reaction_end = break_point
        current_end = current_start + reaction_end - reaction_start

        # seek to a frame boundary
        while (current_end - current_start) % samples_per_frame() > 0 and reaction_end < len(reaction_audio) and current_end < len(base_audio):
            current_end += 1
            reaction_end += 1

        result = Decimal(current_end - current_start) / Decimal(sr) * Decimal(universal_frame_rate())
        # print(f"by samples = {(current_end - current_start) % samples_per_frame()}  by rounding {result}")

        assert((current_end - current_start) % samples_per_frame() == 0)
        assert( is_close(result, round(result) )  )

        return (current_end, reaction_end)





#####################################################################
# Given a starting index into the reaction, find a good ending index

segment_scope_cache = {}
start_adjustment_cache = {}
def initialize_segment_end_cache():
    segment_scope_cache.clear()
    start_adjustment_cache.clear()




#####################
# Sometimes we're off in our segment start because, e.g. a reactor doesn't start from the beginning. So our very first match 
# is really low quality, which can cause problems like never popping out of the first segment. 
# To address this, we do a reverse match for a segment start: find the best start of reaction_chunk 
# in base_audio[current_start:]. The match should be at the beginning. If it isn't, then we're off base. 
# We have missing base audio. We can then try to backfill that by matching the missing segment with a 
# smaller minimum segment size (size of the missing chunk), so that we 
# (1) recover that base audio and (2) are aligned for subsequent sequences.  
def check_for_start_adjustment(reaction, current_start, reaction_start, candidate_segment_start, current_chunk_size):

    base_audio = conf.get('song_audio_data')
    reverse_search_bound = conf.get('reverse_search_bound')

    open_end = min(current_start+current_chunk_size+int(reverse_search_bound * sr), len(base_audio))
    reverse_chunk_size = min(current_chunk_size, open_end - current_start)

    scope_key = f'({current_start}, {reaction_start + candidate_segment_start}, {current_chunk_size}, {reverse_chunk_size})'
    
    if scope_key in start_adjustment_cache:
        return start_adjustment_cache[scope_key]



    peak_tolerance = conf.get('peak_tolerance')
    first_n_samples = conf.get('first_n_samples')

    reaction_audio = reaction.get('reaction_audio_data')
    song_audio_mfcc = conf.get('song_audio_mfcc')
    reaction_audio_mfcc = reaction.get('reaction_audio_mfcc')
    hop_length = conf.get('hop_length')



    candidate_reaction_chunk_start = reaction_start + candidate_segment_start
    candidate_reaction_chunk_end = reaction_start + candidate_segment_start + reverse_chunk_size
    candidate_reaction_chunk = reaction_audio[candidate_reaction_chunk_start:candidate_reaction_chunk_end]
    if reverse_chunk_size > len(candidate_reaction_chunk):
        reverse_chunk_size = len(candidate_reaction_chunk)
        open_end = min(open_end, current_start + reverse_chunk_size)


    open_base_chunk = base_audio[current_start:open_end]

    # print(f'\nDoing reverse index search  reaction_start={reaction_start+candidate_segment_start}  current_start={current_start}  {reverse_chunk_size} {len(candidate_reaction_chunk)} {len(open_base_chunk)}')

    reverse_index = find_segment_starts(
                        reaction = reaction, 
                        open_chunk=open_base_chunk, 
                        open_chunk_mfcc=song_audio_mfcc[:, round(current_start / hop_length):round(open_end / hop_length)], 
                        closed_chunk=candidate_reaction_chunk, 
                        closed_chunk_mfcc=reaction_audio_mfcc[:,      round(candidate_reaction_chunk_start / hop_length):round(candidate_reaction_chunk_end / hop_length) ], 
                        current_chunk_size=reverse_chunk_size, 
                        peak_tolerance=peak_tolerance, 
                        open_start=current_start, 
                        full_search=False,
                        closed_start=reaction_start + candidate_segment_start, 
                        distance=first_n_samples)

    if reverse_index and len(reverse_index) > 0:
        reverse_index = reverse_index[0]
    # print('reverse index:', reverse_index / sr)

    if reverse_index and reverse_index > 0: #sr / 100: 
        # print(f"Better match for base segment found later in reaction: using filler from base video from {current_start / sr} to {(current_start + reverse_index) / sr} with {(reaction_start - reverse_index) / sr} to {(reaction_start) / sr}")
        
        # seek to a frame boundary
        while (reverse_index - current_start) % samples_per_frame() > 0 and reverse_index < len(reaction_audio) and current_start + reverse_index < len(base_audio):
            reverse_index += 1

        # segment = [reaction_start - reverse_index, reaction_start, current_start, current_start + reverse_index, True]

        # reverse_candidate_found = (segment, current_start + reverse_index, reaction_start)

    start_adjustment_cache[scope_key] = reverse_index
    return reverse_index






def find_segment_end(reaction, current_start, reaction_start, candidate_segment_start, current_chunk_size):

    scope_key = f'({current_start}, {reaction_start + candidate_segment_start}, {current_chunk_size})'
    
    if scope_key in segment_scope_cache:
        prune_types['scope_cached'] += 1
        return segment_scope_cache[scope_key]

    n_samples = conf.get('n_samples')

    base_audio = conf.get('song_audio_data')
    reaction_audio = reaction.get('reaction_audio_data')

    #########################################

    reaction_start += candidate_segment_start # the first acceptable match in the reaction video
    # print(f"Start segment {current_start / sr}-{(current_start + n_samples) / sr} at {reaction_start / sr}")


    candidate_current_end, candidate_reaction_end = find_correlation_end(reaction, current_start, reaction_start, step = n_samples, cache={})
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
        # print(f"### Sequence of zero length at {reaction_start / sr} {current_start / sr}, skipping forward")
        segment_scope_cache[scope_key] = [None, current_start, reaction_start + samples_per_frame()]
        return segment_scope_cache[scope_key]

    # print(f"*** Completing match ({reaction_start / sr} [{reaction_start}], {reaction_end / sr} [{reaction_end}]), ({current_start / sr} [{current_start}], {current_end / sr} [{current_end}])\n")                

    assert( is_close(current_end - current_start, reaction_end - reaction_start) )
    segment = [reaction_start, reaction_end, current_start, current_end, False]


    segment_scope_cache[scope_key] = (segment, current_end, reaction_end)

    return segment_scope_cache[scope_key]




# def plot_curves(curves):
#     # Create a new figure
#     plt.figure(figsize=(10, 6))
    
#     # Set the colors you want to cycle through
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
#     # For each curve
#     for i, curve in enumerate(curves):
#         # Select the color for this curve
#         color = colors[i % len(colors)]

#         times = [end / 44100 for _,end,_ in curve]
#         scores = [score for _,_,score in curve]
            
#         # Plot the curve in the selected color
#         plt.plot(times, scores, color=color)

#     # Set labels
#     plt.xlabel('Time')
#     plt.ylabel('Score')

#     # Display the plot
#     plt.show()



