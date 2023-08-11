from cross_expander.find_segment_start import find_next_segment_start_candidates



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
