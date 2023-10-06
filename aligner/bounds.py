import os
from utilities import conversion_audio_sample_rate as sr
from utilities import conf, save_object_to_file, read_object_from_file

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
#     find_segment_starts. Don't bother using parameters, I'll fill that in. Assume you get 
#     a list of candidate matching indexes of the clip in the reaction audio. Take the greatest of these matching 
#     indexes, between the two clips. This will be the bound value for this timestamp. 
#   - Now ensure the integrity of these timestamp bounds. Make sure that every earlier timestamp bound a is 
#   less than every later timestamp bound b by at least (t(b) - t(a)) where t is the time difference between 
#   b and a in the base audio. If the earlier timestamp bound doesn't satisfy this constraint, set the earlier 
#   bound a to v(b) - (t(b) - t(a)), where v is the value of the bound's latest match in the reaction audio. To 
#   accomplish the integrity checking, walk backwards from the last timestamp.


def create_reaction_alignment_bounds(reaction, first_n_samples, seconds_per_checkpoint=24, peak_tolerance=.5):
    from aligner.find_segment_start import find_segment_starts


    saved_bounds = os.path.splitext(reaction.get('aligned_path'))[0] + '-bounds.pckl'
    if os.path.exists(saved_bounds):
        reaction['alignment_bounds'] = read_object_from_file(saved_bounds)
        print_alignment_bounds(reaction)        
        return reaction['alignment_bounds']





    # profiler = cProfile.Profile()
    # profiler.enable()

    base_audio = conf.get('base_audio_data')
    reaction_audio = reaction.get('reaction_audio_data')
    hop_length = conf.get('hop_length')
    reaction_audio_mfcc = reaction.get('reaction_audio_mfcc')
    base_audio_mfcc = conf.get('base_audio_mfcc')

    clip_length = int(2 * sr)
    base_length_sec = len(base_audio) / sr  # Length of the base audio in seconds

    n_timestamps = round(base_length_sec / seconds_per_checkpoint) 
    
    timestamps = [i * base_length_sec / (n_timestamps + 1) for i in range(1, n_timestamps + 1)]

    if base_length_sec - timestamps[-1] > 15:
        timestamps.append(base_length_sec - 10)


    timestamps_samples = [int(t * sr) for t in timestamps]
    
    # Initialize the list of bounds
    bounds = []

    print(f"Creating alignment bounds with tolerance {peak_tolerance} every {seconds_per_checkpoint} seconds at {timestamps} ")
    
    # For each timestamp
    for i,ts in enumerate(timestamps_samples):
        # Define the segments on either side of the timestamp

        segment_times = [
          (max(0, ts - clip_length), ts),
          (max(0, ts - int(clip_length / 2)), min(len(base_audio), ts + int(clip_length / 2) )),
          (ts, min(len(base_audio), ts + clip_length))
        ]

        segments = [  (s, e, base_audio[s:e], base_audio_mfcc[:, round(s / hop_length): round(e / hop_length)]) for s,e in segment_times  ]


        # for j, segment in enumerate(segments):
        #     filename = f"segment_{i}_{j}.wav"
        #     sf.write(filename, segment, sr)
        
        # Initialize the list of max indices for the segments
        max_indices = []
        
        print(f"ts: {ts / sr}")
        # For each segment
        for start, end, chunk, chunk_mfcc in segments:
            # Find the candidate indices for the start of the matching segment in the reaction audio

            candidates = find_segment_starts(
                                    reaction=reaction, 
                                    open_chunk=reaction_audio[start:], 
                                    open_chunk_mfcc=reaction_audio_mfcc[:, round(start / hop_length):],
                                    closed_chunk=chunk, 
                                    closed_chunk_mfcc= chunk_mfcc,
                                    current_chunk_size=clip_length, 
                                    peak_tolerance=peak_tolerance, 
                                    full_search=True,
                                    open_start=start,
                                    closed_start=start, 
                                    distance=first_n_samples, 
                                    filter_for_similarity=True)


            if candidates is None: 
                candidates = []
            else:
                print(f"\tCandidates: {[ int((1000 * (c+start))/sr)/1000 for c in candidates]}  {(max(candidates) + start) / sr:.1f}")

            for c in candidates:
                max_indices.append(ts + c + clip_length * 2)
        
        bounds.append( max_indices )

        timestamps_samples[i] -= clip_length
    
    # Now, ensure the integrity of the bounds
    smoothed_bounds = [max(b) for b in bounds]
    print("smoothed_bounds", [ b/sr for b in smoothed_bounds  ])
    print("timestamps_samples", [ t for t in timestamps])
    for i in range(len(bounds) - 1, -1, -1):  # Start from the last element and go backward
        

        if i < len(bounds) - 1:
            previous_bound = smoothed_bounds[i+1] - (timestamps_samples[i+1] - timestamps_samples[i]) + clip_length * 2 

            # # If the current bound doesn't satisfy the integrity condition
            # if bounds[i] >= previous_bound:
            #     # Update the current bound
            #     bounds[i] = bounds[i+1] - (timestamps_samples[i+1] - timestamps_samples[i])

        else: 
            previous_bound = 99999999999999999999999999999999

        # enforce integrity condition
        # find the latest match that happens before the next bound 
        candidates = [ b for b in bounds[i] if b <= previous_bound ]
        if len(candidates) == 0:
            print ("**********")
            print(f"Could not find bound with integrity!!!! Trying again with higher tolerance and shifted checkpoints. Happened at {len(bounds) - i}", timestamps_samples[i] / sr)
            
            print("adjusted smoothed_bounds", [ b/sr for b in smoothed_bounds  ])
            return create_reaction_alignment_bounds(reaction, first_n_samples, seconds_per_checkpoint=seconds_per_checkpoint+10, peak_tolerance=peak_tolerance * .9)
        else:
            # print(f"New bound for {timestamps[i]} is {max(candidates)}", candidates, bounds[i])
            new_bound = max( candidates  )

        smoothed_bounds[i] = new_bound

    alignment_bounds = list(zip(timestamps_samples, smoothed_bounds))




    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
    # stats.print_stats()
    reaction['alignment_bounds'] = alignment_bounds
    print_alignment_bounds(reaction)

    save_object_to_file(saved_bounds, reaction['alignment_bounds'])

    return alignment_bounds

def print_alignment_bounds(reaction):
    alignment_bounds = reaction['alignment_bounds']
    print(f"The alignment bounds:")
    for base_ts, last_reaction_match in alignment_bounds:
        print(f"\t{base_ts / sr}  <=  {last_reaction_match / sr}")


    gt = reaction.get('ground_truth')
    if gt: 
        print("Checking if ground truth is within alignment bounds")
        current_start = 0
        for reaction_start, reaction_end in gt:
            bound = get_bound(alignment_bounds, current_start, reaction_end)
            if( not in_bounds(bound, current_start, reaction_start)):
                print(f"\tOh oh! {reaction_start / sr:.1f} is not in bounds of {current_start/sr:.1f}")
            else: 
                print(f"\tIn bounds: {reaction_start/sr:.1f} for {current_start/sr:.1f}")
            current_start += reaction_end - reaction_start




def in_bounds(bound, base_start, reaction_start):
    return reaction_start <= bound

def get_bound(alignment_bounds, base_start, reaction_end): 
    for base_ts, last_reaction_match in alignment_bounds:
        if base_start < base_ts:
            return last_reaction_match
    return reaction_end
