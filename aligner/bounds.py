import os
from utilities import conversion_audio_sample_rate as sr
from utilities import conf, save_object_to_file, read_object_from_file
import matplotlib.pyplot as plt
from silence import is_silent

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


def create_reaction_alignment_bounds(reaction, first_n_samples, seconds_per_checkpoint=12, peak_tolerance=.5):
    from aligner.path_painter import get_candidate_starts, get_signals


    saved_bounds = os.path.splitext(reaction.get('aligned_path'))[0] + '-intercept_bounds.pckl'
    if os.path.exists(saved_bounds):
        reaction['alignment_bounds'] = read_object_from_file(saved_bounds)
        print_alignment_bounds(reaction)        
        return reaction['alignment_bounds']


    
    # Initialize the list of bounds
    bounds = []

    
    clip_length = int(3 * sr)

    reaction_audio = reaction.get('reaction_audio_data')
    base_audio = conf.get('song_audio_data')

    if not reaction.get('unreliable_bounds', False):

        hop_length = conf.get('hop_length')
        reaction_audio_mfcc = reaction.get('reaction_audio_mfcc')
        song_audio_mfcc = conf.get('song_audio_mfcc')

        reaction_span = reaction.get('end_reaction_search_at', len(reaction_audio))
        start_reaction_search_at = reaction.get('start_reaction_search_at', 0)


        base_length_sec = len(base_audio) / sr  # Length of the base audio in seconds

        n_timestamps = round(base_length_sec / seconds_per_checkpoint) 
        
        timestamps = [i * base_length_sec / (n_timestamps + 1) for i in range(1, n_timestamps + 1)]

        if base_length_sec - timestamps[-1] > 15:
            timestamps.append(base_length_sec - 10)


        timestamps = [int(t * sr) for t in timestamps]

        print(f"Creating alignment bounds with tolerance {peak_tolerance} every {seconds_per_checkpoint} seconds at {timestamps} ")


        for i,ts in enumerate(timestamps):
            # Define the segments on either side of the timestamp

            segment_times = [
              (max(0, ts - clip_length), ts),
              (max(0, ts - int(clip_length / 2)), min(len(base_audio), ts + int(clip_length / 2) )),
              (ts, min(len(base_audio), ts + clip_length))
            ]

            segments = [  (s, e, base_audio[s:e], song_audio_mfcc[:, round(s / hop_length): round(e / hop_length)]) for s,e in segment_times  ]


            # for j, segment in enumerate(segments):
            #     filename = f"segment_{i}_{j}.wav"
            #     sf.write(filename, segment, sr)
            
            # Initialize the list of max indices for the segments
            max_indices = []
            
            print(f"ts: {ts / sr}")


            predominantly_silent = is_silent(segments[0][2], threshold_db=-20) or is_silent(segments[2][2], threshold_db=-20)

            if predominantly_silent:
                print("\tNOT ADDING BOUND. Too silent.")
                continue

            # For each segment
            highest_bounds = []
            for base_start, end, chunk, chunk_mfcc in segments:
                # Find the candidate indices for the start of the matching segment in the reaction audio

                reaction_start = base_start + start_reaction_search_at

                signals, evaluate_with = get_signals(reaction, base_start, reaction_start, clip_length)

                candidates = get_candidate_starts(
                    reaction=reaction, 
                    signals=signals, 
                    peak_tolerance=peak_tolerance, 
                    open_start=reaction_start, 
                    closed_start=base_start, 
                    chunk_size=clip_length,
                    distance=first_n_samples, 
                    upper_bound=reaction_span - len(base_audio),
                    evaluate_with=evaluate_with
                )


                if candidates is None: 
                    candidates = []
                elif len(candidates) > 0:
                    print(f"\tCandidates: {[ int((1000 * (c+reaction_start))/sr)/1000 for c in candidates]}  {(max(candidates) + reaction_start) / sr:.1f} [{max(candidates)/sr:.1f}]")

                this_ts = []
                for c in candidates:
                    candidate_reaction_start = c + reaction_start
                    intercept = candidate_reaction_start - base_start
                    max_indices.append(intercept)
                    this_ts.append(intercept)
                if len(this_ts) > 0:
                    highest_bounds.append(max(this_ts))
            
            if len(max_indices) < len(segment_times):
                print(f"\tCOULD NOT FIND BOUND FOR {ts / sr}")
            else: 
                agrees = True
                for i,m in enumerate(highest_bounds):
                    if i > 0:
                        agrees = agrees and abs(m - highest_bounds[i - 1]) < sr

                if agrees:
                    bounds.append( [ts, max_indices] )
                else:
                    print("\tNOT ADDING BOUND: max doesn't agree")
                    for i,m in enumerate(highest_bounds):
                        if i > 0:
                            agrees = agrees and abs(m - highest_bounds[i - 1]) < sr
                            # print(f"{agrees} {m/sr} {highest_bounds[i-1]/sr} {abs(m - highest_bounds[i - 1]) / sr}")

    else:
        print("Skipping auto bounds. Deemed unreliable.")
    

    grace = clip_length * 2

    # Factor in manually configured bounds    
    manual_bounds = reaction.get('manual_bounds', False)
    if manual_bounds:
        for mbound in manual_bounds:
            ts, upper = mbound
            ts = int(ts*sr); upper = int(upper*sr)
            bounds.append([ts, [upper - ts - grace + int(.5*sr)]])
            print(f"Inserted upper bound {(upper - ts)/sr} for {ts/sr}")

    if reaction.get('end_reaction_search_at', False):
        bounds.append([len(base_audio), [reaction.get('end_reaction_search_at') - len(base_audio) - grace]])
        print(f"Inserted upper bound at end ({len(base_audio)/sr}): {reaction.get('end_reaction_search_at') / sr} [{reaction.get('end_reaction_search_at') - len(base_audio) - grace}]!")
    else: 
        bounds.append([len(base_audio), [len(reaction_audio) - len(base_audio) - grace]])

    bounds.sort(key=lambda x: x[0], reverse=True)

    for b in bounds:
        b.append(max(b[1]))



    # Now, ensure the integrity of the bounds

    last_intercept = 99999999999999999999999999999999

    for i, (base_ts, all_candidates, current_candidate) in enumerate(bounds):

        

        if i > 0:
            last_base_ts, _, last_intercept = bounds[i-1]
            last_reaction_ts = last_base_ts + last_intercept
            
        # enforce integrity condition
        # find the latest match that happens before the next bound 
        candidates = [ intercept for intercept in all_candidates if intercept <= last_intercept + grace ]

        if len(candidates) == 0:
            print ("**********")
            print(f"Could not find bound with integrity!!!!")
            print(f"\tTrying again with higher tolerance and shifted checkpoints.")
            print(f"\tFor timestamp {base_ts/sr}. Needed value below {(last_intercept+base_ts+grace)/sr}, but had min of {(min(all_candidates) + base_ts)/sr} (and max {(max(all_candidates)+base_ts)/sr})")            
            print("************")

            return create_reaction_alignment_bounds(reaction, first_n_samples, seconds_per_checkpoint=seconds_per_checkpoint+10, peak_tolerance=peak_tolerance * .9)

        if max(all_candidates) != max(candidates):
            print(f"New bound for {base_ts/sr} is {(max(candidates) + base_ts) / sr}, forced value below {(last_intercept+base_ts + grace)/sr}, cuts off {(max(all_candidates)+base_ts)/sr}")
        else: 
            print(f"Bound of {max(all_candidates) / sr} maintained for {base_ts/sr}")
        new_bound = max( candidates  )

        bounds[i][2] = new_bound

    bounds.sort(key=lambda x: x[0], reverse=False)

    alignment_bounds = []
    for base_ts, max_indices, bounding_intercept in bounds:
        alignment_bounds.append( (base_ts, bounding_intercept + grace)  )

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
    for base_ts, intercept in alignment_bounds:
        print(f"\t{base_ts / sr}  <=  {(base_ts + intercept) / sr} (int={intercept})")


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

    if False: 
        # Unzip the tuples to get separate lists of base_ts and reaction_ts
        base_ts, intercepts = zip(*alignment_bounds)
        base_ts = [bs/sr for bs in base_ts]
        reaction_ts = [(b+i)/sr for b,i in alignment_bounds]

        # Plot the points
        plt.scatter(base_ts, reaction_ts)
        
        # Connect points with lines
        plt.plot(base_ts, reaction_ts, '-o')  

        # Find the min and max values of base_ts for the width of the chart
        x_min = min(base_ts)

        # Draw line with slope of 1 through each point
        for x, c in alignment_bounds:
            c /= sr  # Calculate the y-intercept
            plt.plot([x_min, x/sr], [x_min + c, x/sr + c], 'r--', alpha=0.5)  # Plot the line using the y = mx + c equation

        # Set labels and title
        plt.xlabel("Base Timestamps")
        plt.ylabel("Reaction Timestamps")
        plt.title(f"Alignment Bounds Visualization for {reaction.get('channel')}")

        plt.grid(True)
        plt.show()



def in_bounds(bound, base_start, reaction_start):
    return reaction_start <= bound + base_start

def get_bound(alignment_bounds, base_start, reaction_end, base_end=None): 
    if base_end is None:
        base_end = len(conf.get('song_audio_data'))

    matching_intercepts = [intercept for base_ts,intercept in alignment_bounds if base_start < base_ts]
    if len(matching_intercepts) > 0:
        return min(matching_intercepts)
    return reaction_end - base_end 
