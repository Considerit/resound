from utilities import conversion_audio_sample_rate as sr
from utilities import conf, print_profiling, save_object_to_file, read_object_from_file
from aligner.bounds import get_bound, in_bounds, create_reaction_alignment_bounds
from aligner.find_segment_start import find_segment_starts, initialize_segment_start_cache, score_start_candidates, correct_peak_index
from aligner.find_segment_end import find_segment_end, initialize_segment_end_cache
from aligner.scoring_and_similarity import find_best_path, initialize_path_score, initialize_segment_tracking, get_segment_mfcc_cosine_similarity_score, get_segment_mfcc_cosine_similarity_score, path_score, print_path, path_score_by_mfcc_cosine_similarity, truncate_path
from aligner.cross_expander import compress_segments
from aligner.pruning_search import is_path_quality_poor, initialize_path_pruning
from silence import is_silent
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time
import math



attempts_progression = {
    'chunk_size':      [3, 3, 5, 5, 8, 8, 10, 10, 12, 12,  3,  5,  8, 10, 12],
    'allowed_spacing': [3, 6, 3, 6, 3, 6,  3,  6,  3,  6, 12, 12, 12, 12, 12]
}



def get_chunk_size(reaction, attempts=0):
    chunk_size = max(reaction.get('chunk_size', 0), attempts_progression['chunk_size'][attempts])
    chunk_size *= sr    
    chunk_size = int(chunk_size)
    return chunk_size

def paint_paths(reaction, peak_tolerance=.4, allowed_spacing=None, attempts=0):


    chunk_size = get_chunk_size(reaction, attempts=attempts)
    if allowed_spacing is None: 
        allowed_spacing = attempts_progression['allowed_spacing'][attempts]
        # print(f"Attempts={attempts}   {attempts % 3}    {allowed_spacing}")
        allowed_spacing *= sr
        

    print(f"\n###############################\n# {conf.get('song_key')} / {reaction.get('channel')}")
    print(f"Painting path with chunk size {chunk_size / sr} and {allowed_spacing / sr} spacing")

    initialize_segment_end_cache()
    initialize_path_score()
    initialize_segment_tracking()
    initialize_paint_caches()
    initialize_path_pruning()

    
    step = int(.5 * sr)


    start = time.perf_counter()

    print("FIND SEGMENTS")

    segments = find_segments(reaction, chunk_size, step, peak_tolerance)

    splay_paint(reaction, segments, stroke_alpha=.2, show_live=True, chunk_size=chunk_size)

    print("ABSORB")
    consolidated_segments = consolidate_segments(segments, bridge_gaps=True)

    splay_paint(reaction, consolidated_segments, stroke_alpha=.2, show_live=True, chunk_size=chunk_size)


    print("PRUNE UNREACHABLE")

    __ = prune_unreachable_segments(reaction, consolidated_segments, allowed_spacing, prune_links = False)

    splay_paint(reaction, consolidated_segments, stroke_alpha=.2, show_live=True, chunk_size=chunk_size)

    print("SHARPEN INTERCEPT")
    sharpen_intercept(reaction, chunk_size, step, consolidated_segments, allowed_spacing)


    # splay_paint(reaction, consolidated_segments, stroke_alpha=.2, show_live=True, chunk_size=chunk_size)

    print("ABSORB2")
    consolidated_segments = consolidate_segments(consolidated_segments, bridge_gaps=True)

    # for seg in consolidated_segments:
    #     rs,re,bs,be = seg.get('end_points')
    #     print(f"[{rs-bs}] {bs/sr}-{be/sr} / {rs/sr}-{re/sr}   [{seg.get('pruned')}]")

    splay_paint(reaction, consolidated_segments, stroke_alpha=.2, show_live=True, chunk_size=chunk_size)

    print("SHARPEN ENDPOINTS")

    sharpen_endpoints(reaction, chunk_size, step, consolidated_segments, allowed_spacing)

    # splay_paint(reaction, consolidated_segments, stroke_alpha=.2, show_live=True, chunk_size=chunk_size)

    print("SHARPEN INTERCEPT2")

    sharpen_intercept(reaction, chunk_size, step, consolidated_segments, allowed_spacing)

    splay_paint(reaction, consolidated_segments, stroke_alpha=.2, show_live=True, chunk_size=chunk_size)

    print("THIN CLUSTERS")

    prune_low_quality_segments(reaction, consolidated_segments)

    # splay_paint(reaction, consolidated_segments, stroke_alpha=.2, show_live=True, chunk_size=chunk_size)

    # if attempts % 3 == 0:
    print("PRUNE NEIGHBORS")

    prune_neighbors(reaction, consolidated_segments, allowed_spacing)

    splay_paint(reaction, consolidated_segments, stroke_alpha=.2, show_live=True, chunk_size=chunk_size)

    print("PRUNE UNREACHABLE2")

    joinable_segment_map = prune_unreachable_segments(reaction, consolidated_segments, allowed_spacing, prune_links = False)

    splay_paint(reaction, consolidated_segments, stroke_alpha=.2, show_live=True, chunk_size=chunk_size)

    print(f"Constructing paths from {len(consolidated_segments)} viable segments")

    paths = construct_all_paths(reaction, consolidated_segments, joinable_segment_map, allowed_spacing)

    segments_by_key = {}
    for segment in consolidated_segments:
        segments_by_key[segment['key']] = segment

    # paths = finesse_paths(reaction, paths, segments_by_key)

    if len(paths) == 0 and attempts < len(attempts_progression['chunk_size']) - 1: 
        print(f"No paths found.")
        splay_paint(reaction, consolidated_segments, stroke_alpha=.2, show_live=False, chunk_size=chunk_size)

        return paint_paths(reaction, peak_tolerance, attempts=attempts+1)

    print(f"Found {len(paths)} paths")

    best_path = find_best_path(reaction, paths)
    # best_path = compress_segments(best_path)

    micro_aligned = micro_align_path(reaction, best_path, segments_by_key)

    best_path = find_best_path(reaction, [best_path, micro_aligned])
    alignment_duration = (time.perf_counter() - start) / 60 # in minutes

    splay_paint(reaction, segments, stroke_alpha=.2, show_live=False, paths=[best_path], chunk_size=chunk_size)

    return best_path

def consolidate_segments(all_segments, bridge_gaps=False):
    intercepts = {}
    seen = {}  
    intercept_map = {}  
    all_segments = [s for s in all_segments if not s.get('pruned', False)]

    for s in all_segments: 

        dup_key = f"{s['end_points'][2]}-{s['end_points'][3]}-{s['end_points'][0]}-{s['end_points'][1]}"
        if dup_key in seen:
            continue
        seen[dup_key] = True

        intercept = s['end_points'][0] - s['end_points'][2]
        key = round(intercept / sr / 2)
        if key not in intercepts:
            intercepts[key] = []
            intercept_map[key] = intercept

        intercepts[key].append(s)

        intercept_map[key] = min(intercept_map[key], intercept)



    filtered_segments = []
    for key, segments in intercepts.items():        
        intercept = intercept_map[key]
        if len(segments) == 1:
            filtered_segments.append(segments[0])
            continue

        if len(segments) == 0:
            raise(Exception("No segments!", key))

        max_bs = 0
        min_bs = 999999999999999999999999999999999
        not_subsumed = []
        for i, s in enumerate(segments):
            bs = s['end_points'][2]
            be = s['end_points'][3]
            # print('ABSORB:', intercept, bs/sr, be/sr)


            subsumed_by_other = False
            for j, s2 in enumerate(segments):
                if i==j:
                    continue
                bs2 = s2['end_points'][2]
                be2 = s2['end_points'][3]

                if bs2 <= bs and be2 >= be:
                    subsumed_by_other = bs2 != bs or be2 != be
                    break

            if not subsumed_by_other:
                not_subsumed.append(s)

        if len(not_subsumed) == 0:
            print(f"ERROR CASE: everything subsumed {intercept} {len(segments)}", segments)

        filtered_segments += not_subsumed

        if False: 
            if len(not_subsumed) == 1 or not bridge_gaps:
                filtered_segments += not_subsumed
            else:

                # yoyo = intercept < 79*sr and intercept > 77*sr

                bridge_candidates = not_subsumed

                bridge_occurred = True
                while bridge_occurred:

                    bridge_occurred = False

                    bridged = {}

                    bridge_candidates.sort(key=lambda x: x['end_points'][3] - x['end_points'][2], reverse=True) 
                    evaluated_this_round = []

                    while( len(bridge_candidates) > 0 ):


                        super_segment = bridge_candidates.pop()
                        if str(super_segment['end_points']) in bridged:
                            continue

                        bs = super_segment['end_points'][2]
                        be = super_segment['end_points'][3]

                        # if yoyo:
                        #     print(f"SUPER: {bs/sr}-{be/sr}")

                        def dist_from_super(seg):
                            bs2 = s['end_points'][2]
                            be2 = s['end_points'][3]
                            d = min(abs(bs2 - be), abs(bs - be2))
                            return d


                        bridge_candidates.sort(key=dist_from_super)



                        for s in bridge_candidates:
                            if str(s['end_points']) in bridged: 
                                continue

                            bs = super_segment['end_points'][2]
                            be = super_segment['end_points'][3]

                            bs2 = s['end_points'][2]
                            be2 = s['end_points'][3]

                            if bs == bs2 and be2 == be: 
                                continue

                            d = min(abs(bs2 - be), abs(bs-be2))


                            # if yoyo:
                            #     print(f"\tSUB: {bs2/sr}-{be2/sr}")

                            if d <= be - bs: # only subsume if the section to be subsumed is closer to the big segment than its width
                                # if yoyo:
                                #     print(f"\t\tBRIDGED!")

                                bridge_occurred = True

                                super_segment['end_points'] = [ min(bs,bs2) + intercept, max(be,be2) + intercept, min(bs,bs2), max(be,be2) ]
                                # existing_strokes = {}
                                # for stroke in super_segment['strokes']:
                                #     existing_strokes[stroke[2]] = True
                                for stroke in s['strokes']:
                                    # if stroke[2] not in existing_strokes:
                                    # existing_strokes[stroke[2]] = True
                                    super_segment['strokes'].append(stroke)

                                bridged[str(s['end_points'])] = True

                        evaluated_this_round.append(super_segment)


                    bridge_candidates = [s for s in evaluated_this_round if str(s['end_points']) not in bridged]

                filtered_segments += bridge_candidates


    
    merge_thresh = int(sr / 2)
    if bridge_gaps:
        by_intercept = [(s['end_points'][0] - s['end_points'][2], s) for s in filtered_segments]
        by_intercept.sort(  key=lambda x: x[0]  )
        merged = {} # indicies of by_intercept that have already been merged into a different segment

        merge_happened = True
        iteration = 0
        while merge_happened:
            # print(f"Finding merges iteration {iteration}")
            iteration += 1
            merge_happened = False

            for i, (intercept, segment) in enumerate(by_intercept):
                if i in merged: 
                    continue

                for j in range(i+1, len(by_intercept)):
                    if j in merged: 
                        continue

                    # we're done finding merge candidates if our current intercept is too far out of bounds
                    candidate_intercept, candidate_segment = by_intercept[j]
                    if candidate_intercept - intercept > merge_thresh:
                        break

                    # only subsume if the section to be subsumed is closer to the big segment than its width
                    candidate_seg_length = candidate_segment['end_points'][3] - candidate_segment['end_points'][2]                    
                    dx = min( abs(segment['end_points'][3] - candidate_segment['end_points'][2]), abs(candidate_segment['end_points'][3] - segment['end_points'][2]))
                    dy = abs(candidate_intercept - intercept)
                    dist_from_segment = (dx ** 2 + dy ** 2) ** 0.5
                    if dist_from_segment > candidate_seg_length / 2: 
                        continue

                    # merge!
                    merge_happened = True
                    seg_length = segment['end_points'][3] - segment['end_points'][2]
                    if seg_length > candidate_seg_length:
                        src = candidate_segment
                        dest = segment
                        target_intercept = intercept
                        merged[j] = True
                        # print(f"\tMERGE! {target_intercept / sr}: {j} [{src['end_points'][2]/sr} - {src['end_points'][3]/sr}] => {i} [{dest['end_points'][2]/sr} - {dest['end_points'][3]/sr}]")

                    else: 
                        src = segment
                        dest = candidate_segment
                        target_intercept = candidate_intercept                        
                        merged[i] = True
                        # print(f"\tMERGE! {target_intercept / sr}: {i} [{src['end_points'][2]/sr} - {src['end_points'][3]/sr}] => {j} [{dest['end_points'][2]/sr} - {dest['end_points'][3]/sr}]")

                    (src_rs, src_re, src_bs, src_be) = src['end_points']
                    (dest_rs, dest_re, dest_bs, dest_be) = dest['end_points']

                    dest['end_points'] = [ min(src_bs,dest_bs) + target_intercept, max(src_be,dest_be) + target_intercept, min(src_bs,dest_bs), max(src_be,dest_be) ]

                    for stroke in src['strokes']:
                        dest['strokes'].append(stroke)

                    if src == segment:
                        break # we've merged into this one, so stop merging more into it


        consolidated_segments = [s for i, (b, s) in enumerate(by_intercept) if i not in merged]

    else: 
        consolidated_segments = filtered_segments


    print(f'\t\tconsolidate_segments resulted in {len(consolidated_segments)} segments, down from {len(all_segments)}')
    return consolidated_segments


def get_neighborhoods(segments, neighborly_distance): 
    

    # each line segment in segments is a dictionary with an entry 'end_points'. End points is a tuple 
    # (reaction_start, reaction_end, base_start, base_end). (base_start, reaction_start) and 
    # (base_end, reaction_end) are endpoints on a line with a slope of 1. This function returns a list of 
    # "neighborhoods". Each neighborhood is an array of segments such that each segment in the
    # neighborhood has at least one other segment in the neighborhood where:
    #    (1) the y-intercept of the two segments are no more than neighborly_distance apart. The intercept for a 
    #        segment is calculated as reaction_start - base_start.
    #    (2) there is overlap in (base_start1, base_end1) and (base_start2, base_end2) of the two respective 
    #        segments. 

    def overlap(seg1, seg2):
        # Calculate overlap on base axis
        base_overlap_start = max(seg1['end_points'][2], seg2['end_points'][2])
        base_overlap_end = min(seg1['end_points'][3], seg2['end_points'][3])
        base_overlap_length = max(0, base_overlap_end - base_overlap_start)
        
        seg1_base_length = seg1['end_points'][3] - seg1['end_points'][2]
        seg2_base_length = seg2['end_points'][3] - seg2['end_points'][2]
        shortest_base_length = min(seg1_base_length, seg2_base_length)
        
        # Calculate overlap on reaction axis
        reaction_overlap_start = max(seg1['end_points'][0], seg2['end_points'][0])
        reaction_overlap_end = min(seg1['end_points'][1], seg2['end_points'][1])
        reaction_overlap_length = max(0, reaction_overlap_end - reaction_overlap_start)
        
        seg1_reaction_length = seg1['end_points'][1] - seg1['end_points'][0]
        seg2_reaction_length = seg2['end_points'][1] - seg2['end_points'][0]
        shortest_reaction_length = min(seg1_reaction_length, seg2_reaction_length)
        
        # Check if both overlaps are at least 25% of the shortest segment
        base_condition = base_overlap_length >= 0.25 * shortest_base_length
        reaction_condition = reaction_overlap_length >= 0.25 * shortest_reaction_length
        
        return base_condition and reaction_condition

    # helper function to check the y-intercept difference criterion
    def close_enough(seg1, seg2, sr):
        intercept1 = seg1['end_points'][0] - seg1['end_points'][2]
        intercept2 = seg2['end_points'][0] - seg2['end_points'][2]
        return abs(intercept1 - intercept2) <= neighborly_distance


    neighbors = []

    for i, seg1 in enumerate(segments):
        neighborhood = [i]  # Use index instead of the segment itself
        for j, seg2 in enumerate(segments):
            if i != j and overlap(seg1, seg2) and close_enough(seg1, seg2, sr):
                neighborhood.append(j)  # Append index
        neighbors.append(neighborhood)

    # Merge neighborhoods if they share a segment index
    merged = True
    while merged:
        merged = False
        new_neighbors = []

        while neighbors:
            current = neighbors.pop()
            merged_with_another = False
            for other in neighbors:
                if set(current).intersection(set(other)):  # Since we use indices, this is valid now
                    merged_with_another = True
                    neighbors.remove(other)
                    current = current + [o for o in other if o not in current]
                    break
            
            new_neighbors.append(current)
            merged |= merged_with_another

        neighbors = new_neighbors

    # Convert segment indices back to segments for final output
    neighborhoods = [[segments[i] for i in neighborhood] for neighborhood in neighbors if len(neighborhood) > 1]

    return neighborhoods


def prune_neighbors(reaction, segments, allowed_spacing):
    neighborly_distance = 4 * sr 

    segments = [s for s in segments if not s.get('pruned', False)]

    neighborhoods = get_neighborhoods(segments, neighborly_distance)


    # Now we're going to eliminate each neighbor if:
    #  1) there exists a different neighbor with base_start less than and base_end
    #     greater than the neighbor.
    #  2) the two neighbors are within neighborly_distance intercept of each other
    #  3) the other neighbor has a better score in the range of overlap

    to_eliminate = []
    for neighborhood in neighborhoods:
        
        for i, segment in enumerate(neighborhood):
            reaction_start, reaction_end, base_start, base_end = segment['end_points']
            for j, segment2 in enumerate(neighborhood):
                if i == j: 
                    continue
                reaction_start2, reaction_end2, base_start2, base_end2 = segment2['end_points']

                base_encompassed = base_start2 <= base_start and base_end2 >= base_end
                close_enough =  abs( (reaction_start - base_start) - (reaction_start2 - base_start2)  ) < neighborly_distance

                if base_encompassed and close_enough: 
                    diff_front = base_start - base_start2
                    subsegment = (reaction_start2 + diff_front, reaction_start2 + diff_front + base_end - base_start, base_start, base_end)  # subsegment of segment2 that overlaps with segment
                    score = get_segment_mfcc_cosine_similarity_score(reaction, segment['end_points'])
                    big_neighbor_score = get_segment_mfcc_cosine_similarity_score(reaction, subsegment)

                    # print('Considering!', score, big_neighbor_score)
                    # print(f"\t  Neighbor ({segment['end_points'][2]/sr:.1f}, {segment['end_points'][3]/sr:.1f}), ({segment['end_points'][0]/sr:.1f}, {segment['end_points'][1]/sr:.1f})")                    
                    # print(f"\tSubsegment ({subsegment[2]/sr:.1f}, {subsegment[3]/sr:.1f}), ({subsegment[0]/sr:.1f}, {subsegment[1]/sr:.1f})")
                    if score < big_neighbor_score:
                        to_eliminate.append(segment)
                        segment['pruned'] = True
                        break


    for neighborhood in neighborhoods:
        # print(f"Neighborhood #{len(neighborhood)}")
        for segment in neighborhood:
            end_points = segment['end_points']
            # if segment in to_eliminate:
            #     print(f"\t*** Segment ({end_points[2]/sr:.1f}, {end_points[3]/sr:.1f}), ({end_points[0]/sr:.1f}, {end_points[1]/sr:.1f})")
            # else: 
            #     print(f"\tSegment ({end_points[2]/sr:.1f}, {end_points[3]/sr:.1f}), ({end_points[0]/sr:.1f}, {end_points[1]/sr:.1f})")



    to_keep = [s for s in segments if s not in to_eliminate]
    print(f"Pruned neighbors: kept {len(to_keep)} of {len(segments)} segments to start")

    prune_cache['neighbor'] += len(to_eliminate)
    # visualize_neighborhoods(segments, neighborhoods)

    return to_keep


def visualize_neighborhoods(segments, neighborhoods):
    plt.figure(figsize=(10,10))
    
    # List of colors for visualization
    colors = plt.cm.tab10.colors
    num_colors = len(colors)
    
    # First, plot all segments in gray
    for segment in segments:
        end_points = segment['end_points']
        plt.plot([end_points[2]/sr, end_points[3]/sr], [end_points[0]/sr, end_points[1]/sr], color='gray', linewidth=.5)
    
    # Next, overlay segments that belong to neighborhoods with unique colors
    for idx, neighborhood in enumerate(neighborhoods):
        color = colors[idx % num_colors]
        for segment in neighborhood:
            end_points = segment['end_points']
            plt.plot([end_points[2]/sr, end_points[3]/sr], [end_points[0]/sr, end_points[1]/sr], color=color, linewidth=3)
    
    plt.xlabel('Base Time')
    plt.ylabel('Reaction Time')
    plt.title('Visualization of Neighborhoods')
    plt.grid(True)
    plt.show()



# def merge_continuous_segments_separated_by_filler(reaction, path, strokes):
#     new_path = []
#     for idx, segment in enumerate(path):
#         reaction_start, reaction_end, base_start, base_end, is_filler, strokes_key = segment
#         # print(f"COLLAPSING FOR {base_start/sr}-{base_end/sr}")

#         # if strokes_key is not None:
#         #     my_strokes = strokes[strokes_key]

#         #     relevant_strokes = []
#         #     for stroke in my_strokes['strokes']:
#         #         if stroke[2] >= base_start and stroke[3] <= base_end:
#         #             relevant_strokes.append(stroke)
#         #         else: 
#         #             print(f"\tRemoving: {stroke[2]/sr}-{stroke[3]/sr}")


#         #     strokes[strokes_key]['strokes'] = relevant_strokes

#         extended = False
#         if idx > 0 and idx < len(path) - 1:
#             if is_filler and not path[idx-1][-1] and not path[idx+1][-1]: # if this segment is filler surrounded by non-filler...
#                 # and prior segment and later segment are continuous with each other...
#                 prior_segment = path[idx-1]
#                 later_segment = path[idx+1]
#                 intercept_prior = prior_segment[0] - prior_segment[2]
#                 intercept_later = later_segment[0] - later_segment[2]

#                 if abs( intercept_prior - intercept_later ) / sr < .1:
#                     # remove filler, merge prior into later
#                     new_path.pop()
#                     later_segment[0] = prior_segment[0]
#                     later_segment[2] = prior_segment[2]
#                     later_segment[1] = prior_segment[0] + later_segment[3] - later_segment[2]

#                     aaa = len(strokes[later_segment[5]])
#                     strokes[later_segment[5]].extend(strokes[prior_segment[5]])
#                     # print(f"EXTENDING!!!!! Strokes from {aaa} to {len(strokes[later_segment[5]])} by adding {len(strokes[prior_segment[5]])}", )

#                     extended = True

#         if not extended:
#             new_path.append(segment)

#     return new_path


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

def micro_align_path(reaction, path, strokes):

    # print("Microaligning:")
    # print_path(path, reaction)


    reaction_len = len(reaction.get('reaction_audio_data'))
    new_path = []
    for idx, segment in enumerate(path):
        reaction_start, reaction_end, base_start, base_end, is_filler, strokes_key = segment
        if is_filler:
            new_path.append(segment)
            continue


        if True:

            my_strokes = strokes[strokes_key]

            if len(my_strokes['strokes']) == 0:
                new_path.append(segment)
                continue

            scatter = []
            min_intercept = None
            max_intercept = None
            for stroke in my_strokes['strokes']:
                if stroke[2] >= base_start - 3 * sr and stroke[3] <= base_end + 3 * sr:
                    x = stroke[2]
                    b = stroke[0] - stroke[2]

                    scatter.append( (x,b,stroke)  )

                    if min_intercept is None or min_intercept > b:
                        min_intercept = b

                    if max_intercept is None or max_intercept < b:
                        max_intercept = b

            scatter.sort(key=lambda x: x[0])

            if min_intercept is None or max_intercept is None:
                new_path.append(segment)
                continue

            # my_range = max_intercept - min_intercept
            # if my_range < .05 * sr:
            #     new_path.append(segment)
            #     continue

        else: 
            scatter = []
            my_strokes = {
                'strokes': []
            }
            chunk_size = get_chunk_size(reaction)
            step = int(.5 * sr)
            padding = 2 * step

            starting_points = range(base_start, base_end - chunk_size, step)
            for i,start in enumerate(starting_points):
                d = start - base_start
                r_start = reaction_start + d - padding
                r_end = r_start + 2 * padding

            
                signals, evaluate_with = get_signals(reaction, start, r_start, chunk_size)

                candidates = get_candidate_starts(reaction, signals, peak_tolerance=.7, open_start=r_start, closed_start=start, chunk_size=chunk_size, distance=1 * sr, upper_bound=r_end, evaluate_with=evaluate_with)

                candidates = [c + r_start + padding for c in candidates if c + r_start + padding >= reaction_start + d and c + r_start + padding + chunk_size <= reaction_end + d  ]

                y1 = candidates[0] + r_start + padding

                y2 = y1+chunk_size
                x1 = start
                x2 = start+chunk_size

                new_stroke = (y1,y2,x1,x2)

                b = y1 - x1
                scatter.append( (x1, b, new_stroke)  )

                scatter.sort(key=lambda x: x[0])

                my_strokes['strokes'].append(new_stroke)




        # Extract the x and y values for DBSCAN
        X = [(x, b) for x, b, _ in scatter]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply DBSCAN. You might need to adjust the 'eps' and 'min_samples' parameters based on your data's nature.
        db = DBSCAN(eps=.25, min_samples=8).fit(X_scaled)

        # Labels assigned by DBSCAN to each point in scatter. -1 means it's an outlier.
        labels = db.labels_

        misc_subsegment = []

        unique_labels = np.unique(labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        subsegments = [[] for _ in unique_labels]

        for i, label in enumerate(labels):
            if label == -1:
                misc_subsegment.append(scatter[i])
            else:
                idx = label_to_idx[label]
                subsegments[idx].append(scatter[i])

        subsegments = [s for s in subsegments if s and len(s) > 0]



        # Break up a subsegment into new subsegments when there is a gap of at 
        # least 3 seconds on the x-axis amongst the strokes in the subsegment.
        new_subsegments = []

        for subsegment in subsegments:
            # Sort the strokes in the subsegment by their start time
            subsegment = sorted(subsegment, key=lambda x: x[0])
            
            # This list will temporarily store strokes of the current fragment of the subsegment
            current_fragment = [subsegment[0]]

            for i in range(1, len(subsegment)):
                # If there's a gap between the current and the previous stroke...
                if subsegment[i][0] - subsegment[i-1][0] >= 4 * sr:
                    # ... then finalize the current fragment as a new subsegment
                    new_subsegments.append(current_fragment)
                    # ... and start a new fragment for the next strokes
                    current_fragment = []

                # Add the current stroke to the current fragment
                current_fragment.append(subsegment[i])

            # Finalize the last fragment after exiting the loop
            if current_fragment:
                new_subsegments.append(current_fragment)

        # Replace the original subsegments with the new ones
        subsegments = new_subsegments

        if len(subsegments) == 0:
            new_path.append(segment)
            continue

        # if len(subsegments) == 1:
        #     subsegment = subsegments[0]
        #     start_x, intercept, _ = subsegment[0]
        #     end_x, _, _ = subsegment[-1]
        #     new_subsegments = [
        #         [],
        #         [],
        #         [],
        #         [],
        #     ]

        #     for stroke in subsegment:
        #         idx = min(3, math.floor(  4 * (stroke[0] - start_x) / (end_x - start_x)   ))

        #         # stroke = list(stroke)
        #         # import random
        #         # variation = random.randint(-int(sr / 5), int(sr / 5) )
        #         # stroke[0] += variation
        #         # stroke[1] += variation

        #         new_subsegments[idx].append(stroke)

        #     subsegments = [s for s in new_subsegments if len(s) > 0]



        # Filter out clusters that don't span at least 3 seconds
        filtered_subsegments = []
        for subsegment in subsegments:
            start_x, _, _ = subsegment[0]
            end_x, _, _ = subsegment[-1]

            if end_x - start_x < 3 * sr:
                misc_subsegment.extend(subsegment)
            else: 
                filtered_subsegments.append(subsegment)

        subsegments = filtered_subsegments
        # lines = find_representative_lines(subsegments)


        # Fit lines to the clusters
        lines = []
        intercept = reaction_start - base_start

        for i,subsegment in enumerate(subsegments):
            subsegment.sort(key=lambda x: x[0])

            intercepts = [point[1] for point in subsegment]

            subsegment_start = subsegment[0][2][2]
            subsegment_end   = subsegment[-1][2][3]


            subsegment_reaction_start = max(0,            reaction_start + (subsegment_start - base_start) - int(sr/2))
            subsegment_reaction_end   = min(reaction_len, reaction_end   + (subsegment_end   - base_end)   + int(sr/2))

            intercepts.append(intercept)
            # print(f"{base_start / sr}-{base_end / sr}  {subsegment_start / sr}-{subsegment_end / sr}  {subsegment_reaction_start / sr}-{subsegment_reaction_end / sr} {(subsegment_end - subsegment_start) / sr} {(max_intercept - min_intercept) / sr}")
            best_intercept = find_best_intercept(reaction, intercepts, subsegment_start, subsegment_end, include_cross_correlation=False, reaction_start=subsegment_reaction_start, reaction_end=subsegment_reaction_end)
                
            lines.append( (best_intercept, subsegment, subsegment_start, subsegment_end)   )


        lines.sort(key=lambda x: x[1][0])

        if len(new_path) == 0:
            previous_end = 0
        else: 
            previous_end = new_path[-1][3]

        default_intercept = reaction_start - base_start
        sub_path = []
        if len(lines) == 0:
            new_path.append(segment)
            continue

        # create the new path
        for i, (intercept, subsegment, min_x, max_x) in enumerate(lines):
            min_x = max(min_x, previous_end, base_start)

            if min_x > previous_end + 1:
                my_reaction_start = reaction_start + (previous_end - base_start)
                my_reaction_end   = my_reaction_start + (min_x - previous_end)
                fill_intercept = find_best_intercept(reaction, [default_intercept], previous_end, min_x, include_cross_correlation=True, reaction_start=my_reaction_start, reaction_end=my_reaction_end, print_intercepts=False)
                sub_path.append( [previous_end + fill_intercept, min_x + fill_intercept, previous_end, min_x, False ]        )


            if i < len(lines) - 1:
                max_x = min(max_x, lines[i+1][2], base_end)
            else: 
                max_x = min(max_x, base_end)

            sub_path.append( [min_x + intercept, max_x + intercept, min_x, max_x, False ] )


            if i == len(lines) - 1 and max_x < base_end:
                fill_intercept = find_best_intercept(reaction, [default_intercept], max_x, base_end, include_cross_correlation=True, reaction_start=reaction_start + (max_x - base_start), reaction_end=reaction_end, print_intercepts=False)

                sub_path.append( [max_x + fill_intercept, base_end + fill_intercept, max_x, base_end, False ]        )

            previous_end = max_x



        # Sharpen up the edges of the new path
        sharpened_sub_path = sharpen_path_boundaries(reaction, sub_path)

        # commit the new path to the overall path
        new_path += sharpened_sub_path

        if False: 

            x_vals = []  # to store starts for plotting
            y_vals = []  # to store intercepts for plotting


            for stroke in my_strokes['strokes']:
                if stroke[2] >= base_start - 3 * sr and stroke[3] <= base_end + 3 * sr:

                    b = stroke[0] - stroke[2]
                    # Collecting data for plotting
                    x_vals.append(stroke[2]/sr)
                    y_vals.append(b)

            # Define a list of colors for each subsegment. Make sure there are enough colors for all subsegments.
            colors = cm.rainbow(np.linspace(0, 1, len(subsegments)))

            # Plot each subsegment with its unique color
            for idx, subsegment in enumerate(subsegments):
                subsegment_x_vals = [s[0]/sr for s in subsegment]
                subsegment_y_vals = [s[1] for s in subsegment]
                plt.scatter(subsegment_x_vals, subsegment_y_vals, marker='o', color=colors[idx], label=f'Subsegment {idx + 1}')

            # Plot the misc_subsegment with a distinguishable color, like black
            misc_x_vals = [s[0]/sr for s in misc_subsegment]
            misc_y_vals = [s[1] for s in misc_subsegment]
            if misc_x_vals:  # Only plot if there are any miscellaneous points
                plt.scatter(misc_x_vals, misc_y_vals, marker='o', color='black', label='Miscellaneous')

            # Plotting best_line_def as a green line
            plt.plot([x_vals[0], x_vals[-1]], [reaction_start - base_start, reaction_start - base_start], color='green', label='Best Line')

            for intercept, subsegment, min_x, max_x in lines:
                stroke_length = subsegment[0][2][1] - subsegment[0][2][0]
                min_x = min( [p[0] for p in subsegment]  )
                max_x = max( [p[0] for p in subsegment]  ) + stroke_length
                # plt.plot([min_x / sr, max_x / sr], [intercept, intercept], color='red')

            for rs,re,bs,be,filler in sub_path:
                plt.plot([bs / sr, be / sr], [rs-bs, rs-bs], linewidth=3, color='orange', label='Best Line')

            for rs,re,bs,be,filler in sharpened_sub_path:
                plt.plot([bs / sr, be / sr], [rs-bs, rs-bs], linewidth=3, color='purple', label='Sharpened Best Line')

            plt.title("Intercept vs. Start of Strokes")
            plt.xlabel("Start of Stroke")
            plt.ylabel("Intercept")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            plt.show()


    return new_path


# def finesse_paths(reaction, paths, strokes):
#     merged_paths = []
#     for path in paths:
#         merged_path = merge_continuous_segments_separated_by_filler(reaction, path, strokes)
#         merged_paths.append(merged_path)

#     return merged_paths



#########################################
# Getting rid of clusters of bad segments
# 1) Identify clusters of candidate segments where there are 5 or more segments such 
# that each segment overlaps each other segment in the cluster by 80% or more in their 
# position in the song (base_start, base_end)

# 2) For each cluster, use get_segment_mfcc_cosine_similarity_score to score each 
# candidate. Mark the ones that fall below 50% of the best cluster score as 
# prunable.

def get_overlap_percentage(segment1, segment2):
    # Assuming segments are tuples/lists of (base_start, base_end)
    overlap_start = max(segment1[0], segment2[0])
    overlap_end = min(segment1[1], segment2[1])
    overlap_duration = max(0, overlap_end - overlap_start)
    segment1_duration = segment1[1] - segment1[0]
    return (overlap_duration / segment1_duration) if segment1_duration > 0 else 0

def find_clusters(segments):
    clusters = []

    overlap_threshold = .7
    for current_segment in segments:
        found_cluster = False
        current_segment_range = (current_segment['end_points'][2], current_segment['end_points'][3])
        for cluster in clusters:
            if all(get_overlap_percentage(current_segment_range, (seg['end_points'][2], seg['end_points'][3])) >= overlap_threshold for seg in cluster) and \
               all(get_overlap_percentage((seg['end_points'][2], seg['end_points'][3]), current_segment_range) >= overlap_threshold for seg in cluster):
                cluster.append(current_segment)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([current_segment])
    return [cluster for cluster in clusters if len(cluster) >= 5]

def prune_low_quality_segments(reaction, segments):
    # Step 1: Cluster Identification

    segments = [s for s in segments if not s.get('pruned')]
    clusters = find_clusters(segments)

    # Step 2 and 3: Scoring and Pruning
    for cluster in clusters:
        for segment in cluster:
            segment['score'] = get_segment_mfcc_cosine_similarity_score(reaction, segment['end_points'])
            segment['pruned'] = False  # Assume not pruned initially

        # Find the best score in the cluster
        best_score = max(segment['score'] for segment in cluster)
        
        # Prune segments scoring less than 50% of the best score
        for segment in cluster:
            if segment['score'] < 0.5 * best_score:
                segment['pruned'] = True

    # Visualize the clusters
    base_audio_len = len(conf['song_audio_data'])
    reaction_audio_len = len(reaction['reaction_audio_data'])
    # visualize_clusters(clusters, base_audio_len, reaction_audio_len)

    # Filter out the pruned segments
    final_segments = [segment for cluster in clusters for segment in cluster if not segment['pruned']]

    print(f"Kept {len(final_segments)} of {len(segments)} segments after pruning")
    return final_segments


import matplotlib.colors as mcolors

def visualize_clusters(clusters, base_audio_len, reaction_audio_len):
    # Create a colormap from red (0) to green (1)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "green"])
    
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Iterate over each cluster
    for cluster in clusters:
        # Get the best score within the cluster for color scaling
        best_score = max(segment['score'] for segment in cluster)

        # Plot each segment in the cluster
        for segment in cluster:
            base_start, base_end = segment['end_points'][2], segment['end_points'][3]
            reaction_start, reaction_end = segment['end_points'][0], segment['end_points'][1]
            score = segment['score']
            is_pruned = segment.get('pruned', False)

            # Scale the color by the score relative to the best score in the cluster
            color = cmap(score / best_score)
            line_style = "--" if is_pruned else "-"

            # Draw the segment
            ax.plot([base_start, base_end], [reaction_start, reaction_end], 
                    linestyle=line_style, color=color, linewidth=2)

    # Set plot limits and labels
    ax.set_xlim([0, base_audio_len])
    ax.set_ylim([0, reaction_audio_len])
    ax.set_xlabel('Position in Base Audio')
    ax.set_ylabel('Position in Reaction Audio')

    # Show the plot
    plt.show()





#######

def sharpen_path_boundaries(reaction, original_path):
    sharpener_width = 1 * sr
    step_width = .01 * sr

    edge = 3 * sr

    path = copy.deepcopy(original_path)

    path_reaction_start = path[0][0]
    path_base_start = path[0][2]
    path_reaction_end = path[-1][1]
    path_base_end = path[-1][3]

    for i, segment1 in enumerate(path):
        if i < len(path) - 1:
            # Finding the best separator between the coarse segment1 and segment2 definition
            segment2 = path[i+1]
            (reaction_start1, reaction_end1, base_start1, base_end1, fill1) = segment1
            (reaction_start2, reaction_end2, base_start2, base_end2, fill2) = segment2

            b1 = reaction_start1 - base_start1
            b2 = reaction_start2 - base_start2

            start = max(base_start1, base_end1 - edge)
            score1 = []
            score2 = []
            time_x = []



            # Finding the scores of each segment in the border region 
            while(start < min(base_start2 + edge, base_end2 - sharpener_width)):
                end = start + sharpener_width
                section1 = (start+b1, end+b1, start, end)
                section2 = (start+b2, end+b2, start, end)

                s1 = get_segment_mfcc_cosine_similarity_score(reaction, section1)
                s2 = get_segment_mfcc_cosine_similarity_score(reaction, section2)
                score1.append(s1)
                score2.append(s2)
                time_x.append(start)

                start += step_width

            if len(score1) == 0 or len(score2) == 0:
                continue

            # Identify a segmentation point (a value in time_x) such that the scores at times less 
            # than the segmentation point are generally higher for segment1, and the scores at 
            # times greater than the segmentation point are generally higher for segment 2.

            # Create cumulative sums
            cumulative_score1 = np.cumsum(score1)
            cumulative_score2 = np.cumsum(score2)

            # Total area under the curves
            total_area_score1 = cumulative_score1[-1]
            total_area_score2 = cumulative_score2[-1]

            potential_transition_points = []
            for i, t in enumerate(time_x):
                # Calculate remaining areas under the curves
                remaining_area_score1 = total_area_score1 - cumulative_score1[i]
                remaining_area_score2 = total_area_score2 - cumulative_score2[i]
                
                # Check if both conditions are met
                if score2[i] > score1[i] and remaining_area_score2 > remaining_area_score1:
                    potential_transition_points.append(i)

            # If no transition points were found, use the last point
            if not potential_transition_points:
                segmentation_point = time_x[-1]
            else:
                # Find the transition point that maximizes the combined difference in cumulative scores
                max_combined_diff = float('-inf')
                for idx in potential_transition_points:
                    diff_before_transition = cumulative_score1[idx] - cumulative_score2[idx]
                    diff_after_transition = (total_area_score2 - cumulative_score2[idx]) - (total_area_score1 - cumulative_score1[idx])
                    
                    combined_diff = diff_before_transition + diff_after_transition
                    if combined_diff > max_combined_diff:
                        max_combined_diff = combined_diff
                        segmentation_point = time_x[idx]


            if False: 
                # Plot score1 and score2 as y vals of two lines, with time_x being the x axis. 
                # Draw a vertical dashed line at the segmentation point. 
                plt.plot([t/sr for t in time_x], score1, label='Segment 1 Scores')
                plt.plot([t/sr for t in time_x], score2, label='Segment 2 Scores')

                plt.axvline(x=segmentation_point/sr, color='r', linestyle='--', label='Segmentation Point')
                plt.xlabel('Time (s)')
                plt.ylabel('MFCC Cosine Similarity Score')
                plt.legend()
                plt.show()


            # Now we'll sharpen up the path definition
            segmentation_point = int(segmentation_point)

            assert(segmentation_point >= path_base_start, f"Segmentation point at {segmentation_point / sr} is less than than {path_base_start/sr}", i, segment1, segment2, path)
            assert(segmentation_point <= path_base_end, f"Segmentation point at {segmentation_point / sr} is greater than {path_base_end/sr}", i, segment1, segment2, path)

            # min(path_base_end, max(path_base_start, int(segmentation_point)))

            segment1[1] = min(path_reaction_end, b1 + segmentation_point) # new reaction_end
            segment2[0] = max(path_reaction_start, b2 + segmentation_point) # new reaction_start
            segment1[3] = min(path_base_end, segmentation_point) # new base_end
            segment2[2] = max(path_base_start, segmentation_point) # new base_start

            # segment1[1] = b1 + segmentation_point # new reaction_end
            # segment2[0] = b2 + segmentation_point # new reaction_start
            # segment1[3] = segmentation_point # new base_end
            # segment2[2] = segmentation_point # new base_start


    removed_segment = False
    completed_path = []
    for s in path:
        if s[3] > s[2]:
            completed_path.append(s)
        else:
            print(f"************\nREMOVED SEGMENT!!!!! {s[0]/sr} {s[1]/sr} {s[2]/sr} {s[3]/sr}")
            removed_segment = True

    if removed_segment:
        
        return sharpen_path_boundaries(reaction, completed_path)

    else: 
        return path




def construct_all_paths(reaction, segments, joinable_segment_map, allowed_spacing):
    paths = []
    song_length = len(conf.get('song_audio_data'))
    partial_paths = []


    def complete_path(path):
        completed_path = copy.deepcopy(path)

        fill_len = song_length - path[-1][3]
        if fill_len > 0:

            if True: # fill_len < sr: 
                path[-1][3] += fill_len
                path[-1][1] += fill_len
            else:
                completed_path.append( [path[-1][1], path[-1][1] + fill_len, path[-1][3], path[-1][3] + fill_len, True, None] ) 

        score = path_score_by_mfcc_cosine_similarity(path, reaction) # path_score(path, reaction)
        if best_score_cache['best_overall_path'] is None or best_score_cache['best_overall_score'] < score:
            best_score_cache['best_overall_path'] = path 
            best_score_cache['best_overall_score'] = score

            print("\nNew best score!")
            print_path(path, reaction)

        if score > .9 * best_score_cache['best_overall_score']:
            paths.append(completed_path)


    for c in segments:
        if c.get('pruned', False):
            continue

        if near_song_beginning(c, allowed_spacing):
            reaction_start, reaction_end, base_start, base_end = c['end_points']
            start_path = [ [reaction_start, reaction_end, base_start, base_end, False, c['key']] ]
            if base_start > 0:
                start_path.insert(0, (reaction_start - base_start, reaction_start, 0, base_start, True, None) )


            score = path_score_by_mfcc_cosine_similarity(start_path, reaction) #path_score(start_path, reaction, end=start_path[-1][1])
            partial_paths.append( [[ start_path, c ], score] )

            if near_song_end(c, allowed_spacing): # in the case of one long completion
                complete_path(start_path)

    i = 0 
    start_time = time.perf_counter()
    was_prune_eligible = False 
    backlog = []
    while len(partial_paths) > 0:

        i += 1
        prune_eligible = time.perf_counter() - start_time > 3 * 60 #True #len(partial_paths) > 100 #or len(paths) > 10000

        if i < 50000:
            sort_every = 2500
        elif i < 200000:
            sort_every = 15000
        elif i < 500000:
            sort_every = 45000
        else:
            sort_every = 75000

        if (len(partial_paths) < 10 or i % sort_every == 4900) and len(backlog) > 0:
            partial_paths.extend(backlog)
            backlog.clear()


            if prune_eligible and not was_prune_eligible:
                print("\nENABLING PRUNING\n")
                iii = len(partial_paths) - 1

                for partial, score in reversed(partial_paths):
                    if False and is_path_quality_poor(reaction, partial[0]):
                        prune_cache['poor_path'] += 1
                        partial_paths.pop(iii)
                    else:
                        partial_path = []
                        for segment in partial[0]:
                            partial_path.append(segment)
                            should_prune, score = should_prune_path(reaction, partial_path, song_length)
                            if should_prune:
                                partial_paths.pop(iii)
                                break
                    iii -= 1
                was_prune_eligible = True
                continue


        if i % 10 == 1 or len(partial_paths) > 1000:
            partial_paths.sort(key=lambda x: x[1], reverse=True)
            if len(partial_paths) > 200:
                backlog.extend( partial_paths[200:]  )
                del partial_paths[200:]



        if i % 1000 == 999:
            # print_prune_data()
            print_profiling()


        partial_path, score = partial_paths.pop(0)

        # print(len(partial_paths), end='\r')

        should_prune, score = should_prune_path(reaction, partial_path[0], song_length, score)
        
        if should_prune: # and prune_eligible:
            continue

        print(i, len(partial_paths) + len(backlog), len(paths), len(partial_path[0]), end='\r')

        if partial_path[1]['key'] in joinable_segment_map:
            next_partials = branch_from( reaction, partial_path, joinable_segment_map[partial_path[1]['key']], allowed_spacing  )

            for partial in next_partials:
                path, last_segment = partial
                if near_song_end(last_segment, allowed_spacing):
                    complete_path(path)


                should_prune, score = should_prune_path(reaction, path, song_length)
                if not should_prune and prune_eligible and is_path_quality_poor(reaction, path):
                    prune_cache['poor_path'] += 1
                    should_prune = True

                if (not should_prune) and last_segment['key'] in joinable_segment_map and len(joinable_segment_map[last_segment['key']]) > 0:
                    partial_paths.append([partial, score])

    return paths



location_cache = {}
best_score_cache = {}
prune_cache = {}

def initialize_paint_caches():

    location_cache.clear()
    best_score_cache.clear()
    best_score_cache.update({'best_overall_path': None, 'best_overall_score': None})
    prune_cache.clear()
    prune_cache.update({
        'location': 0,
        'best_score': 0,
        'poor_path': 0,
        'poor_link': 0,

        'neighbor': 0,
        'unreachable': 0,
        'segment_quality': 0,
    })

def should_prune_path(reaction, path, song_length, score = None):

    reaction_end = path[-1][1]

    score = path_score_by_mfcc_cosine_similarity(path, reaction) #path_score(path, reaction, end=reaction_end)

    prune_for_location = should_prune_for_location(reaction, path, song_length, score)

    if prune_for_location:
        prune_cache['location'] += 1
        return True, score

    return False, score

    if reaction_end / sr > 20 and best_score_cache['best_overall_path'] is not None:
        
        __, truncated_path = truncate_path(best_score_cache['best_overall_path'], end=reaction_end)
        # print(best_score_cache['best_overall_path'], truncated_path, reaction_end)
        best_score_here = path_score_by_mfcc_cosine_similarity(truncated_path, reaction)  # path_score(best_score_cache['best_overall_path'], reaction, end=reaction_end)

        prune_threshold = .7 + path[-1][3] / song_length * .2

        prune = best_score_here * prune_threshold > score

        if prune: 
            prune_cache['best_score'] += 1
            return True, score

    return False, score






def should_prune_for_location(reaction, path, song_length, score):
    global location_cache

    last_segment = path[-1]
    location_key = str(last_segment)
    segment_quality_threshold = .750

    has_bad_segment = False
    for segment in path:
        if not segment[-1]:
            segment_score = get_segment_mfcc_cosine_similarity_score(reaction, segment)
            if segment_score < segment_quality_threshold:
                has_bad_segment = True
                break

    location_prune_threshold = .7 #.7 + .25 * last_segment[3] / song_length
    if has_bad_segment: 
        location_prune_threshold = .99

    if location_key in location_cache:
        best_score = location_cache[location_key]
        if score >= best_score:
            location_cache[location_key] = score
        else: 
            prunable = location_prune_threshold * best_score > score
            if prunable:
                # print('\npruned!', best_score[2], score[2])
                return True
    else: 
        location_cache[location_key] = score

    return False



def branch_from( reaction, partial_path, joinable_segments, allowed_spacing ):
    branches = []

    current_path, last_segment = partial_path

    reaction_start, reaction_end, base_start, base_end, is_filler, strokes = current_path[-1]

    b_current = reaction_end - base_end  # as in, y = ax + b

    for candidate in joinable_segments:

        candidate_reaction_start, candidate_reaction_end, candidate_base_start, candidate_base_end = candidate['end_points']

        distance = base_end - candidate_base_start
        if candidate_base_end > base_end and distance > -allowed_spacing:
            branch = copy.copy(current_path)

            if distance < 0:
                branch.append( [reaction_end, reaction_end - distance, base_end, base_end - distance, True, None] )
                filled = -distance
            else: 
                filled = 0

            b_candidate = candidate_reaction_end - candidate_base_end

            branch.append( [reaction_end + filled + b_candidate - b_current, candidate_reaction_end, base_end + filled, candidate_base_end, False, candidate['key']]      )

            branches.append( [branch, candidate]  )

    return branches






def prune_unreachable_segments(reaction, segments, allowed_spacing, prune_links = False): 

    segments = [s for s in segments if not s.get('pruned', False)]
    
    starting_segment_num = len(segments)
    segments.sort( key=lambda x: x['end_points'][1]  )  # sort by reaction_end

    joinable_segments = {}

    for segment in segments:
        segment_id = segment['key']
        joins = joinable_segments[segment_id] = get_joinable_segments(reaction, segment, segments, allowed_spacing, prune_links = prune_links)

        at_end = segment['at_end'] = near_song_end(segment, allowed_spacing)

        if len(joins) == 0 and not at_end:
            # if prune_links: 
            #     print("PRUNING BECAUSE NO JOINS FOUND", segment_id)
            #     song_length = len(conf.get('song_audio_data'))
            #     print(song_length/sr, (song_length - segment['end_points'][3])/sr, allowed_spacing/sr)


            segment['pruned'] = True
            del joinable_segments[segment_id]
            prune_cache['unreachable'] += 1

    segments = [s for s in segments if not s.get('pruned', False)]

    # clean up segments that can't reach the end
    pruned_another = True
    while pruned_another:
        segments = [s for s in segments if not s.get('pruned', False)]

        pruned_another = False
        for segment in segments:
            segment_id = segment['key']
            if segment_id in joinable_segments:
                joins = joinable_segments[segment_id] = [s for s in joinable_segments[segment_id] if not s.get('pruned', False)]

                if len(joins) == 0 and not segment.get('at_end', False):
                    segment['pruned'] = True
                    del joinable_segments[segment_id]
                    pruned_another = True
                    prune_cache['unreachable'] += 1


                    # if prune_links: 
                    #     print("PRUNING BECAUSE CANT REACH END", segment_id)


    # now clean up all segments that can't be reached from the start
    pruned_another = True

    while pruned_another:
        pruned_another = False

        segments_reached = {}
        for segment_id, joins in joinable_segments.items():
            joins = joinable_segments[segment_id] = [s for s in joins if not s.get('pruned', False)]
            for s in joins:
                segments_reached[ s['key'] ] = True

        segments = [s for s in segments if not s.get('pruned', False)]

        for segment in segments:
            segment_id = segment['key']
            if segment_id not in segments_reached and not near_song_beginning(segment, allowed_spacing): 
                segment['pruned'] = True
                if segment_id in joinable_segments:
                    del joinable_segments[segment_id]
                pruned_another = True
                prune_cache['unreachable'] += 1


                # if prune_links: 
                #     print("PRUNING BECAUSE CANT REACH FROM STARET", segment_id)


    segments = [s for s in segments if not s.get('pruned', False)]

    # for k,v in joinable_segments.items():
    #     print(k,v)

    print(f"Pruned unreachable segments: {len(segments)} remaining of {starting_segment_num} to start")

    return joinable_segments





# returns all segments that a given segment could jump to
def get_joinable_segments(reaction, segment, all_segments, allowed_spacing, prune_links): 

    if segment.get('pruned', True):
        return []

    ay, by, ax, bx = segment['end_points']
    candidates = []
    for candidate in all_segments:
        if candidate.get('pruned', False):
            continue

        cy, dy, cx, dx = candidate['end_points']

        if ay==cy and by == dy and ax==cx and bx == dx:
            continue

        on_top = cy - cx > ay - ax  # solving for b in y=ax+b, where a=1
        over_each_other = bx + allowed_spacing >= cx -allowed_spacing and ax < dx


        extends_past = dx > bx   # Most of the time this is correct, but in the future we 
                                 # might want to support link joins that aren't from the end
                                 # of one segment. 

        if on_top and over_each_other and (extends_past or not prune_links):
            candidates.append(candidate)


    if prune_links:
        candidates = prune_poor_links(reaction, segment, candidates)
        

    return candidates




def get_link_score(reaction, link, reaction_start, base_start):
    link_segment = copy.copy(link['end_points'])

    filler = 0
    if link_segment[0] > reaction_start: # if we need filler
        filler = link_segment[0] - reaction_start
        # link_segment[0] += filler
        # link_segment[2] += filler

    link_segment[0] = max(reaction_start, link_segment[0])
    link_segment[2] = max(base_start, link_segment[2])
    
    filler_penalty = 1 - filler / (link_segment[3] - link_segment[2])
    score_link = filler_penalty *   get_segment_mfcc_cosine_similarity_score(reaction, link_segment)
    return score_link

def prune_poor_links(reaction, segment, links):

    # Don't follow some branches when a prior branch looks substantially better 
    # and extends further into the future
    link_prune_threshold = .9

    good_links = []

    # ...for the link 
    reaction_start = segment['end_points'][1] 
    base_start = segment['end_points'][3]

    # sort links by reverse reaction_end
    # links.sort( key=lambda x: x['end_points'][1], reverse=True )


    for i,link in enumerate(links):
        prune = False
        score_link = get_link_score(reaction, link, reaction_start, base_start)
        link_intercept = link['end_points'][0] - link['end_points'][2]

        for j,pruning_link in enumerate(links):
            if j == i: 
                continue 

            pruning_link_intercept = pruning_link['end_points'][0] - pruning_link['end_points'][2]

            if link_intercept < pruning_link_intercept: # only prune links that come later
                continue

            if  pruning_link['end_points'][3] < link['end_points'][3]: 
                continue

            assert pruning_link['end_points'][3] > base_start

            # Now we want to compare the score of pruning_link over the scope
            # of link, to link, and if it is greater by a certain threshold, prune link.
            # One tricky thing is that we might need filler for one or both of them.

            score_pruning_link = get_link_score(reaction, pruning_link, reaction_start, base_start)

            if score_pruning_link * link_prune_threshold > score_link:
                prune = True
                prune_cache['poor_link'] += 1                    
                break

            
        if not prune:
            good_links.append(link)


    return good_links


from scipy.signal import correlate

def find_best_intercept(reaction, intercepts, base_start, base_end, include_cross_correlation=False, reaction_start=None, reaction_end=None, print_intercepts=False):
    unique_intercepts = {} # y-intercepts as in y=ax+b, solving for b when a=1

    if include_cross_correlation:
        # intercept_range = (min(intercepts), max(intercepts))

        song_data = conf.get('song_audio_data')
        reaction_data = reaction.get('reaction_audio_data')
        song_segment = song_data[base_start:base_end]

        width = (base_end - base_start)

        search_start = max(0, reaction_start - width)
        search_end   = min(len(reaction_data), reaction_end + width)
        reaction_segment = reaction_data[search_start:search_end]

        # Perform the correlation
        cross_corr = correlate(reaction_segment, song_segment, mode="valid")

        search_window = int(sr/2)

        # Calculate the start and end index in cross_corr corresponding to reaction_start and reaction_end
        corr_search_start = min(reaction_start, width) - search_window
        corr_search_end   = corr_search_start + 2 * search_window

        # Ensure the bounds are within the length of the cross_corr array
        corr_search_start = max(0, corr_search_start)
        corr_search_end = min(len(cross_corr) - 1, corr_search_end)

        # print(f"{corr_search_start/sr}-{corr_search_end/sr} (prev={(corr_search_start + (base_end - base_start) + search_window)/sr}) ({len(cross_corr)/sr})")

        # Find the maximum correlation value within the constrained window
        windowed_corr_max_index = np.argmax(cross_corr[corr_search_start:corr_search_end]) + corr_search_start

        # Calculate the starting index of the best match within the original reaction_data
        corr_reaction_start = search_start + windowed_corr_max_index

        # Ensure that corr_reaction_start is within the desired range
        # Since we are bounding our argmax, this should inherently be the case.
        corr_intercept = corr_reaction_start - base_start

        # print('corr', base_start / sr, base_end / sr, reaction_start/sr, reaction_end/sr, corr_reaction_start / sr, corr_intercept / sr, offset/sr, reaction_start <= corr_reaction_start <= reaction_end)        
        assert(  reaction_start <= corr_reaction_start <= reaction_end,  f"{reaction_start/sr} <= {corr_reaction_start/sr} <= {reaction_end/sr}"  )

        intercepts.append(corr_intercept)

    for b in intercepts:
        if not include_cross_correlation or reaction_start - .05 * sr <= b + base_start <= reaction_end + .05 * sr:
            unique_intercepts[b] = True
        # else: 
        #     print('intr', base_start / sr, base_end / sr, reaction_start/sr, reaction_end/sr, (b + base_start) / sr, b/sr, reaction_start <= b + base_start <= reaction_end)

        #     print("FILTERED!")


    best_line_def = None
    best_line_def_score = -1
    best_intercept = None
    for intercept in unique_intercepts.keys():

        intercept = int(intercept)

        candidate_line_def = [intercept + base_start, intercept + base_end, base_start, base_end]

        score = get_segment_mfcc_cosine_similarity_score(reaction, candidate_line_def)
        if include_cross_correlation and print_intercepts:
            if intercept == corr_intercept:
                print('*SCOR', score, base_start / sr, base_end / sr, reaction_start/sr, reaction_end/sr, (intercept+base_start) / sr, intercept / sr)        
            else: 
                print(' SCOR', score, base_start / sr, base_end / sr, reaction_start/sr, reaction_end/sr, (intercept+base_start) / sr, intercept / sr)        

        if score > best_line_def_score:
            best_line_def_score = score
            best_line_def = candidate_line_def
            best_intercept = intercept

    if False and include_cross_correlation:
        scores = []
        for intercept in unique_intercepts.keys():
            candidate_line_def = [intercept + base_start, intercept + base_end, base_start, base_end]
            score = get_segment_mfcc_cosine_similarity_score(reaction, candidate_line_def)
            scores.append(score)
    
        plot_scores_of_intercepts(list(unique_intercepts.keys()), scores, corr_intercept if include_cross_correlation else None, base_start, base_end)



    # print(f"\t{base_start / sr} - {base_end/sr}", best_intercept, len(intercepts) )

    return int(best_intercept)

def plot_scores_of_intercepts(intercepts, scores, corr_intercept, base_start, base_end):
    plt.figure(figsize=(10, 6))

    for intercept, score in zip(intercepts, scores):
        if intercept == corr_intercept:
            plt.scatter(intercept, score, c='red', marker='*', s=100)  # Mark the correlation-derived intercept in red
        else:
            plt.scatter(intercept, score, c='blue', marker='o')

    plt.title(f"Scores of Intercepts (base_start: {base_start/sr}, base_end: {base_end/sr})")
    plt.xlabel("Intercepts")
    plt.ylabel("Scores")
    plt.grid(True)
    plt.show()



def sharpen_intercept(reaction, chunk_size, step, segments, allowed_spacing):


    reaction_len = len(reaction.get('reaction_audio_data'))

    for segment in segments:
        if segment.get('pruned', False): 
            continue

        ###################################
        # if we have an imprecise segment composed of strokes that weren't exactly aligned, 
        # we'll want to find the best line through them
        # if segment.get('imprecise', False):
        # print("FINDING BEST INTERCEPT IN SHARPEN SEGMENTS")

        padding = int(sr/2)
        int_reaction_start = max(segment['end_points'][0] - padding, 0)
        int_reaction_end   = min(segment['end_points'][1] + padding, reaction_len)
        
        stroke_intercepts = [s[0]-s[2] for s in segment['strokes']]
        
        intercept = find_best_intercept(reaction, stroke_intercepts, segment['end_points'][2], segment['end_points'][3], include_cross_correlation=True, reaction_start=int_reaction_start, reaction_end=int_reaction_end)

        base_start = segment['end_points'][2]
        base_end = segment['end_points'][3]

        segment['old_end_points'] = segment['end_points']

        segment['end_points'] = [intercept + base_start, intercept + base_end, base_start, base_end]


        # else: 
        #     print(f"PRECISE ALIGNMENT {segment['end_points'][2]/sr}-{segment['end_points'][3]/sr}")


        if near_song_end(segment, allowed_spacing):
            continue

        # ###################################################################################
        # # Now we're going to a local correlation to see if we can improve precise alignment
        # adjustment_padding = sr
        # reaction_start, reaction_end, base_start, base_end = segment['end_points']
        # reaction_adjustment_start = reaction_start - adjustment_padding
        # closed_chunk = conf.get('song_audio_data')[base_start : base_end]
        # open_chunk = reaction.get('reaction_audio_data')[reaction_adjustment_start:reaction_end + adjustment_padding]
        # correlation = correlate(open_chunk, closed_chunk)
        # max_corr = int(np.argmax(correlation))
        # new_reaction_start = reaction_adjustment_start + correct_peak_index(max_corr, base_end - base_start)

        # if new_reaction_start != reaction_start and abs(new_reaction_start - reaction_start) < adjustment_padding:
        #     print(f'Adjusting reaction start by {(new_reaction_start - reaction_start) / sr}')
        #     segment['end_points'] = [new_reaction_start, new_reaction_start + base_end - base_start, base_start, base_end]


def sharpen_endpoints(reaction, chunk_size, step, segments, allowed_spacing):

    for segment in segments:
        if segment.get('pruned', False): 
            continue

        #########################################
        # Now we're going to try to sharpen up the endpoint
        
        reaction_start, reaction_end, base_start, base_end = segment['end_points']

        heat = [0 for i in range(0, int((base_end-base_start) / step))]

        for stroke in segment['strokes']:
            (__,__,stroke_start, stroke_end) = stroke
            position = stroke_start
            while position + step < stroke_end:
                idx = int( (position - base_start) / step  )
                heat[idx] += 1
                position += step

        highest_idx = -1
        highest_val = -1
        for idx,val in enumerate(heat):
            if val >= highest_val:
                highest_idx = idx
                highest_val = val

        # now we're going to use find_segment_end starting from the last local maximum
        sharpen_start = max(0, highest_idx * step - chunk_size)

        current_start = base_start + sharpen_start
        degraded_reaction_start = reaction_start + sharpen_start

        end_segment, _, _ = find_segment_end(reaction, current_start, degraded_reaction_start, 0, chunk_size)

        if end_segment is not None:
            new_reaction_end = end_segment[1]
            new_base_end = end_segment[3]

            if new_base_end < base_start + highest_idx * step:
                new_reaction_end = reaction_start + highest_idx * step
                new_base_end = base_start + highest_idx * step


            if new_base_end < base_end:        
                segment['end_points'] = [reaction_start, new_reaction_end, base_start, new_base_end]
                segment['key'] = str(segment['end_points'])
        else:
            print("AGG! Could not find segment end")


        ####################################################
        # Now we're going to try to sharpen up the beginning
        increment = int(step / 100)
        beginning_sharpen_threshold = .9

        candidate_base_start = base_start
        candidate_reaction_start = reaction_start

        first_score = None
        while candidate_base_start >= 0 and candidate_base_start >= base_start - step:
            candidate_segment =  (candidate_reaction_start, candidate_reaction_start + step, candidate_base_start, candidate_base_start + step)

            section_score = get_segment_mfcc_cosine_similarity_score(reaction, candidate_segment)

            if first_score is None: 
                first_score = section_score
            elif section_score < beginning_sharpen_threshold * first_score:                
                break

            candidate_base_start -= increment
            candidate_reaction_start -= increment

        shift = base_start - (candidate_base_start + increment)
        if shift > 0:
            segment['end_points'][0] -= shift
            segment['end_points'][1] -= shift
            segment['end_points'][2] -= shift
            segment['end_points'][3] -= shift
            # print(f"Shifted by {shift / sr} to {segment['end_points'][2] / sr}")






def determine_dominance(vocal_data, accompaniment_data):
    """
    Determines if a segment is vocal-dominated or accompaniment-dominated.

    Parameters:
    - vocal_data: Array representing the audio data of the vocal segment.
    - accompaniment_data: Array representing the audio data of the accompaniment segment.

    Returns:
    - "vocal" if the segment is vocal-dominated, "accompaniment" otherwise.
    """

    # Calculate the energy (or RMS value) for each segment
    vocal_energy = np.sqrt(np.mean(np.square(vocal_data)))
    accompaniment_energy = np.sqrt(np.mean(np.square(accompaniment_data)))

    # Determine dominance
    if vocal_energy > .1 * accompaniment_energy:
        return "vocal"
    else:
        return "accompaniment"


def get_signals(reaction, start, reaction_start, chunk_size):
    base_audio = conf.get('song_audio_data')
    reaction_audio = reaction.get('reaction_audio_data')

    hop_length = conf.get('hop_length')
    reaction_audio_mfcc = reaction.get('reaction_audio_mfcc')
    song_audio_mfcc = conf.get('song_audio_mfcc')


    # base_audio_vocals = conf.get('song_audio_vocals_data')
    # reaction_audio_vocals = reaction.get('reaction_audio_vocals_data')


    chunk = base_audio[start:start+chunk_size]
    chunk_mfcc = song_audio_mfcc[:,round(start/hop_length):round((start+chunk_size)/hop_length)]

    open_chunk = reaction_audio[reaction_start:]
    open_chunk_mfcc = reaction_audio_mfcc[:, round( reaction_start / hop_length):]

    predominantly_silent = is_silent(chunk, threshold_db=-40)
    # vocal_dominated = 'vocal' == determine_dominance(base_audio_vocals[start:start+chunk_size], base_audio_accompaniment[start:start+chunk_size])


    signals = {
        'standard': (1, chunk, open_chunk),
        'standard mfcc': (hop_length, chunk_mfcc, open_chunk_mfcc)
    }

    # we'll also send the accompaniment if it isn't vocally dominated
    if False and predominantly_silent:
        base_audio_accompaniment = conf.get('song_audio_accompaniment_data')
        reaction_audio_accompaniment = reaction.get('reaction_audio_accompaniment_data')

        reaction_audio_accompaniment_mfcc = reaction.get('reaction_audio_accompaniment_mfcc')    
        base_audio_accompaniment_mfcc = conf.get('song_audio_accompaniment_mfcc')

        print("SILENT!", start / sr)
        chunk = base_audio_accompaniment[start:start+chunk_size]
        open_chunk = reaction_audio_accompaniment[reaction_start:]

        chunk_mfcc = base_audio_accompaniment_mfcc[:,round(start/hop_length):round((start+chunk_size)/hop_length)]
        open_chunk_mfcc = reaction_audio_accompaniment_mfcc[:, round( reaction_start / hop_length):]

        signals['accompaniment'] = (hop_length, chunk_mfcc, open_chunk_mfcc)
        signals['accompaniment mfcc'] = (hop_length, chunk_mfcc, open_chunk_mfcc)

        evaluate_with = 'accompaniment'
    else:
        evaluate_with = 'standard'

    # elif vocal_dominated: 
    #     song_pitch_contour = conf.get('song_audio_vocals_pitch_contour')
    #     reaction_pitch_contour = reaction.get('reaction_audio_vocals_pitch_contour')

    #     chunk_mfcc      =     song_pitch_contour[:, round( start / hop_length):round((start+chunk_size)/hop_length)]
    #     open_chunk_mfcc = reaction_pitch_contour[:, round( reaction_start / hop_length):]

    #     signals['pitch contour on vocals'] = (hop_length, chunk_mfcc, open_chunk_mfcc)


    if False: 

        song_spectral_flux = conf.get('song_audio_accompaniment_spectral_flux')
        reaction_spectral_flux = reaction.get('reaction_audio_accompaniment_spectral_flux')

        song_root_mean_square_energy = conf.get('song_audio_accompaniment_root_mean_square_energy')
        reaction_root_mean_square_energy = reaction.get('reaction_audio_accompaniment_root_mean_square_energy')

        # chunk_mfcc = song_spectral_flux[:,round(start/hop_length):round((start+chunk_size)/hop_length)]
        # open_chunk_mfcc = reaction_spectral_flux[:, round( reaction_start / hop_length):]

        chunk_mfcc = song_root_mean_square_energy[:,round(start/hop_length):round((start+chunk_size)/hop_length)]
        open_chunk_mfcc = reaction_root_mean_square_energy[:, round( reaction_start / hop_length):]

    return signals, evaluate_with


def get_candidate_starts(reaction, signals, peak_tolerance, open_start, closed_start, chunk_size, distance, upper_bound, evaluate_with='standard'):


    peak_indices = {}
    for signal, (hop_length, chunk, open_chunk) in signals.items():
        new_candidates = find_segment_starts(
                                signal=signal,
                                reaction=reaction, 
                                open_chunk=open_chunk, 
                                closed_chunk=chunk, 
                                current_chunk_size=chunk_size, 
                                peak_tolerance=peak_tolerance, 
                                open_start=open_start,
                                closed_start=closed_start, 
                                distance=1 * sr, 
                                upper_bound=upper_bound,
                                hop_length=hop_length)

        if new_candidates is not None: 
            peak_indices[signal] = new_candidates

    candidates = score_start_candidates(
                                signals = signals,
                                peak_indices=peak_indices,
                                open_chunk=signals[evaluate_with][2], 
                                closed_chunk=signals[evaluate_with][1], 
                                open_chunk_mfcc=signals[f"{evaluate_with} mfcc"][2],
                                closed_chunk_mfcc=signals[f"{evaluate_with} mfcc"][1],
                                current_chunk_size=chunk_size, 
                                peak_tolerance=peak_tolerance, 
                                open_start=open_start,
                                closed_start=closed_start)
    return candidates


def get_lower_bounds(reaction, chunk_size): 
    manual_bounds = reaction.get('manual_bounds', None)
    lower_bounds = []
    if manual_bounds:
        for mbound in manual_bounds:
            ts, upper = mbound
            lower = upper - 2 
            ts = int(ts*sr)
            lower = int(lower*sr) # + chunk_size / 2)
            lower_bounds.append( (ts, lower)   )
    lower_bounds.reverse()
    return lower_bounds


def find_segments(reaction, chunk_size, step, peak_tolerance):

    base_audio = conf.get('song_audio_data')
    reaction_audio = reaction.get('reaction_audio_data')


    starting_points = range(0, len(base_audio) - chunk_size, step)

    start_reaction_search_at = int(reaction.get('start_reaction_search_at'))

    minimums = [start_reaction_search_at]


    # Factor in manually configured bounds    
    lower_bounds = get_lower_bounds(reaction, chunk_size)

    seg_cache_key = f"{minimums[0]}-{chunk_size}-{conf['first_n_samples']}-{reaction.get('unreliable_bounds','')}-{len(lower_bounds)}-{reaction.get('end_reaction_search_at',0)}-{reaction.get('start_reaction_search_at',0)}-{int(peak_tolerance*100)}"

    strokes = []
    active_strokes = []


    cache_dir = os.path.join(conf.get('song_directory'), '_cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file_name = f"{reaction.get('channel')}-start_cache-{seg_cache_key}.pckl"
    candidate_cache_file = os.path.join( cache_dir, cache_file_name )    
    candidate_cache_file_legacy = os.path.join( conf.get('song_directory'), cache_file_name )

    if os.path.exists(candidate_cache_file_legacy):
        import shutil
        shutil.move(candidate_cache_file_legacy, candidate_cache_file)

    if os.path.exists(candidate_cache_file):
        candidate_cache = read_object_from_file(candidate_cache_file)
    else:
        candidate_cache = {}


    alignment_bounds = create_reaction_alignment_bounds(reaction, conf['first_n_samples'])

    reaction_span = reaction.get('end_reaction_search_at', len(reaction_audio))

    for i,start in enumerate(starting_points):

        print(f"\tStroking...{i/len(starting_points)*100:.2f}%",end="\r")

        print_profiling()

        latest_lower_bound = 0
        for lower_bound_ts, lower_bound in lower_bounds:
            if lower_bound_ts <= start:
                latest_lower_bound = lower_bound + (start - lower_bound_ts)
                break

        candidate_lower_bound = max( min(minimums[-18:]), latest_lower_bound, chunk_size )

        assert(candidate_lower_bound >= latest_lower_bound)

        reaction_start = max( minimums[0] + start, candidate_lower_bound - chunk_size)

        upper_bound = len(reaction_audio) - len(base_audio)
        if start not in candidate_cache:
            upper_bound = get_bound(alignment_bounds, start, reaction_span - (len(base_audio) - start))

            signals, evaluate_with = get_signals(reaction, start, reaction_start, chunk_size)
            candidates = get_candidate_starts(reaction, signals, peak_tolerance, reaction_start, start, chunk_size, 1 * sr, upper_bound, evaluate_with=evaluate_with)

            candidates = [c + reaction_start for c in candidates if c + reaction_start >= latest_lower_bound]
            candidates.sort()


            candidate_cache[start] = candidates

            if len(candidates) == 0:
                continue

        else: 
            candidates = candidate_cache[start]

        

        for c in candidates:
            assert(c >= latest_lower_bound, (c-start) / sr)
            assert(c <= upper_bound, (c-start) / sr)

        if len(candidates) > 0:
            minimums.append( candidates[0] )
        else: 
            minimums.append( 0 )
        still_active_strokes = []

        already_matched = {}
        active_strokes.sort( key=lambda x: x['end_points'][2]  )
        for i, segment in enumerate(active_strokes):

            best_match = None  
            best_match_overlap = None        

            for y1 in candidates:
                if y1 in already_matched: 
                    continue

                y2 = y1+chunk_size

                x1 = start
                x2 = start+chunk_size

                new_stroke = (y1,y2,x1,x2)

                overlap = are_continuous_or_overlap( new_stroke, segment['end_points'])
                if overlap is not None:
                    if best_match is None or best_match_overlap > overlap:
                        best_match = new_stroke
                        best_match_overlap = overlap

                    if overlap == 0:
                        break


            if best_match is not None:
                new_stroke = best_match
                segment['end_points'][3] = new_stroke[3]
                segment['end_points'][1] = segment['end_points'][0] + segment['end_points'][3] - segment['end_points'][2]
                segment['strokes'].append( new_stroke )

                if best_match_overlap != 0:
                    segment['imprecise'] = True

                already_matched[new_stroke[0]] = True 

        for y1 in candidates:
            if y1 in already_matched: 
                continue

            y2 = y1+chunk_size

            x1 = start
            x2 = start+chunk_size

            new_stroke = (y1,y2,x1,x2)

            # create a new segment
            segment = {
                'end_points': [y1,y2,x1,x2],
                'strokes': [ new_stroke ],
                'pruned': False
            }
            strokes.append( segment )
            still_active_strokes.append( segment )

        for stroke in active_strokes:
            if stroke['end_points'][3] >= start - chunk_size:
                still_active_strokes.append(stroke)
        active_strokes = still_active_strokes



    save_object_to_file(candidate_cache_file, candidate_cache)

    for segment in strokes: 
        segment['key'] = str(segment['end_points'])
        if len(segment['strokes']) < 4:
            segment['pruned'] = True

    # splay_paint(reaction, strokes, stroke_alpha=step/chunk_size, show_live=False, chunk_size=chunk_size)
    return [stroke for stroke in strokes if not stroke['pruned']]


####################
# Utility functions

def are_continuous_or_overlap(line1, line2, epsilon=None):
    if epsilon is None:
        epsilon = .25 * sr # this should be less than the distance parameter for find_segment_starts
    cy, dy, cx, dx = line1
    ay, by, ax, bx = line2
    
    close_enough = ax <= cx and cx <= bx and ay <= cy and cy <= by

    # if close_enough:
    #     slope_if_continued = (dy-ay) / (dx-ax)

    #     if abs(slope_if_continued - 1) < epsilon:
    #         return abs(slope_if_continued - 1)

    if close_enough:
        intercept_1 = cy - cx
        intercept_2 = ay - ax

        if abs(intercept_1 - intercept_2) < epsilon:
            return abs(intercept_1 - intercept_2)


    return None

def near_song_beginning(segment, allowed_spacing):
    return segment['end_points'][2] < allowed_spacing


def near_song_end(segment, allowed_spacing):
    song_length = len(conf.get('song_audio_data'))
    return song_length - segment['end_points'][3] < 3 * allowed_spacing  # because of weird ends of songs, let it be farther away




#########################
# Drawing





def splay_paint(reaction, strokes, stroke_alpha, done=False, show_live=True, paths=None, chunk_size=None):
    plt.figure(figsize=(20, 10)) 

    
    # Creating the first plot
    plt.subplot(1, 2, 1)

    plt.title(f"Path painting of {reaction.get('channel')} / {conf.get('song_key')}")

    draw_strokes(reaction, chunk_size, strokes, stroke_alpha, paths=paths, intercept_based_figure=False)

    plt.subplot(1, 2, 2)

    draw_strokes(reaction, chunk_size, strokes, stroke_alpha, paths=paths, intercept_based_figure=True)

    plt.tight_layout()

    # Save the plot instead of displaying it
    filename = os.path.join(conf.get('temp_directory'), f"{reaction.get('channel')}-painting-{int(chunk_size/sr)}.png")
    plt.savefig(filename, dpi=300)

    if show_live:
        plt.show()
    else:
        plt.close()


def draw_strokes(reaction, chunk_size, strokes, stroke_alpha, paths, intercept_based_figure=False):

    alignment_bounds = reaction['alignment_bounds']
    if len(alignment_bounds) > 0:
        base_ts, intercepts = zip(*alignment_bounds)
        base_ts = [bs/sr for bs in base_ts]
    else: 
        base_ts = [0, len(conf.get('song_audio_data'))]


    if reaction.get('ground_truth'):
        for rs,re,cs,ce,f in reaction.get('ground_truth_path'):
            make_stroke((rs,re,cs,ce), alpha = 1, color='#aaa', linewidth=4, intercept_based_figure=intercept_based_figure)


    # for segment in strokes:
    #     for stroke in segment['strokes']:   
    #         make_stroke(stroke, linewidth=2, alpha = stroke_alpha, intercept_based_figure=intercept_based_figure)

    for segment in strokes:
        # if not segment.get('pruned', False) and 'old_end_points' in segment:
        if segment.get('pruned', False):
            alpha = .2
        else:
            alpha = 1
        make_stroke(segment.get('old_end_points', segment['end_points']), linewidth=2, color='orange', alpha = alpha, intercept_based_figure=intercept_based_figure)

        make_stroke(segment.get('end_points', segment['end_points']), linewidth=2, color='blue', alpha = alpha, intercept_based_figure=intercept_based_figure)

    # for segment in strokes:
    #     if not segment.get('pruned', False):

    #         for stroke in segment['strokes']:   
    #             make_stroke(stroke, linewidth=4, alpha = stroke_alpha, intercept_based_figure=intercept_based_figure)

    #         # make_stroke(segment['end_points'], linewidth=1, color='blue', alpha = 1, intercept_based_figure=intercept_based_figure)


    if paths is not None:
        for path in paths: 
            visualize_candidate_path(path, intercept_based_figure=intercept_based_figure)      


    
    # Find the min and max values of base_ts for the width of the chart
    x_min = min(base_ts)
    x_max = max(base_ts)
    alignment_bound_linewidth=2
    for xx, c in alignment_bounds:
        c /= sr  # Calculate the y-intercept
        if intercept_based_figure: 
            plt.plot([x_min, xx/sr], [c, c], linewidth=alignment_bound_linewidth, alpha=1, color='black')  # Plot the line using the y = mx + c equation
        else:
            plt.plot([x_min, xx/sr], [x_min + c, xx/sr + c], linewidth=alignment_bound_linewidth, alpha=1, color='black')  # Plot the line using the y = mx + c equation

    lower_bounds = get_lower_bounds(reaction, chunk_size)
    for xx, c in lower_bounds:
        c /= sr 
        if intercept_based_figure: 
            plt.plot([xx/sr, x_max], [c-xx/sr, c-xx/sr], linewidth=alignment_bound_linewidth, alpha=1, color='red')  # Plot the line using the y = mx + c equation
        else:
            plt.plot([xx/sr, x_max], [c, c+(x_max-xx/sr)], linewidth=alignment_bound_linewidth, alpha=1, color='red')  # Plot the line using the y = mx + c equation


    x = conf.get('song_audio_data')
    y = reaction.get('reaction_audio_data')

    if intercept_based_figure: 
        plt.ylabel("React Audio intercept [s]")        
        # plt.yticks(np.arange(0, max(intercepts) / sr + 30, 30))
        plt.yticks(np.arange(0, max(y) / sr + 30, 30))

    else:
        plt.ylabel("Time in React Audio [s]")
        plt.yticks(np.arange(0, len(y) / sr + 30, 30))

    plt.xlabel("Time in Base Audio [s]")
    plt.xticks(np.arange(0, len(x) / sr + 30, 30))

    plt.grid(True)




def make_stroke(stroke, color=None, linewidth=1, alpha=1, intercept_based_figure=False):

    reaction_start, reaction_end, current_start, current_end = stroke

    if color is None:
        color = 'red'

    if intercept_based_figure:
        plt.plot( [current_start/sr, current_end/sr], [(reaction_start-current_start)/sr, (reaction_start-current_start)/sr]  , color=color, linestyle='solid', linewidth=linewidth, alpha=alpha)
    else:
        plt.plot( [current_start/sr, current_end/sr], [reaction_start/sr, reaction_end/sr]  , color=color, linestyle='solid', linewidth=linewidth, alpha=alpha)

    

def visualize_candidate_path(path, color=None, linewidth=1, alpha=1, intercept_based_figure=False):
    segment_color = color
    for i, segment in enumerate(path):


        if len(segment) == 5:
            reaction_start, reaction_end, current_start, current_end, filler = segment
        else: 
            reaction_start, reaction_end, current_start, current_end, filler, strokes = segment

        if color is None:
            segment_color = 'green'
            if filler:
                segment_color = 'turquoise'

        if i > 0:
            if len(previous_segment) == 5:
                last_reaction_start, last_reaction_end, last_current_start, last_current_end, last_filler = previous_segment
            else:
                last_reaction_start, last_reaction_end, last_current_start, last_current_end, last_filler, strokes = previous_segment

            if intercept_based_figure:
                plt.plot( [last_current_end/sr, current_start/sr], [(reaction_start-current_start)/sr, (reaction_start-current_start)/sr], color=segment_color, linestyle='dashed', linewidth=linewidth, alpha=alpha)
            else:
                plt.plot( [last_current_end/sr, current_start/sr], [last_reaction_end/sr, reaction_start/sr], color=segment_color, linestyle='dashed', linewidth=linewidth, alpha=alpha)

        if intercept_based_figure:
            plt.plot( [current_start/sr, current_end/sr], [(reaction_start-current_start)/sr, (reaction_start-current_start)/sr], color=segment_color, linestyle='solid', linewidth=linewidth, alpha=alpha)
        else: 
            plt.plot( [current_start/sr, current_end/sr], [reaction_start/sr, reaction_end/sr], color=segment_color, linestyle='solid', linewidth=linewidth, alpha=alpha)
        previous_segment = segment


from prettytable import PrettyTable
def print_prune_data():

    x = PrettyTable()
    x.border = False
    x.align = "r"
    x.field_names = ['\t', 'Prune type', 'Count']
    

    for k,v in prune_cache.items():
        x.add_row(['\t', k, v])  

    print(x)

