from utilities import conversion_audio_sample_rate as sr
from utilities import conf, print_profiling, save_object_to_file, read_object_from_file
from aligner.bounds import get_bound, in_bounds, create_reaction_alignment_bounds
from aligner.find_segment_start import find_segment_starts, initialize_segment_start_cache
from aligner.find_segment_end import find_segment_end, initialize_segment_end_cache
from aligner.scoring_and_similarity import find_best_path, initialize_path_score, initialize_segment_tracking, get_segment_mfcc_cosine_similarity_score, path_score, print_path, path_score_by_mfcc_cosine_similarity, truncate_path
from aligner.cross_expander import compress_segments
from aligner.pruning_search import is_path_quality_poor, initialize_path_pruning
from silence import is_silent
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time




def paint_paths(reaction, peak_tolerance=.4, allowed_spacing=None, chunk_size=None):

    if chunk_size is None: 
        chunk_size = reaction.get('chunk_size', 3)

    print(f"\n###############################\n# {conf.get('song_key')} / {reaction.get('channel')}")

    initialize_segment_start_cache()
    initialize_segment_end_cache()
    initialize_path_score()
    initialize_segment_tracking()
    initialize_paint_caches()
    initialize_path_pruning()

    chunk_size *= sr
    chunk_size = int(chunk_size)
    step = int(.5 * sr)

    if allowed_spacing is None:
        allowed_spacing = 3 * sr

    start = time.perf_counter()

    segments = find_segments(reaction, chunk_size, step, peak_tolerance)

    # splay_paint(reaction, segments, stroke_alpha=.2, show_live=True)

    pruned_segments, __ = prune_unreachable_segments(reaction, segments, allowed_spacing, prune_links = False)

    # splay_paint(reaction, segments, stroke_alpha=.2, show_live=True)

    sharpen_segments(reaction, chunk_size, step, pruned_segments, allowed_spacing)

    # pruned_segments = prune_low_quality_segments(reaction, pruned_segments, allowed_spacing)
    
    if allowed_spacing == 3 * sr:
        pruned_segments = prune_neighbors(reaction, pruned_segments, allowed_spacing)

    pruned_segments, joinable_segment_map = prune_unreachable_segments(reaction, segments, allowed_spacing, prune_links = True)

    # splay_paint(reaction, segments, stroke_alpha=.2, show_live=True)

    print(f"Constructing paths from {len(pruned_segments)} viable segments")

    paths = construct_all_paths(reaction, pruned_segments, joinable_segment_map, allowed_spacing)

    paths = finesse_paths(reaction, paths)

    if len(paths) == 0 and allowed_spacing < 10 * sr: 
        print(f"No paths found. Trying with {allowed_spacing * 2 / sr} spacing")
        splay_paint(reaction, segments, stroke_alpha=.2, show_live=False)

        return paint_paths(reaction, peak_tolerance, allowed_spacing=allowed_spacing * 2)

    print(f"Found {len(paths)} paths")

    best_path = find_best_path(reaction, paths)
    best_path = compress_segments(best_path)

    alignment_duration = (time.perf_counter() - start) / 60 # in minutes

    splay_paint(reaction, segments, stroke_alpha=.2, show_live=False, paths=[best_path])

    return best_path




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
        print(f"Neighborhood #{len(neighborhood)}")
        for segment in neighborhood:
            end_points = segment['end_points']
            if segment in to_eliminate:
                print(f"\t*** Segment ({end_points[2]/sr:.1f}, {end_points[3]/sr:.1f}), ({end_points[0]/sr:.1f}, {end_points[1]/sr:.1f})")
            else: 
                print(f"\tSegment ({end_points[2]/sr:.1f}, {end_points[3]/sr:.1f}), ({end_points[0]/sr:.1f}, {end_points[1]/sr:.1f})")



    to_keep = [s for s in segments if s not in to_eliminate]
    print(f"Pruned {len(segments) - len(to_keep)} of {len(segments)} neighbor segments")

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


def finesse_paths(reaction, paths):
    final_paths = []
    for path in paths:
        new_path = []
        for idx, segment in enumerate(path):
            reaction_start, reaction_end, base_start, base_end, is_filler = segment

            extended = False
            if idx > 0 and idx < len(path) - 1:
                if is_filler and not path[idx-1][-1] and not path[idx+1][-1]: # if this segment is filler surrounded by non-filler...
                    # and prior segment and later segment are continuous with each other...
                    prior_segment = path[idx-1]
                    later_segment = path[idx+1]
                    intercept_prior = prior_segment[0] - prior_segment[2]
                    intercept_later = later_segment[0] - later_segment[2]

                    if abs( intercept_prior - intercept_later ) / sr < .1:
                        # remove filler, extend previous, merge

                        new_path.pop()
                        later_segment[0] = prior_segment[0]
                        later_segment[2] = prior_segment[2]
                        later_segment[1] = prior_segment[0] + later_segment[3] - later_segment[2]
                        extended = True

            if not extended:
                new_path.append(segment)

        final_paths.append(new_path)

    return final_paths


def construct_all_paths(reaction, segments, joinable_segment_map, allowed_spacing):
    paths = []
    song_length = len(conf.get('song_audio_data'))
    partial_paths = []


    def complete_path(path):
        completed_path = copy.deepcopy(path)

        fill_len = song_length - path[-1][3]
        if fill_len > 0:

            if fill_len < sr: 
                path[-1][3] += fill_len
                path[-1][1] += fill_len
            else:
                completed_path.append( [path[-1][1], path[-1][1] + fill_len, path[-1][3], path[-1][3] + fill_len, True] ) 

        score = path_score_by_mfcc_cosine_similarity(path, reaction) # path_score(path, reaction)
        if best_score_cache['best_overall_path'] is None or best_score_cache['best_overall_score'] < score:
            best_score_cache['best_overall_path'] = path 
            best_score_cache['best_overall_score'] = score

            print("\nNew best score!")
            print_path(path, reaction)

        if score > .9 * best_score_cache['best_overall_score']:
            paths.append(completed_path)


    for c in segments:
        if near_song_beginning(c, allowed_spacing):
            reaction_start, reaction_end, base_start, base_end = c['end_points']
            start_path = [ [reaction_start, reaction_end, base_start, base_end, False] ]
            if base_start > 0:
                start_path.insert(0, (reaction_start - base_start, reaction_start, 0, base_start, True) )


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

    reaction_start, reaction_end, base_start, base_end, is_filler = current_path[-1]

    b_current = reaction_end - base_end  # as in, y = ax + b

    for candidate in joinable_segments:

        candidate_reaction_start, candidate_reaction_end, candidate_base_start, candidate_base_end = candidate['end_points']

        distance = base_end - candidate_base_start
        if candidate_base_end > base_end and distance > -allowed_spacing:
            branch = copy.copy(current_path)

            if distance < 0:
                branch.append( [reaction_end, reaction_end - distance, base_end, base_end - distance, True] )
                filled = -distance
            else: 
                filled = 0

            b_candidate = candidate_reaction_end - candidate_base_end

            branch.append( [reaction_end + filled + b_candidate - b_current, candidate_reaction_end, base_end + filled, candidate_base_end, False]      )

            branches.append( [branch, candidate]  )

    return branches


def prune_low_quality_segments(reaction, segments, allowed_spacing):
    if len(segments) == 0:
        return segments

    good_segments = []
    score_total = 0
    for segment in segments:
        score = get_segment_mfcc_cosine_similarity_score

        cosine_score = get_segment_mfcc_cosine_similarity_score(reaction, segment['end_points'])

        good_segments.append([segment, cosine_score])
        score_total += cosine_score
    avg = score_total / len(segments)

    final_segments = []
    for segment, cosine_score in good_segments:
        if cosine_score < .6 and cosine_score < .75 * avg: 
            prune_cache['segment_quality'] += 1
            continue 
        else: 
            final_segments.append(segment)

    print(f"Kept {len(final_segments)} of {len(segments)} based on quality")
    return final_segments





def prune_unreachable_segments(reaction, segments, allowed_spacing, prune_links = False): 

    segments = [s for s in segments if not s.get('pruned', False)]
    
    segments.sort( key=lambda x: x['end_points'][1]  )  # sort by reaction_end

    joinable_segments = {}

    for segment in segments:
        segment_id = segment['key']
        joins = joinable_segments[segment_id] = get_joinable_segments(reaction, segment, segments, allowed_spacing, prune_links = prune_links)

        at_end = segment['at_end'] = near_song_end(segment, allowed_spacing)

        if len(joins) == 0 and not at_end:
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


    segments = [s for s in segments if not s.get('pruned', False)]

    # for k,v in joinable_segments.items():
    #     print(k,v)

    return segments, joinable_segments





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

        on_top = cy - cx >= .999 * (ay - ax)  # solving for b in y=ax+b, where a=1
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



def sharpen_segments(reaction, chunk_size, step, segments, allowed_spacing):

    for segment in segments:

        ###################################
        # if we have an imprecise segment composed of strokes that weren't exactly aligned, 
        # we'll want to find the best line through them
        if segment.get('imprecise', False):
            unique_intercepts = {} # y-intercepts as in y=ax+b, solving for b when a=1
            for stroke in segment['strokes']:
                b = stroke[0] - stroke[2]
                unique_intercepts[b] = True

            best_line_def = None
            best_line_def_score = -1
            for intercept in unique_intercepts.keys():
                candidate_line_def = copy.copy(segment['end_points'])
                candidate_line_def[0] = intercept + candidate_line_def[2]
                candidate_line_def[1] = intercept + candidate_line_def[3]

                score = get_segment_mfcc_cosine_similarity_score(reaction, candidate_line_def)
                
                if score > best_line_def_score:
                    best_line_def_score = score
                    best_line_def = candidate_line_def

            segment['end_points'] = best_line_def



        segment['old_end_points'] = segment['end_points']

        if near_song_end(segment, allowed_spacing):
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


def find_segments(reaction, chunk_size, step, peak_tolerance):



    base_audio = conf.get('song_audio_data')
    reaction_audio = reaction.get('reaction_audio_data')

    hop_length = conf.get('hop_length')
    reaction_audio_mfcc = reaction.get('reaction_audio_mfcc')
    song_audio_mfcc = conf.get('song_audio_mfcc')

    base_audio_accompaniment = conf.get('song_audio_accompaniment_data')
    reaction_audio_accompaniment = reaction.get('reaction_audio_accompaniment_data')

    reaction_audio_accompaniment_mfcc = reaction.get('reaction_audio_accompaniment_mfcc')    
    base_audio_accompaniment_mfcc = conf.get('song_audio_accompaniment_mfcc')

    base_audio_vocals = conf.get('song_audio_vocals_data')
    reaction_audio_vocals = reaction.get('reaction_audio_vocals_data')




    alignment_bounds = create_reaction_alignment_bounds(reaction, conf['first_n_samples'])

    starting_points = range(0, len(base_audio) - chunk_size, step)

    start_reaction_search_at = reaction.get('start_reaction_search_at', 3)

    minimums = [start_reaction_search_at * sr]



    strokes = []
    active_strokes = []

    candidate_cache_file = os.path.join( conf.get('song_directory'), f"{reaction.get('channel')}-start_cache-{minimums[0]}.pckl"   )

    if os.path.exists(candidate_cache_file):
        candidate_cache = read_object_from_file(candidate_cache_file)
    else:
        candidate_cache = {}
    
    for i,start in enumerate(starting_points):

        print(f"\tStroking...{i/len(starting_points)*100:.2f}%",end="\r")

        print_profiling()

        reaction_start = max( minimums[0] + start, min(minimums[-18:]))

        if start not in candidate_cache:

            upper_bound = get_bound(alignment_bounds, start, len(reaction_audio))

            if reaction.get('unreliable_bounds', False):
                upper_bound = min(len(reaction_audio), upper_bound * 1.5)


            chunk = base_audio[start:start+chunk_size]
            chunk_mfcc = song_audio_mfcc[:,round(start/hop_length):round((start+chunk_size)/hop_length)]

            open_chunk = reaction_audio[reaction_start:]
            open_chunk_mfcc = reaction_audio_mfcc[:, round( reaction_start / hop_length):]



            predominantly_silent = is_silent(chunk, threshold_db=-30)
            vocal_dominated = 'vocal' == determine_dominance(base_audio_vocals[start:start+chunk_size], base_audio_accompaniment[start:start+chunk_size])


            candidate_start_metrics = [ ('standard', chunk, open_chunk, chunk_mfcc, open_chunk_mfcc),   ]


            # we'll also send the accompaniment if it isn't vocally dominated
            if not vocal_dominated or predominantly_silent:
                chunk = base_audio_accompaniment[start:start+chunk_size]
                open_chunk = reaction_audio_accompaniment[reaction_start:]

                chunk_mfcc = base_audio_accompaniment_mfcc[:,round(start/hop_length):round((start+chunk_size)/hop_length)]
                open_chunk_mfcc = reaction_audio_accompaniment_mfcc[:, round( reaction_start / hop_length):]

                candidate_start_metrics.append( ('accompaniment', chunk, open_chunk, chunk_mfcc, open_chunk_mfcc) )

            else: 
                song_pitch_contour = conf.get('song_audio_vocals_pitch_contour')
                reaction_pitch_contour = reaction.get('reaction_audio_vocals_pitch_contour')

                chunk_mfcc      =     song_pitch_contour[:, round( start / hop_length):round((start+chunk_size)/hop_length)]
                open_chunk_mfcc = reaction_pitch_contour[:, round( reaction_start / hop_length):]

                candidate_start_metrics.append( ('vocals', chunk, open_chunk, chunk_mfcc, open_chunk_mfcc) )


            if False: 

                song_spectral_flux = conf.get('song_audio_accompaniment_spectral_flux')
                reaction_spectral_flux = reaction.get('reaction_audio_accompaniment_spectral_flux')

                song_root_mean_square_energy = conf.get('song_audio_accompaniment_root_mean_square_energy')
                reaction_root_mean_square_energy = reaction.get('reaction_audio_accompaniment_root_mean_square_energy')

                # chunk_mfcc = song_spectral_flux[:,round(start/hop_length):round((start+chunk_size)/hop_length)]
                # open_chunk_mfcc = reaction_spectral_flux[:, round( reaction_start / hop_length):]

                chunk_mfcc = song_root_mean_square_energy[:,round(start/hop_length):round((start+chunk_size)/hop_length)]
                open_chunk_mfcc = reaction_root_mean_square_energy[:, round( reaction_start / hop_length):]


            candidates = []
            for metric_group, chunk, open_chunk, chunk_mfcc, open_chunk_mfcc in candidate_start_metrics:
                new_candidates = find_segment_starts(
                                        metric_group=metric_group,
                                        reaction=reaction, 
                                        open_chunk=open_chunk, 
                                        open_chunk_mfcc=open_chunk_mfcc,
                                        closed_chunk=chunk, 
                                        closed_chunk_mfcc= chunk_mfcc,
                                        current_chunk_size=chunk_size, 
                                        peak_tolerance=peak_tolerance, 
                                        full_search=True,
                                        open_start=reaction_start,
                                        closed_start=start, 
                                        distance=1 * sr, 
                                        filter_for_similarity=True,
                                        upper_bound=upper_bound)

                if new_candidates is not None: 
                    for c in new_candidates:
                        if c not in candidates:
                            candidates.append(c)

            if len(candidates) == 0:
                continue
            candidates.sort()
            candidate_cache[start] = candidates
        else: 
            candidates = candidate_cache[start]



        minimums.append( candidates[0] + reaction_start )

        still_active_strokes = []

        already_matched = {}
        active_strokes.sort( key=lambda x: x['end_points'][2]  )
        for i, segment in enumerate(active_strokes):

            best_match = None  
            best_match_overlap = None        

            for c in candidates:
                y1 = reaction_start+c
                if y1 in already_matched: 
                    continue

                y2 = reaction_start+c+chunk_size

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

        for c in candidates:
            y1 = reaction_start+c
            if y1 in already_matched: 
                continue

            y2 = reaction_start+c+chunk_size

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

    # splay_paint(reaction, strokes, stroke_alpha=step/chunk_size, show_live=False)
    return strokes


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
    return song_length - segment['end_points'][3] < allowed_spacing




#########################
# Drawing

def splay_paint(reaction, strokes, stroke_alpha, done=False, show_live=True, paths=None):
    plt.figure(figsize=(10, 10))

    x = conf.get('song_audio_data')
    y = reaction.get('reaction_audio_data')

    if reaction.get('ground_truth'):
        for rs,re,cs,ce,f in reaction.get('ground_truth_path'):
            make_stroke((rs,re,cs,ce), alpha = 1, color='#aaa', linewidth=4)


    # for segment in strokes:
    #     for stroke in segment['strokes']:   
    #         make_stroke(stroke, linewidth=2, alpha = stroke_alpha)


    for segment in strokes:
        # if not segment.get('pruned', False) and 'old_end_points' in segment:
        if segment.get('pruned', False):
            alpha = .25
        else:
            alpha = 1
        make_stroke(segment.get('old_end_points', segment['end_points']), linewidth=2, color='orange', alpha = alpha)

        make_stroke(segment.get('end_points', segment['end_points']), linewidth=2, color='blue', alpha = alpha)

    # for segment in strokes:
    #     if not segment.get('pruned', False):

    #         for stroke in segment['strokes']:   
    #             make_stroke(stroke, linewidth=2, alpha = stroke_alpha)

    #         # make_stroke(segment['end_points'], linewidth=1, color='blue', alpha = 1)


    if paths is not None:
        for path in paths: 
            visualize_candidate_path(path)      


    segment_count = len([s for s in strokes if not segment.get('pruned', False)])

    plt.ylabel("Time in React Audio [s]")
    plt.xlabel("Time in Base Audio [s]")

    plt.title(f"Path painting of {reaction.get('channel')} / {conf.get('song_key')}: {segment_count} viable found")

    plt.xticks(np.arange(0, len(x) / sr + 30, 30))
    plt.yticks(np.arange(0, len(y) / sr + 30, 30))

    plt.grid(True)

    plt.tight_layout()


    # Save the plot instead of displaying it
    filename = os.path.join(conf.get('temp_directory'), f"{reaction.get('channel')}-painting.png")
    plt.savefig(filename, dpi=300)

    if show_live:
        plt.show()
    else:
        plt.close()


def make_stroke(stroke, color=None, linewidth=1, alpha=1):

    reaction_start, reaction_end, current_start, current_end = stroke

    if color is None:
        color = 'red'

    plt.plot( [current_start/sr, current_end/sr], [reaction_start/sr, reaction_end/sr]  , color=color, linestyle='solid', linewidth=linewidth, alpha=alpha)

    

def visualize_candidate_path(path, color=None, linewidth=1, alpha=1):
    segment_color = color
    for i, segment in enumerate(path):

        reaction_start, reaction_end, current_start, current_end, filler = segment

        if color is None:
            segment_color = 'green'
            if filler:
                segment_color = 'turquoise'

        if i > 0:
            last_reaction_start, last_reaction_end, last_current_start, last_current_end, last_filler = previous_segment
            plt.plot( [last_current_end/sr, current_start/sr], [last_reaction_end/sr, reaction_start/sr], color=segment_color, linestyle='dashed', linewidth=linewidth, alpha=alpha)

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

