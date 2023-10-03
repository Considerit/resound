from utilities import conversion_audio_sample_rate as sr
from utilities import conf, print_profiling, save_object_to_file, read_object_from_file
from cross_expander.bounds import get_bound, in_bounds, create_reaction_alignment_bounds
from cross_expander.find_segment_start import find_segment_starts, initialize_segment_start_cache
from cross_expander.find_segment_end import find_segment_end, initialize_segment_end_cache
from cross_expander.scoring_and_similarity import find_best_path, initialize_path_score, initialize_segment_tracking, get_segment_mfcc_cosine_similarity_score, path_score, print_path
from cross_expander.cross_expander import compress_segments
from cross_expander.pruning_search import is_path_quality_poor, initialize_path_pruning
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time


path_cache = {}
location_cache = {}
def initialize_paint_caches():
    path_cache.clear()
    location_cache.clear()


def paint_paths(reaction, peak_tolerance=.4, allowed_spacing=None):


    print(f"\n###############################\n# {conf.get('song_key')} / {reaction.get('channel')}")

    initialize_segment_start_cache()
    initialize_segment_end_cache()
    initialize_path_score()
    initialize_segment_tracking()
    initialize_paint_caches()
    initialize_path_pruning()

    chunk_size = 3 * sr
    step = int(chunk_size/6)

    if allowed_spacing is None:
        allowed_spacing = 3 * sr




    start = time.perf_counter()

    segments = find_segments(reaction, chunk_size, step, peak_tolerance)


    pruned_segments, __ = prune_unreachable_segments(reaction, segments, allowed_spacing)


    sharpen_segments(reaction, chunk_size, step, pruned_segments, allowed_spacing)

    pruned_segments, joinable_segment_map = prune_unreachable_segments(reaction, segments, allowed_spacing)

    # if True or reaction.get('channel') == 'Knox Hill':
    #     splay_paint(reaction, segments, stroke_alpha=.2, show_live=True)


    print(f"Constructing paths from {len(pruned_segments)} viable segments")

    paths = construct_all_paths(reaction, pruned_segments, joinable_segment_map, allowed_spacing)

    paths = finesse_paths(reaction, paths)

    if len(paths) == 0 and allowed_spacing < 10 * sr: 
        return paint_paths(reaction, peak_tolerance, allowed_spacing=allowed_spacing * 2)

    print(f"Found {len(paths)} paths")

    best_path = find_best_path(reaction, paths)
    best_path = compress_segments(best_path)

    alignment_duration = (time.perf_counter() - start) / 60 # in minutes

    splay_paint(reaction, segments, stroke_alpha=.2, show_live=False, paths=[best_path])

    return best_path


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
    song_length = len(conf.get('base_audio_data'))
    partial_paths = []
    for c in segments:
        if near_song_beginning(c, allowed_spacing):
            reaction_start, reaction_end, base_start, base_end = c['end_points']
            start_path = [ [reaction_start, reaction_end, base_start, base_end, False] ]
            if base_start > 0:
                start_path.insert(0, (reaction_start - base_start, reaction_start, 0, base_start, True) )
            seen = {}
            seen[c['key']] = True

            score = path_score(start_path, reaction, end=start_path[-1][1])
            partial_paths.append( [[ start_path, c, seen ], score] )

            if near_song_end(c, allowed_spacing): # in the case of one long completion

                completed_path = copy.deepcopy(start_path)

                fill_len = song_length - start_path[-1][3]
                if fill_len > 0:

                    if fill_len < sr: 
                        start_path[-1][3] += fill_len
                        start_path[-1][1] += fill_len
                    else:
                        completed_path.append( [start_path[-1][1], start_path[-1][1] + fill_len, start_path[-1][3], start_path[-1][3] + fill_len, True] ) 


                
                paths.append(completed_path)

    i = 0 
    start_time = time.perf_counter()
    was_prune_eligible = False 
    while len(partial_paths) > 0:
        
        prune_eligible = time.perf_counter() - start_time > 3 * 60 #True #len(partial_paths) > 100 #or len(paths) > 10000

        if prune_eligible and not was_prune_eligible:
            print("ENABLING PRUNING")
            iii = len(partial_paths) - 1

            for partial, score in reversed(partial_paths):
                if is_path_quality_poor(reaction, partial[0]):
                    partial_paths.pop(iii)
                else:
                    partial_path = []
                    for segment in partial[0]:
                        partial_path.append(segment)
                        should_prune, score = should_prune_for_location(reaction, partial_path, song_length)
                        if should_prune:
                            partial_paths.pop(iii)
                            break
                iii -= 1
            was_prune_eligible = True


        i += 1
        if i % 10 == 1:
            partial_paths.sort(key=lambda x: x[1][2], reverse=True)



        partial_path, score = partial_paths.pop(0)

        # print(len(partial_paths), end='\r')


        should_prune, score = should_prune_for_location(reaction, partial_path[0], song_length, score)
        
        if should_prune and prune_eligible:
            continue

        print(len(partial_paths), len(paths), len(partial_path[0]), end='\r')


        next_partials = branch_from( reaction, partial_path, joinable_segment_map[partial_path[1]['key']], allowed_spacing  )

        for partial in next_partials:
            path, last_segment, seen = partial
            if near_song_end(last_segment, allowed_spacing):

                completed_path = copy.deepcopy(path)

                fill_len = song_length - path[-1][3]
                if fill_len > 0:

                    if fill_len < sr: 
                        path[-1][3] += fill_len
                        path[-1][1] += fill_len
                    else:
                        completed_path.append( [path[-1][1], path[-1][1] + fill_len, path[-1][3], path[-1][3] + fill_len, True] ) 
                
                paths.append(completed_path)


            should_prune, score = should_prune_for_location(reaction, path, song_length)

            should_prune = should_prune or is_path_quality_poor(reaction, path)
            if (not should_prune or not prune_eligible) and last_segment['key'] in joinable_segment_map and len(joinable_segment_map[last_segment['key']]) > 0:
                partial_paths.append([partial, score])

    return paths

def should_prune_for_location(reaction, path, song_length, score=None):
    global location_cache

    last_segment = path[-1]            
    location_key = str(last_segment[-1])
    location_prune_threshold = .5 + .35 * last_segment[3] / song_length
    if score is None:
        score = path_score(path, reaction,end=last_segment[1])
    if location_key in location_cache:
        best_score = location_cache[location_key]
        if score[0] > best_score[0]:
            location_cache[location_key] = score
        else: 
            prunable = True
            for i in [0,2]:
                prunable = prunable and location_prune_threshold * best_score[i] > score[i]
            if prunable:
                # print('\npruned!', best_score[2], score[2])
                return True, score
    else: 
        location_cache[location_key] = score

    return False, score



def branch_from( reaction, partial_path, joinable_segments, allowed_spacing ):
    branches = []

    current_path, last_segment, seen = partial_path

    reaction_start, reaction_end, base_start, base_end, is_filler = current_path[-1]

    b_current = reaction_end - base_end  # as in, y = ax + b

    for candidate in joinable_segments:

        # if len(joinable_segments) > 3:
        #     my_segment = copy.deepcopy(candidate['end_points'])
        #     my_segment.append(False)
        #     cosine_score = get_segment_mfcc_cosine_similarity_score(reaction, my_segment)
        #     if cosine_score < .4: 
        #         continue 


        candidate_reaction_start, candidate_reaction_end, candidate_base_start, candidate_base_end = candidate['end_points']


        if candidate['key'] in seen:
            assert(False, "We have a loop in joinable_segment_map")

        distance = base_end - candidate_base_start
        if candidate_base_end > base_end and distance > -allowed_spacing:
            branch = copy.deepcopy(current_path)
            new_seen = copy.deepcopy(seen)
            new_seen[candidate['key']] = 1

            if distance < 0:
                branch.append( [reaction_end, reaction_end - distance, base_end, base_end - distance, True] )
                filled = -distance
            else: 
                filled = 0

            b_candidate = candidate_reaction_end - candidate_base_end

            branch.append( [reaction_end + filled + b_candidate - b_current, candidate_reaction_end, base_end + filled, candidate_base_end, False]      )


            key = str(branch)
            if key not in path_cache:
                path_cache[key] = True
                branches.append( [branch, candidate, new_seen]  )
            else: 
                print("DUPLICATE PATH FOUND")
    return branches



def prune_unreachable_segments(reaction, segments, allowed_spacing): 

    segments = [s for s in segments if not s.get('pruned', False)]
    
    segments.sort( key=lambda x: x['end_points'][1]  )  # sort by reaction_end

    joinable_segments = {}

    for segment in segments:
        segment_id = segment['key']
        joins = joinable_segments[segment_id] = get_joinable_segments(reaction, segment, segments, allowed_spacing)

        at_end = segment['at_end'] = near_song_end(segment, allowed_spacing)

        if len(joins) == 0 and not at_end:
            segment['pruned'] = True
            del joinable_segments[segment_id]

    segments = [s for s in segments if not s.get('pruned', False)]

    # clean up segments that can't reach the end
    pruned_another = True
    while pruned_another:
        segments = [s for s in segments if not s.get('pruned', False)]

        pruned_another = False
        for segment in segments:
            segment_id = segment['key']
            joins = joinable_segments[segment_id] = [s for s in joinable_segments[segment_id] if not s.get('pruned', False)]

            if len(joins) == 0 and not segment.get('at_end', False):
                segment['pruned'] = True
                del joinable_segments[segment_id]
                pruned_another = True

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


    segments = [s for s in segments if not s.get('pruned', False)]

    # for k,v in joinable_segments.items():
    #     print(k,v)

    return segments, joinable_segments





# returns all segments that a given segment could jump to
def get_joinable_segments(reaction, segment, all_segments, allowed_spacing): 

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

        on_top = cy - cx >= ay - ax  # solving for b in y=ax+b, where a=1
        over_each_other = bx - cx >= -allowed_spacing and ax < dx

        if on_top and over_each_other:
            candidates.append(candidate)

    return candidates



def sharpen_segments(reaction, chunk_size, step, segments, allowed_spacing):

    for segment in segments:


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





def find_segments(reaction, chunk_size, step, peak_tolerance):



    base_audio = conf.get('base_audio_data')
    reaction_audio = reaction.get('reaction_audio_data')
    hop_length = conf.get('hop_length')
    reaction_audio_mfcc = reaction.get('reaction_audio_mfcc')
    base_audio_mfcc = conf.get('base_audio_mfcc')


    alignment_bounds = create_reaction_alignment_bounds(reaction, conf['first_n_samples'])

    starting_points = range(0, len(base_audio) - chunk_size, step)

    minimums = [3 * sr]

    strokes = []
    active_strokes = []

    candidate_cache_file = os.path.join( conf.get('temp_directory'), f"{reaction.get('channel')}-start_cache-{minimums[0]}.pckl"   )

    if os.path.exists(candidate_cache_file):
        candidate_cache = read_object_from_file(candidate_cache_file)
    else:
        candidate_cache = {}
    
    for i,start in enumerate(starting_points):

        print(f"\tStroking...{i/len(starting_points)*100:.2f}%",end="\r")

        print_profiling()

        reaction_start = min(minimums[-18:])

        if start not in candidate_cache:

            reaction_end = get_bound(alignment_bounds, start, len(reaction_audio))

            chunk = base_audio[start:start+chunk_size]
            chunk_mfcc = base_audio_mfcc[:,round(start/hop_length):round((start+chunk_size)/hop_length)]

            candidates = find_segment_starts(
                                    reaction=reaction, 
                                    open_chunk=reaction_audio[reaction_start:reaction_end], 
                                    open_chunk_mfcc=reaction_audio_mfcc[:, round( reaction_start / hop_length):round(reaction_end/hop_length)],
                                    closed_chunk=chunk, 
                                    closed_chunk_mfcc= chunk_mfcc,
                                    current_chunk_size=chunk_size, 
                                    peak_tolerance=peak_tolerance, 
                                    full_search=True,
                                    open_start=reaction_start,
                                    closed_start=start, 
                                    distance=1 * sr, 
                                    filter_for_similarity=True)

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
    song_length = len(conf.get('base_audio_data'))
    return song_length - segment['end_points'][3] < allowed_spacing




#########################
# Drawing

def splay_paint(reaction, strokes, stroke_alpha, done=False, show_live=True, paths=None):
    plt.figure(figsize=(10, 10))

    x = conf.get('base_audio_data')
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
            alpha = .5
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

