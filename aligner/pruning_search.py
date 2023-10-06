import random, copy

from aligner.bounds import in_bounds, get_bound
from aligner.scoring_and_similarity import get_segment_mfcc_cosine_similarity_score

from utilities import conversion_audio_sample_rate as sr
from utilities import conf, on_press_key

prune_types = {}
paths_visited = {}

def initialize_path_pruning():
    global prune_types
    prunes = [  
                "poor_path_branching", 
                "poor_path_fill",
                "poor_path_cosine",
                "bounds",
                "continuity",
                'duplicate_path_prune',
                'mfcc_correlate_overlap',
                "scope_cached",
                'exact_search_start_cache',
                'full_search_start_cache'
              ]

    for prune_type in prunes:
        prune_types[prune_type] = 0


    global paths_visited
    paths_visited.clear()

    global last_checkpoint_cache
    last_checkpoint_cache.clear()


def should_prune_path(reaction, current_path, current_start, reaction_start):
    global prune_types

    reaction_audio = reaction.get('reaction_audio_data')
    base_audio_length = len(conf.get('base_audio_data'))


    if current_start < .95 * base_audio_length and is_path_quality_poor(reaction, current_path):
        prune_types["poor_path_branching"] += 1
        return True

    if len(current_path) > 0:
        path_key = str(current_path)
        if path_key in paths_visited:
            prune_types["duplicate_path_prune"] += 1
            # print('duplicate path')
            return True
        paths_visited[path_key] = 1

    alignment_bounds = reaction['alignment_bounds']
    if alignment_bounds is not None:
        upper_bound = get_bound(alignment_bounds, current_start, len(reaction_audio))
        if not in_bounds(upper_bound, current_start, reaction_start):
            # print(f'\tBounds Prune! {current_start / sr} {reaction_start / sr} not in bounds (upper_bound={upper_bound}!')
            prune_types['bounds'] += 1
            # print('bounds prune')
            return True

    return False    






def is_path_quality_poor(reaction, path):
    global prune_types

    spacing_min_length = 2.5 * sr
    spacing_max_separation = 15 * sr
    path_without_fillers = [p for p in path if not p[-1]]
    if len(path_without_fillers) >= 3:
        first = path_without_fillers[-3]
        middle = path_without_fillers[-2]
        last = path_without_fillers[-1]
        if not first[-1] and not middle[-1] and not last[-1]: # ignore fillers
            # if first[1] - first[0] < spacing_min_length and middle[1] - middle[0] < spacing_min_length and last[1] - last[0] < spacing_min_length:
            if (first[1] - first[0]) + (middle[1] - middle[0]) + (last[1] - last[0]) < 3 * spacing_min_length:
                if middle[0] - first[1] > spacing_max_separation and last[0] - middle[1] > spacing_max_separation:
                    return True


    # Poor quality: 
    #  - reaction separation greater than .1s and less than 1s and no filler
    #  - three+ non filler segments less than 2.5 seconds with spacing between
    #  - first segment is less than 3s, followed by 60s+ of space...
    #      ...or first two segments are separated by .1s and average < 6s

    short_segments = 0
    short_separation = 0

    filler_duration = 0
    duration = 0

    for i, segment in enumerate(path): 
        (reaction_start, reaction_end, current_start, current_end, filler) = segment
        duration += current_end - current_start
        if filler:
            filler_duration += current_end - current_start
            continue

        if i > 0: 
            space_before = reaction_start - path[i-1][1]
            filler_before = path[i-1][4]
        else:
            filler_before = True
            space_before = 0

        if i < len(path) - 1:
            space_after = path[i+1][0] - reaction_end
            filler_after = path[i+1][4]
        else:
            filler_after = True
            space_after = 0 

        if i == 0 and len(path) > 1 and not filler_after and space_after > 60 * sr and (reaction_end - reaction_start) < 3 * sr:
            return True

        if i == 0 and len(path) > 2 and not filler_after and space_after > .1 * sr and path[i+2][0] - path[i+1][1] > .1 * sr and ((reaction_end - reaction_start) + (path[1][1] - path[1][0])) < 6 * sr:
            return True

        if (not filler_before and .1 * sr < space_before < sr) or (not filler_after and .1 * sr < space_after < sr):
            return True

        if (reaction_end - reaction_start) < 2.5 * sr:

            # we're not going to consider the last segment, because we might add to it the next time it branches
            if (i == 0 or space_before > sr / 10 ) and i < len(path) - 1 and space_after > sr / 10:
                short_segments += 1
                if short_segments >= 3:
                    return True



        cosine_sim = get_segment_mfcc_cosine_similarity_score(reaction, segment)
        if not filler and cosine_sim < .250:
            prune_types['poor_path_cosine'] += 1
            return True

    if filler_duration * sr  > (10 * len(conf.get('base_audio_data')) / 3): # no more than 10 seconds of filler per three minutes of base audio
        prune_types['poor_path_fill']
        return True 


    return False




last_checkpoint_cache = {}
def find_last_checkpoint_crossed(current_start):
    global last_checkpoint_cache
    checkpoints = conf.get('checkpoints')

    if current_start not in last_checkpoint_cache:
        previous_checkpoint = -1
        for checkpoint in checkpoints:
            if current_start < checkpoint:
                break
            previous_checkpoint = checkpoint
        last_checkpoint_cache[current_start] = previous_checkpoint

    return last_checkpoint_cache[current_start]


def print_prune_data():
    global prune_types

    checkpoints = conf.get('checkpoints')

    for k,v in prune_types.items():
        print(f"\t{k}: {v}")


def initialize_checkpoints(): 
    base_audio = conf.get('base_audio_data')

    samples_per_checkpoint = 2 * sr 

    timestamps = []
    start = s = 6
    while s < len(base_audio):
        if s / sr >= start:
            timestamps.append(s)
        s += samples_per_checkpoint

    return timestamps


on_press_key('ø', print_prune_data)
