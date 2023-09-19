import random, copy

from cross_expander.bounds import in_bounds, get_bound

from utilities import conversion_audio_sample_rate as sr
from utilities import conf, on_press_key

prune_types = {}
paths_visited = {}

def initialize_path_pruning():
    global prune_types
    prunes = [  "queue_prune",
                "best_score",
                "bounds",
                "spacing",
                "scope_cached",
                "continuity",
                'duplicate_path_prune',
                'mfcc_correlate_overlap',
                'manual_branch_prune'
              ]

    for prune_type in prunes:
        prune_types[prune_type] = 0


    global paths_visited
    paths_visited.clear()

    global last_checkpoint_cache
    last_checkpoint_cache.clear()


def should_prune_path(reaction, current_path, best_finished_path, current_start, reaction_start):
    global prune_types

    checkpoints = conf.get('checkpoints')
    reaction_audio = reaction.get('reaction_audio_data')
    base_audio_length = len(conf.get('base_audio_data'))

    depth = len(current_path)

    spacing_min_length = 2.5 * sr
    spacing_max_separation = 15 * sr
    path_without_fillers = [p for p in current_path if not p[-1]]
    if len(path_without_fillers) >= 3 and current_start < .75 * base_audio_length:
        first = path_without_fillers[-3]
        middle = path_without_fillers[-2]
        last = path_without_fillers[-1]
        if not first[-1] and not middle[-1] and not last[-1]: # ignore fillers
            # if first[1] - first[0] < spacing_min_length and middle[1] - middle[0] < spacing_min_length and last[1] - last[0] < spacing_min_length:
            if (first[1] - first[0]) + (middle[1] - middle[0]) + (last[1] - last[0]) < 3 * spacing_min_length:
                if middle[0] - first[1] > spacing_max_separation and last[0] - middle[1] > spacing_max_separation:
                    prune_types["spacing"] += 1
                    return True

    if len(current_path) > 0:
        path_key = str(current_path)
        if path_key in paths_visited:
            prune_types["duplicate_path_prune"] += 1
            # print('duplicate path')
            return True
        paths_visited[path_key] = 1

    alignment_bounds = conf.get('alignment_bounds')
    if alignment_bounds is not None:
        upper_bound = get_bound(alignment_bounds, current_start, len(reaction_audio))
        if not in_bounds(upper_bound, current_start, reaction_start):
            # print(f'\tBounds Prune! {current_start / sr} {reaction_start / sr} not in bounds (upper_bound={upper_bound}!')
            prune_types['bounds'] += 1
            # print('bounds prune')
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
