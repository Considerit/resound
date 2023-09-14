import random, copy

from cross_expander.bounds import in_bounds, get_bound

from utilities import conversion_audio_sample_rate as sr
from utilities import conf 

prune_types = {}
paths_visited = {}

def initialize_path_pruning():
    global prune_types
    prunes = [  "queue_prune",
                "best_score",
                "bounds",
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


def should_prune_path(reaction, current_path, current_path_checkpoint_scores, best_finished_path, current_start, reaction_start, path_counts):
    global prune_types

    checkpoints = conf.get('checkpoints')
    reaction_audio = reaction.get('reaction_audio_data')

    depth = len(current_path)

    max_visited = 0 
    for ddd, cnt in path_counts.items():
        if cnt['open'] > max_visited:
            max_visited = cnt['open']

    total_visited = path_counts[-1]['open']

    if current_start not in path_counts[depth]['current_starts']:
        path_counts[depth]['current_starts'][current_start] = 0
    path_counts[depth]['current_starts'][current_start] += 1

    # if random.random() < .001:
    #     depths = list(path_counts.keys())
    #     depths.sort()
    #     print("***********************")
    #     print("Current_starts by depth")
    #     for ddepth in depths:
    #         print(f"\t{ddepth}:")
    #         starts = list(path_counts[ddepth]['current_starts'].keys())
    #         starts.sort()
    #         for sstart in starts:
    #             cnt = path_counts[ddepth]['current_starts'][sstart]
    #             print(f"\t\t{sstart} [{sstart / sr:.1f}]: {path_counts[ddepth]['current_starts'][sstart]}")

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
    s = 6
    while s < len(base_audio):
        if s / sr >= 6:
            timestamps.append(s)
        s += samples_per_checkpoint

    return timestamps
