import random, copy
from typing import List, Tuple

from cross_expander.scoring_and_similarity import calculate_partial_score, path_score
from cross_expander.bounds import in_bounds, get_bound


prune_types = {}
paths_by_checkpoint = {}
paths_from_current_start = {}
paths_visited = {}

def initialize_path_pruning():
    global prune_types
    prunes = [  "checkpoint",
                "best_score",
                "exact",
                "bounds",
                "scope_cached",
                "combinatorial",
                "continuity",
                'duplicate_path_prune'
              ]

    for prune_type in prunes:
        prune_types[prune_type] = 0


    global paths_by_checkpoint
    paths_by_checkpoint.clear()

    global paths_from_current_start
    paths_from_current_start.clear()

    global paths_visited
    paths_visited.clear()


def should_prune_path(basics, options, current_path, current_path_checkpoint_scores, best_finished_path, current_start, reaction_start, path_counts):
    global paths_by_checkpoint
    global paths_from_current_start

    checkpoints = basics.get('checkpoints')
    reaction_audio = basics.get('reaction_audio')
    sr = basics.get('sr')

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

    path_key = str(current_path)
    if path_key in paths_visited:
        prune_types["duplicate_path_prune"] += 1
        return True
    paths_visited[path_key] = 1

    if depth > 3:
        prior_current_start = current_path[-1][2]
        prior_prior_current_start = current_path[-2][2]
        if prior_current_start in path_counts[depth - 1]['current_starts'] and path_counts[depth - 1]['current_starts'][prior_current_start] > 1:  # path_counts[depth]['completed'] > 250:
            check_depth = depth - 1
            while check_depth > 3 and depth - check_depth < 10:
                new_checkpoint = current_path[check_depth - depth][2]
                add_new_checkpoint(checkpoints, new_checkpoint, basics)
                check_depth -= 1


        if False and prior_prior_current_start in path_counts[depth - 2]['current_starts'] and max_visited > 2500 and path_counts[depth - 2]['current_starts'][prior_prior_current_start] > 5:  # path_counts[depth]['completed'] > 250:
            prune_types["combinatorial"] += 1
            return True

    # aggressive prune based on scores after having passed a checkpoint 
    if depth > 0:
        should_prune = check_if_prune_at_nearest_checkpoint(current_path, current_path_checkpoint_scores, best_finished_path, current_start, basics)
        if should_prune and max_visited > 2500:
            prune_types[should_prune] += 1
            return True

    # specific prune based on exact match of current start
    if False and depth > 0:
        score = None
        if current_start not in paths_from_current_start: 
            paths_from_current_start[current_start] = []
        else: 
            score = path_score(current_path, basics, relative_to = current_start) 
            for i, (comp_reaction_start, comp_path, comp_score) in enumerate(paths_from_current_start[current_start]):
                if comp_reaction_start <= reaction_start:
                    if comp_score is None:
                        comp_score = path_score(comp_path, basics, relative_to = current_start)
                        paths_from_current_start[current_start][i][2] = comp_score

                    ts_thresh_contrib = min( current_start / (2 * 60 * sr), .09)
                    prune_threshold = .9 + ts_thresh_contrib

                    if score[0] < prune_threshold * comp_score[0] and max_visited > 1000:
                        # print(f"\tExact Prune! {comp_reaction_start} {comp_score[0]}  >  {reaction_start} {score[0]} @ threshold {prune_threshold}")
                        prune_types['exact'] += 1
                        return True      

        paths_from_current_start[current_start].append( [reaction_start, copy.deepcopy(current_path), score] )


    alignment_bounds = options.get('alignment_bounds')
    if alignment_bounds is not None:
        upper_bound = get_bound(alignment_bounds, current_start, len(reaction_audio))
        if not in_bounds(upper_bound, current_start, reaction_start):
            # print(f'\tBounds Prune! {current_start / sr} {reaction_start / sr} not in bounds (upper_bound={upper_bound}!')
            prune_types['bounds'] += 1
            return True

    return False


def check_if_prune_at_nearest_checkpoint(current_path, current_path_checkpoint_scores, best_finished_path, current_start, basics):
    base_audio = basics.get('base_audio')
    sr = basics.get('sr')
    new_checkpoint_every_n_prunes = 5
    checkpoints = basics.get('checkpoints')
    len_audio = len(base_audio)

    global paths_by_checkpoint

    if len(current_path) == 0:
        print("Weird zero length path!")
        return 'checkpoint'


    # go through all checkpoints we've passed
    for its, current_ts in enumerate(checkpoints):
        if current_start < current_ts: 
            break

        if current_ts in current_path_checkpoint_scores and current_path_checkpoint_scores[current_ts]:
            continue

        partial_score = calculate_partial_score(current_path, current_ts, basics)
        if partial_score is None:
            return 'checkpoint'

        adjusted_reaction_end, current_score = partial_score

        current_path_checkpoint_scores[current_ts] = True #[current_ts, current_score, adjusted_reaction_end]

        if current_ts not in paths_by_checkpoint:
            paths_by_checkpoint[current_ts] = {'prunes_here': 0, 'paths': []}

        prunes_here = paths_by_checkpoint[current_ts]['prunes_here']



        # if current_ts > len_audio * .75:
        #     confidence = .95
        # else: 
        #     confidence = .45 + .499 * (  current_ts / (len_audio * .75) )

        if current_ts > len_audio / 2:
            confidence = .99
        else: 
            confidence = .5 + .499 * (  current_ts / (len_audio / 2) )

        if 'score' in best_finished_path:
            if current_ts not in best_finished_path['partials']:
                _, best_at_checkpoint = calculate_partial_score(best_finished_path['path'], current_ts, basics)
                best_finished_path['partials'][current_ts] = best_at_checkpoint
            best_at_checkpoint = best_finished_path['partials'][current_ts]

            if confidence * best_at_checkpoint[2] * best_at_checkpoint[3] * best_at_checkpoint[3] > current_score[2] * current_score[3] * current_score[3]: 
                paths_by_checkpoint[current_ts]['prunes_here'] += 1
                return 'best_score'

        # ts_thresh_contrib = min( current_ts / (3 * 60 * sr), .1)
        # prunes_thresh_contrib = min( .04 * prunes_here / 50, .04 )
        # prunes_thresh_contrib = 0

        # prune_threshold = .85 + ts_thresh_contrib + prunes_thresh_contrib


        for comp_reaction_end, comp_score, ppath in paths_by_checkpoint[current_ts]['paths']:
            # print(f"\t{comp_reaction_end} <= {adjusted_reaction_end}?")
            if comp_reaction_end <= adjusted_reaction_end:

                # if current_score[0] < prune_threshold * comp_score[0]:
                if confidence * comp_score[2] * comp_score[3] * comp_score[3] > current_score[2] * current_score[3] * current_score[3]: 
                    paths_by_checkpoint[current_ts]['prunes_here'] += 1 # increment prunes at this checkpoint
                    # print(f"\tCheckpoint Prune at {current_ts / sr}: {current_score[0]} compared to {comp_score[0]}. Prunes here: {paths_by_checkpoint[current_ts]['prunes_here']} @ thresh {prune_threshold}")
                    
                    return 'checkpoint'

        paths_by_checkpoint[current_ts]['paths'].append( (adjusted_reaction_end, current_score, list(current_path)) )
        prune_paths_at_checkpoint(current_ts, paths_by_checkpoint)

            
    # if random.random() < .01:
    #     prune_paths_at_checkpoints()

    return False

def prune_paths_at_checkpoint(checkpoint, paths_by_checkpoint):
    prune_data = paths_by_checkpoint[checkpoint]
    paths = prune_data["paths"]
    new_paths = []

    paths.sort(key=lambda x: x[0])

    original_path_length = path_length = len(paths)
    new_path_length = -1

    while path_length != new_path_length:
        path_length = len(paths)
        paths[:] = [path for i,path in enumerate(paths) if i == 0 or path[1] > paths[i-1][1]]
        new_path_length = len(paths)

    # print(f"\t{ts}: from {original_path_length} to {new_path_length}")

def prune_paths_at_checkpoints():
    global paths_by_checkpoint

    # print("##############")
    # print("Cleaning out prune paths")
    # print('##############')

    for ts, prune_data in paths_by_checkpoint.items():
        prune_paths_at_checkpoint(ts, paths_by_checkpoint)



def print_prune_data(basics):
    global paths_by_checkpoint
    global prune_types

    sr = basics.get('sr')
    checkpoints = basics.get('checkpoints')

    for k,v in prune_types.items():
        print(f"\t{k}: {v}")


    for tss in checkpoints:
        if tss in paths_by_checkpoint:
            prunes = paths_by_checkpoint[tss]['prunes_here']
        else:
            prunes = "<nil>"
        print(f"\t\t{tss / sr}: {prunes}")    


def initialize_checkpoints(basics): 
    base_audio = basics.get('base_audio')
    sr = basics.get('sr')

    samples_per_checkpoint = 10 * sr 

    timestamps = []
    s = samples_per_checkpoint
    while s < len(base_audio):
        if s / basics.get('sr') >= 10:
            timestamps.append(s)
        s += samples_per_checkpoint

    return timestamps


def add_new_checkpoint(checkpoints, current_start, basics):

    if current_start in checkpoints or current_start < checkpoints[0]: 
        return

    global paths_by_checkpoint

    previous_checkpoint = -1
    for idx, checkpoint in enumerate(checkpoints):
        if checkpoint > current_start:
            break
        else:
            previous_checkpoint = checkpoint

    # don't add a checkpoint if it is less than a second after a previous checkpoint
    if (current_start - previous_checkpoint) / basics.get('sr') < 1: 
        return

    checkpoints.append(current_start)
    checkpoints.sort()

    idx = checkpoints.index(current_start)

    if idx < len(checkpoints) - 1:


        reference_checkpoint = checkpoints[idx + 1]

        if reference_checkpoint not in paths_by_checkpoint:
            paths_by_checkpoint[reference_checkpoint] = {'prunes_here': 0, 'paths': []}

        for rs, scr, current_path in paths_by_checkpoint[reference_checkpoint]["paths"]:
            partial_score = calculate_partial_score(current_path, current_start, basics)
            if partial_score is None:
                continue
            if current_start not in paths_by_checkpoint:
                paths_by_checkpoint[current_start] = {'prunes_here': 0, 'paths': []}

            paths_by_checkpoint[current_start]['paths'].append( (partial_score[0], partial_score[1], list(current_path))  )


        # If the reference checkpoint is within a second, we're going to remove it. Effectively,
        # we'll replace the reference checkpoint with the new checkpoint. This helps keep the 
        # checkpoints spaced out, as similarity measures are costly. 
        if (reference_checkpoint - current_start) / basics.get('sr') < .5:
            del paths_by_checkpoint[reference_checkpoint]
            checkpoints.pop(idx + 1)
