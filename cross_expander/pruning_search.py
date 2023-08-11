import random

from cross_expander.scoring_and_similarity import calculate_partial_score


prune_types = {}

def initialize_prune_types(): 
    global prune_types
    prunes = [  "checkpoint",
                "best_score",
                "exact",
                "bounds",
                "length",
                "cached",
                "scope_cached",
                "combinatorial",
                "continuity"
              ]

    for prune_type in prunes:
        prune_types[prune_type] = 0


def initialize_checkpoints(basics): 
    base_audio = basics.get('base_audio')
    sr = basics.get('sr')

    samples_per_checkpoint = 10 * sr 

    timestamps = []
    s = samples_per_checkpoint
    while s < len(base_audio):
        if s / basics.get('sr') >= 30:
            timestamps.append(s)
        s += samples_per_checkpoint

    return timestamps


def add_new_checkpoint(checkpoints, current_start, paths_by_checkpoint, basics):
    if current_start in checkpoints or current_start / basics.get('sr') < 30: 
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



def check_if_prune_at_nearest_checkpoint(current_path, current_path_checkpoint_scores, paths_by_checkpoint, best_finished_path, current_start, basics):
    base_audio = basics.get('base_audio')
    sr = basics.get('sr')
    new_checkpoint_every_n_prunes = 5
    checkpoints = basics.get('checkpoints')
    len_audio = len(base_audio)

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


        if 'score' in best_finished_path:
            if current_ts not in best_finished_path['partials']:
                _, best_at_checkpoint = calculate_partial_score(best_finished_path['path'], current_ts, basics)
                best_finished_path['partials'][current_ts] = best_at_checkpoint
            best_at_checkpoint = best_finished_path['partials'][current_ts]

            if current_ts > len_audio / 2:
                confidence = .99
            else: 
                confidence = .5 + .499 * (  current_ts / (len_audio / 2) )

            if random.random() < .001:
                print(f"Best score is {best_finished_path['score']}, at {current_ts / sr:.1f} comparing:")
                print(f"{best_at_checkpoint}")
                print(f"{current_score}")
                print(f"({100 * current_score[2] * current_score[3] * current_score[3] / (best_at_checkpoint[2] * best_at_checkpoint[3] * best_at_checkpoint[3]):.1f}%) ")

            if confidence * best_at_checkpoint[2] * best_at_checkpoint[3] * best_at_checkpoint[3] > current_score[2] * current_score[3] * current_score[3]: 
                paths_by_checkpoint[current_ts]['prunes_here'] += 1
                return 'best_score'

        ts_thresh_contrib = min( current_ts / (3 * 60 * sr), .1)
        prunes_thresh_contrib = min( .04 * prunes_here / 50, .04 )

        prune_threshold = .85 + ts_thresh_contrib + prunes_thresh_contrib


        full_comp_score = None
        for comp_reaction_end, comp_score, ppath in paths_by_checkpoint[current_ts]['paths']:
            # print(f"\t{comp_reaction_end} <= {adjusted_reaction_end}?")
            if comp_reaction_end <= adjusted_reaction_end:
                # (cs1,cs2,cs3) = comp_score
                # (s1,s2,s3) = current_score

                # m1 = max(cs1,s1); m2 = max(cs2,s2); m3 = max(cs3,s3)

                # full_comp_score = cs1 / m1 + cs2 / m2 + cs3 / m3
                # full_score      =  s1 / m1 +  s2 / m2 +  s3 / m3
                # print(f"\t\t{full_score} < {.9 * full_comp_score}?")

                if current_score[0] < prune_threshold * comp_score[0]:
                    paths_by_checkpoint[current_ts]['prunes_here'] += 1 # increment prunes at this checkpoint
                    # print(f"\tCheckpoint Prune at {current_ts / sr}: {current_score[0]} compared to {comp_score[0]}. Prunes here: {paths_by_checkpoint[current_ts]['prunes_here']} @ thresh {prune_threshold}")
                    
                    return 'checkpoint'

        paths_by_checkpoint[current_ts]['paths'].append( (adjusted_reaction_end, current_score, list(current_path)) )


        # print("no prune", paths_by_checkpoint[current_ts])

            
    if random.random() < .01:
        prune_path_prunes(paths_by_checkpoint)

    return False

def prune_path_prunes(paths_by_checkpoint):
    # print("##############")
    # print("Cleaning out prune paths")
    # print('##############')

    for ts, prune_data in paths_by_checkpoint.items():
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

