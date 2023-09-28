import os
import random
import matplotlib.pyplot as plt
import time
import numpy as np

from prettytable import PrettyTable

from utilities import conf, print_profiling, conversion_audio_sample_rate as sr

from cross_expander.pruning_search import print_prune_data, is_path_quality_poor, prune_types
from cross_expander.scoring_and_similarity import print_path, calculate_partial_score
from cross_expander.branching_search import branching_search


def align_by_checkpoint_probe(reaction):
    global print_when_possible
    global prune_types

    checkpoints = conf.get('checkpoints')
    base_audio = conf.get('base_audio_data')
    song_length = len(base_audio)


    low_score_queue = []
    paths = []


    print_profiling()

    print("Getting starting points")
    starting_points = []
    new_paths = branching_search(reaction, 
                                 current_path=[], 
                                 current_path_checkpoint_scores={}, 
                                 current_start=0, 
                                 reaction_start=0, 
                                 continuations=starting_points, 
                                 recursive=False, 
                                 peak_tolerance=conf.get('peak_tolerance')*.25)
                                            # if changing the initial peak tolerance, check with suicide/that'snotacting

    update_paths(reaction, paths, new_paths)

    checkpoints = [c for c in checkpoints]
    checkpoints.append(None)


    prunes_all = {
        'tight': 0,
        'light': 0,
        'location': 0,
        'diff_tight': 0,
        'diff_light': 0,
        'location_diff': 0,
        'poor_path_checkpoint': 0
    }

    for k,v in prune_types.items():
        prunes_all[k] = 0

    prunes = {}

    for idx,checkpoint_ts in enumerate(checkpoints): 
        past_the_checkpoint = []

        if checkpoint_ts is not None:
            print(f"\nProcessing checkpoint {round(checkpoint_ts / sr)}s ({idx / len(checkpoints)*100:.1f}%) of {reaction['channel']} / {conf.get('song_key')}")
        else: 
            print(f"To the end! ({idx / len(checkpoints)})")


        starting_points.sort(key=lambda x: x[3])
        for idx2, starting_point in enumerate(starting_points):
            path, checkpoint_scores, current_start, reaction_start = starting_point

            print(f"\t[{idx / len(checkpoints)*100:.1f}% / {100 * idx2 / len(starting_points):.1f}%] Reaction: {reaction_start / sr:.8f} Base: {current_start / sr:.4f} ... {len(past_the_checkpoint)} continuations", end="\r")

            if (checkpoint_ts and current_start >= checkpoint_ts):
                past_the_checkpoint.append([path, checkpoint_scores, current_start, reaction_start])

            else: 
                continuations = []
                new_paths = branching_search(reaction, current_path=path, current_path_checkpoint_scores=checkpoint_scores, current_start=current_start, reaction_start=reaction_start, continuations=continuations, recursive=True, probe_to_time=checkpoint_ts)
                update_paths(reaction, paths, new_paths)
                past_the_checkpoint.extend(continuations)
                # if abs( reaction_start / sr - 623.5551473922902 ) < .5:
                #     print(f"\t\t Interest! {current_start / sr} {reaction_start / sr}") 
                #     for path, checkpoint_scores, current_start, reaction_start in continuations:
                #         print(f"\t\t\t {current_start / sr}  {reaction_start / sr}", path[-1])

        print("")

        plot_relationships = False #or len(past_the_checkpoint) > 1000
        plot_candidate_paths = True

        if checkpoint_ts is None:
            # should be finished!
            if (len(past_the_checkpoint) > 0):
                print("ERRRORRRRR!!!! We have paths remaining at the end.")

        else: 

            if (len(past_the_checkpoint) == 0):
                print("ERRRORRRRR!!!! We didn't find any paths!")

            percent_through = 100 * checkpoint_ts / len(base_audio)

            
            for k,v in prunes_all.items():
                if k not in prune_types or k not in prunes:
                    prunes[k] = 0 

            # weed out candidates based on heuristics for poor path quality
            vetted_candidates = [c for c in past_the_checkpoint if checkpoint_ts > .95 * song_length or not is_path_quality_poor(reaction, c[0])]
            prunes['poor_path_checkpoint'] += len(past_the_checkpoint) - len(vetted_candidates)


            best_score_past_checkpoint = None
            best_value_past_checkpoint = 0 
            best_path_past_checkpoint = None
            best_candidate_past_the_checkpoint = None

            best_score_to_point = None
            best_value_to_point = 0 
            best_path_to_point = None
            best_candidate_to_point = None                 



            checkpoint_threshold_tight = checkpoint_threshold_tight_diff = .9  # Black Pegasus / Suicide can't go to .8 when light threshold = .5 (ok, that's only true w/o the path quality filters)
            
            #checkpoint_threshold_light = checkpoint_threshold_light_diff = .5    # > .15 doesn't work for Black Pegasus / Suicide in conjunction with high tight threshold (>.9)            
            checkpoint_threshold_light = .25 + .6 * (checkpoint_ts / song_length)    
            checkpoint_threshold_light_diff = .5    # > .15 doesn't work for Black Pegasus / Suicide in conjunction with high tight threshold (>.9)


            checkpoint_threshold_location = .95  
            location_rounding = 10


            # checkpoint_threshold_tight = .7  # Black Pegasus / Suicide can't go to .8 when light threshold = .5 (ok, that's only true w/o the path quality filters)
            # checkpoint_threshold_tight_diff = .7
            # checkpoint_threshold_light = .2 + .3 * (checkpoint_ts / song_length)    
            # checkpoint_threshold_light_diff = .5    # > .15 doesn't work for Black Pegasus / Suicide in conjunction with high tight threshold (>.9)
            # checkpoint_threshold_location = .6  
            # location_rounding = 10

            all_by_current_location = {}
            all_by_score = []

            candidates_to_plot = []

            checkpoints_reversed = [c for c in checkpoints if c and c <= checkpoint_ts][::-1]


            # if checkpoint_ts > 60 * sr: 
            #     start_at = round(checkpoint_ts - 60 * sr)
            # else: 
            #     start_at = 0
            start_at = 0

            candidates = []
            num_to_evaluate = 0
            for candidate in vetted_candidates:
                path, score_at_each_checkpoint, current_start, reaction_start = candidate
                reaction_end, score_at_checkpoint, truncated_path = calculate_partial_score(reaction, path, end=checkpoint_ts, start=start_at)
                candidates.append([candidate, reaction_end, score_at_checkpoint, truncated_path])
                score_at_each_checkpoint[checkpoint_ts] = score_at_checkpoint

                value = get_score_for_overall_comparison(score_at_checkpoint)

                if value > best_value_past_checkpoint: 
                    best_value_past_checkpoint = value
                    best_score_past_checkpoint = score_at_checkpoint
                    best_path_past_checkpoint = path
                    best_candidate_past_the_checkpoint = candidate


                passes_time = idx < len(checkpoints) - 1 and checkpoints[idx + 1] and path[-1][3] and path[-1][3] > checkpoints[idx + 1]
                if not passes_time:
                    num_to_evaluate += 1


            if num_to_evaluate == 0:
                starting_points = vetted_candidates

            else:
                candidates.sort( key=lambda x: (x[1], -x[2][0]) )  # sort by reaction_end, with a tie breaker of -overall score. 
                                                                   # The negative is so that the best score comes first, for 
                                                                   # pruning purposes.

                for expanded_candidate in candidates:
                    candidate, reaction_end, score_at_checkpoint, truncated_path = expanded_candidate
                    path, score_at_each_checkpoint, current_start, reaction_start = candidate

                    value = get_score_for_tight_bound(score_at_checkpoint)
                    if  value > best_value_to_point:
                        best_value_to_point = value
                        best_score_to_point = score_at_checkpoint
                        best_path_to_point = path
                        best_candidate_to_point = candidate                


                    # Don't prune out paths until we're at the checkpoint just before current_end of the latest path. 
                    # These are very low cost options to keep, and it gives them a chance to shine and not prematurely pruned.
                    passes_time = idx < len(checkpoints) - 1 and checkpoints[idx + 1] and path[-1][3] and path[-1][3] > checkpoints[idx + 1]

                    passes_tight = passes_time or checkpoint_threshold_tight <= get_score_ratio_at_checkpoint(reaction, score_at_checkpoint, best_score_to_point, get_score_for_tight_bound, checkpoint_ts) 
                    passes_mfcc = passes_time or checkpoint_threshold_light <= get_score_ratio_at_checkpoint(reaction, score_at_checkpoint, best_score_past_checkpoint, get_score_for_overall_comparison, checkpoint_ts)

                    if not passes_tight: 
                        prunes['tight'] += 1
                    if not passes_mfcc:
                        prunes['light'] += 1

                    if passes_tight and passes_mfcc:

                        passes_tight_diff = passes_time or checkpoint_threshold_tight_diff <= get_score_ratio_at_checkpoint_diff(checkpoints_reversed, reaction, candidate, best_candidate_to_point, get_score_for_tight_bound, checkpoint_ts) 
                        passes_mfcc_diff = passes_time or checkpoint_threshold_light_diff <= get_score_ratio_at_checkpoint_diff(checkpoints_reversed, reaction, candidate, best_candidate_past_the_checkpoint, get_score_for_overall_comparison, checkpoint_ts)

                        if not passes_tight_diff: 
                            prunes['diff_tight'] += 1
                        if not passes_mfcc_diff:
                            prunes['diff_light'] += 1

                        if passes_tight_diff and passes_mfcc_diff:
                            location = f"{int(location_rounding * current_start / sr)} {int(location_rounding * reaction_start / sr)}"
                            if location not in all_by_current_location:
                                all_by_current_location[location] = []
                            all_by_current_location[location].append( expanded_candidate  )

                    if plot_relationships: 
                        ideal_filter = passes_time or (passes_tight and passes_mfcc and passes_tight_diff and passes_mfcc_diff)
                        all_by_score.append([score_at_checkpoint, ideal_filter, path])



                best_past_the_checkpoint = []

                for location, candidates in all_by_current_location.items():
                    if len(candidates) < 2:
                        best_past_the_checkpoint.append(candidates[0][0])
                    else:
                        kept = 0
                        best_at_location = 0
                        best_score_at_location = None
                        best_candidate_at_location = None

                        for candidate, reaction_end, score, truncated_path in candidates:
                            if get_score_for_location(score) >= best_at_location:
                                best_at_location = get_score_for_location(score)
                                best_candidate_at_location = candidate
                                best_score_at_location = score

                        for candidate, reaction_end, score, truncated_path in candidates:
                            relative_current_end = candidate[0][-1][3]
                            passes_time = idx < len(checkpoints) - 1 and checkpoints[idx + 1] and relative_current_end and relative_current_end > checkpoints[idx + 1]
                            passes_location = get_score_for_location(score) / best_at_location >= checkpoint_threshold_location
                            if passes_time or passes_location:
                                passes_location_diff = checkpoint_threshold_location <= get_score_ratio_at_checkpoint_diff(checkpoints_reversed, reaction, candidate, best_candidate_at_location, get_score_for_location, checkpoint_ts) 
                                
                                if passes_location_diff:
                                    best_past_the_checkpoint.append(candidate)
                                else: 
                                    prunes['location_diff'] += 1
                            else: 
                                prunes['location'] += 1

                        if plot_candidate_paths: 
                            candidates_to_plot.append(  (best_candidate_at_location[0], best_score_at_location)   )

                if plot_candidate_paths:
                    candidates_to_plot.append( (best_path_past_checkpoint, best_score_past_checkpoint)  )


                if len(past_the_checkpoint) > 0:
                    print(f"\tKept {len(best_past_the_checkpoint)} of {len(past_the_checkpoint)} ({100 * len(best_past_the_checkpoint) / len(past_the_checkpoint):.1f}%)")
                    print(f"\tBest path at checkpoint:")
                    print_path(best_path_past_checkpoint, reaction)

                    # if idx > 0 and len(best_past_the_checkpoint) > 1000:
                    #     print("RANDOM")
                    #     for p in random.sample(best_past_the_checkpoint, 25):
                    #         print_path(p[0], reaction)


                    print("")

                    x = PrettyTable()
                    x.border = False
                    x.align = "r"
                    x.field_names = ['\t', 'Prune type', 'Checkpoint', 'Overall']
                    

                    for k,v in prunes.items():
                        if k in prune_types:
                            prunes[k] = prune_types[k] - prunes_all[k]
                            prunes_all[k] = prune_types[k]
                        else: 
                            prunes_all[k] += v   
                        x.add_row(['\t', k, prunes[k], prunes_all[k]])  

                        # print(f"\tPrune {k}: {v} [{prunes_all[k]}]")
                    print(x)



                if plot_relationships:
                    if len(past_the_checkpoint) > 1:
                        plot_candidates(reaction, all_by_score)


                if plot_candidate_paths and len(candidates_to_plot) > 0:
                    visualize_candidate_paths(reaction, idx, checkpoint_ts, candidates_to_plot, best_score_past_checkpoint, scoring_function=get_score_for_overall_comparison, done=idx==len(checkpoints) - 2, show_each_iteration=False)




                starting_points = best_past_the_checkpoint

    print_prune_data()


    return [p for p in paths if p]





def update_paths(reaction, paths, new_paths):
    for new_path in new_paths: 
        if new_path is not None:
            paths.append(new_path)

def get_score_ratio_at_checkpoint(reaction, score, best_score, scoring_function, checkpoint):
    if scoring_function(best_score) == 0:
        return 1

    return scoring_function(score) / scoring_function(best_score)


# Threshold have to increase as we get farther into the video, because pathways often share a big portion of 
# their pathways. Suboptimum pathway choices 80% of the way through aren't pruned because they're benefiting 
# from 80% good score from the shared optimal pathway. In this method, we look at the differential in score
# from when two path's scores last diverged.

def get_score_ratio_at_checkpoint_diff(checkpoints_reversed, reaction, candidate, best_candidate, scoring_function, checkpoint):
    path, checkpoint_scores, current_start, reaction_start = candidate        
    best_path, best_checkpoint_scores, best_current_start, best_reaction_start = best_candidate

    past_checkpoint_to_compare = None
    for past_checkpoint in checkpoints_reversed:
        if (checkpoint - past_checkpoint) > 6 * sr and abs(1 - scoring_function(checkpoint_scores[past_checkpoint]) / scoring_function(best_checkpoint_scores[past_checkpoint])) < .01:
            past_checkpoint_to_compare = past_checkpoint
            break

    if past_checkpoint_to_compare is None:
        # didn't find an equal score, so we can't use this method
        return 1

    score = scoring_function(checkpoint_scores[checkpoint])
    best_score = scoring_function(best_checkpoint_scores[checkpoint])

    previous_score = scoring_function(checkpoint_scores[past_checkpoint_to_compare])
    previous_best_score = scoring_function(best_checkpoint_scores[past_checkpoint_to_compare])

    if best_score == 0:
        return 1

    diff = score - previous_score
    best_diff = best_score - previous_best_score


    if diff >= best_diff:
        return 1

    if abs(best_diff) < 1:
        diff += 1
        best_diff += 1

    if best_diff < 0 and diff < 0:
        return best_diff / diff
    elif best_diff > 0 and diff > 0:
        return diff / best_diff
    elif diff < 0:
        return abs(diff - best_diff) /  (-2 * diff + best_diff)

    else: # best_diff < 0, can't happen
        assert(False)

    if best_diff < 0:

        return best_diff / diff
    else:
        return diff / best_diff




def get_score_for_tight_bound(score):
    return score[2] * score[3]

def get_score_for_overall_comparison(score):
    return score[2]

def get_score_for_location(score):
    return score[0]


def plot_candidates(reaction, data):
    
    # Separate the data into different lists for easier plotting
    # y = [item[0][2] for item in data] # plotting overall score

    x = [item[0][4] for item in data] # plotting overall mse similarity
    y = [item[0][5] for item in data] # plotting overall cosine similarity


    # x = [item[2][0][0] / sr for item in data]   # plotting scores of paths that start from a particular reaction_start 
    # x2 = [item[3] for item in data]   # plotting summed score (MSE)
    # y2 = [item[0][4] for item in data] # plotting overall mse similarity

    selected = [item[1] for item in data]

    #############
    # Create plot 1
    # plt.figure(figsize=(14, 10))
    plt.figure()
    plt.subplot(1, 2, 1)

    max_x = max(x)
    max_y = max(y)

    for i in range(len(y)):
        plt.scatter( i / len(y) * max_x, i / len(y) * max_y, color='blue'   )


    # Loop through the data to plot points color-coded by 'selected'
    for i in range(len(y)):

        if selected[i]:
            plt.scatter(x[i], y[i], color='yellow')
        else:
            plt.scatter(x[i], y[i], color='red')

    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Add labels and title
    plt.xlabel('segment cosine')
    plt.ylabel('full cosine')
    plt.title(f"{reaction.get('channel')} / {conf.get('song_key')}")


    # # ###########
    # # # create plot 2
    # plt.subplot(1, 2, 2)
    # y = y2
    # x = x2

    # max_x = max(x)
    # max_y = max(y)

    # for i in range(len(y)):
    #     plt.scatter( i / len(y) * max_x, i / len(y) * max_y, color='blue'   )


    # # Loop through the data to plot points color-coded by 'selected'
    # for i in range(len(y)):

    #     if selected[i]:
    #         plt.scatter(x[i], y[i], color='green')
    #     else:
    #         plt.scatter(x[i], y[i], color='red')

    # plt.xlim(left=0)
    # plt.ylim(bottom=0)

    # # Add labels and title
    # plt.xlabel('segment mse')
    # plt.ylabel('full mse')
    # plt.title(f"{reaction.get('channel')} / {conf.get('song_key')}")



    # Show the plot
    plt.show()


def visualize_candidate_paths(reaction, idx, checkpoint_ts, paths, best_score, scoring_function, done=False, show_each_iteration=False):
    plt.figure(figsize=(10, 10))

    y = conf.get('base_audio_data')
    x = reaction.get('reaction_audio_data')



    plt.axhline(y=checkpoint_ts / sr, color='black', linestyle='--')

    if reaction.get('ground_truth'):
        visualize_candidate_path(reaction.get('ground_truth_path'), color='green', linewidth=2)

    for path, score in paths:
        alpha = min(1, .2 + .8 * scoring_function(score) / scoring_function(best_score))
        visualize_candidate_path(path, alpha=alpha)

    # Calculate the time in seconds for each audio

    plt.xlabel("Time in React Audio [s]")
    plt.ylabel("Time in Base Audio [s]")
    # plt.legend()

    plt.title(f"{checkpoint_ts / sr} seconds of {reaction.get('channel')} / {conf.get('song_key')}")

    plt.xticks(np.arange(0, len(x) / sr + 30, 30))
    plt.yticks(np.arange(0, len(y) / sr + 30, 30))

    plt.grid(True)

    plt.tight_layout()

    # Save the plot instead of displaying it
    dirs = os.path.join(conf.get('temp_directory'), reaction.get('channel'))
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    if show_each_iteration:
        plt.show()

    filename = os.path.join(dirs, f"candidate_paths_frame_{idx:04d}.png")
    plt.savefig(filename)
    plt.close()

    if done or True: 
        video_filename = os.path.join(dirs, "video.mp4")
        compile_images_to_video(dirs, video_filename)





def visualize_candidate_path(path, color=None, linewidth=1, alpha=1):
    segment_color = color
    for i, segment in enumerate(path):

        reaction_start, reaction_end, current_start, current_end, filler = segment

        if color is None:
            segment_color = 'red'
            if filler:
                segment_color = 'purple'

        if i > 0:
            last_reaction_start, last_reaction_end, last_current_start, last_current_end, last_filler = previous_segment
            plt.plot( [last_reaction_end/sr, reaction_start/sr], [last_current_end/sr, current_start/sr], color=segment_color, linestyle='dashed', linewidth=linewidth, alpha=alpha)

        plt.plot( [reaction_start/sr, reaction_end/sr], [current_start/sr, current_end/sr]    , color=segment_color, linestyle='solid', linewidth=linewidth, alpha=alpha)

        previous_segment = segment
    


def compile_images_to_video(img_dir, video_filename, FPS=5):
    import cv2

    images = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(".png")])
    if not images:
        raise ValueError("No images found in the specified directory!")

    # Find out the frame width and height from the first image
    frame = cv2.imread(images[0])
    h, w, layers = frame.shape
    size = (w, h)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, size)  # 1 FPS

    for i in range(len(images)):
        img = cv2.imread(images[i])
        out.write(img)

    out.release()




