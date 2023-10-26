

import os
import time

from decimal import Decimal, getcontext

from aligner.create_trimmed_video import trim_and_concat_video

from utilities import compute_precision_recall, universal_frame_rate, is_close
from utilities import conf, save_object_to_file, read_object_from_file
from utilities import conversion_audio_sample_rate as sr

from aligner.pruning_search import initialize_path_pruning, initialize_checkpoints
from aligner.scoring_and_similarity import path_score, find_best_path, initialize_path_score, initialize_segment_tracking, print_path
from aligner.find_segment_start import initialize_segment_start_cache
from aligner.find_segment_end import initialize_segment_end_cache

from aligner.checkpoint_probe import align_by_checkpoint_probe

from aligner.bounds import create_reaction_alignment_bounds, print_alignment_bounds


def find_alignments(reaction):
    # global best_finished_path
    global print_when_possible

    initialize_segment_start_cache(reaction)
    initialize_segment_end_cache()
    initialize_path_score()
    initialize_path_pruning()
    initialize_segment_tracking()


    # best_finished_path.clear()

    reaction['alignment_bounds'] = create_reaction_alignment_bounds(reaction, conf['first_n_samples'])

    


    paths = align_by_checkpoint_probe(reaction)

    return paths



def cross_expander_aligner(reaction):
    global conf

    conf['checkpoints'] = initialize_checkpoints()

    paths = find_alignments(reaction)

    path = find_best_path(reaction, paths)

    sequences = compress_segments(path)

    ground_truth = reaction.get('ground_truth')
    if ground_truth:
        compute_precision_recall(sequences, ground_truth, tolerance=1.5)

    return sequences



def create_aligned_reaction_video(reaction, extend_by = 0):
    global conf

    output_file = reaction.get('aligned_path')



    if conf['create_alignment'] or conf['alignment_test']:
        alignment_metadata_file = os.path.splitext(output_file)[0] + '.pckl'

        if not os.path.exists(alignment_metadata_file):
            conf['load_reaction'](reaction['channel'])
            

            # Determine the number of decimal places to try avoiding frame boundary errors given python rounding issues
            fr = Decimal(universal_frame_rate())
            precision = Decimal(1) / fr
            precision_str = str(precision)
            getcontext().prec = len(precision_str.split('.')[-1])

            start = time.perf_counter()



            if conf.get('paint_paths'):
                from aligner.path_painter import paint_paths
                best_path = paint_paths(reaction)
            else:
                best_path = cross_expander_aligner(reaction)



            alignment_duration = (time.perf_counter() - start) / 60 # in minutes
            best_path_score = path_score(best_path, reaction)

            best_path_output = print_path(best_path, reaction).get_string()

            metadata = {
                'best_path_score': best_path_score,
                'best_path': best_path,
                'best_path_output': best_path_output,
                'alignment_duration': alignment_duration
            }                

            if conf['save_alignment_metadata']:
                save_object_to_file(alignment_metadata_file, metadata)
        else: 
            metadata = read_object_from_file(alignment_metadata_file)

        reaction.update(metadata)

        if not os.path.exists(output_file) and conf["output_alignment_video"]:
            conf['load_reaction'](reaction['channel'])

            react_video = reaction.get('video_path')
            base_video = conf.get('base_video_path')

            reaction_sample_rate = Decimal(sr)
            best_path_converted = [ ( Decimal(s[0]) / reaction_sample_rate, Decimal(s[1]) / reaction_sample_rate, Decimal(s[2]) / reaction_sample_rate, Decimal(s[3]) / reaction_sample_rate, s[4]) for s in reaction['best_path'] ]

            trim_and_concat_video(reaction, react_video, best_path_converted, base_video, output_file, extend_by = extend_by, use_fill = conf.get('include_base_video', True))
        

    return output_file




def compress_segments(match_segments):
    compressed_subsequences = []

    
    idx = 0 
    segment_groups = []
    current_group = []
    current_filler = match_segments[0][4]
    for segment in match_segments:
        if len(segment) == 5:
            current_start, current_end, current_base_start, current_base_end, filler = segment
        else: 
            current_start, current_end, current_base_start, current_base_end, filler, strokes = segment
            
        if filler != current_filler:
            if len(current_group) > 0:
                segment_groups.append(current_group)
                current_group = []
            segment_groups.append([(current_start, current_end, current_base_start, current_base_end, filler)])
            current_filler = filler
        else: 
            current_group.append((current_start, current_end, current_base_start, current_base_end, filler))

    if len(current_group) > 0:
        segment_groups.append(current_group)


    for group in segment_groups:

        if len(group) == 1:
            compressed_subsequences.append(group[0])
            continue

        current_start, current_end, current_base_start, current_base_end, filler = group[0]
        for i, (start, end, base_start, base_end, filler) in enumerate(group[1:]):
            if start - current_end <= 1:
                # This subsequence is continuous with the current one, extend it
                # print("***COMBINING SEGMENT", current_end, start, (start - current_end), (start - current_end) / sr   )
                current_end = end
                current_base_end = base_end
            else:
                # This subsequence is not continuous, add the current one to the list and start a new one
                compressed_segment = (current_start, current_end, current_base_start, current_base_end, filler)
                # if not filler:
                #     print('not contiguous', current_end, start, current_base_end, base_start)
                #     # assert( is_close(current_end - current_start, current_base_end - current_base_start) )
                compressed_subsequences.append( compressed_segment )
                current_start, current_end, current_base_start, current_base_end, _ = start, end, base_start, base_end, filler

        # Add the last subsequence
        compressed_subsequences.append((current_start, current_end, current_base_start, current_base_end, filler))
        # print(f"{end - start} vs {base_end - base_start}"      )
        # if not filler:
        #     if not is_close(current_end - current_start, current_base_end - current_base_start):
        #         print("NOT CLOSE!!!! Possible error", current_start, current_end - current_start, current_base_end - current_base_start)

    # compressed_subsequences = match_segments
    return compressed_subsequences

