

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

from aligner.bounds import create_reaction_alignment_bounds, print_alignment_bounds
from aligner.path_painter import paint_paths


def create_aligned_reaction_video(reaction, extend_by = 0):
    global conf

    output_file = reaction.get('aligned_path')



    if conf['create_alignment'] or conf['alignment_test']:
        alignment_metadata_file = os.path.splitext(output_file)[0] + '.json'

        if not os.path.exists(alignment_metadata_file):
            conf['load_reaction'](reaction['channel'])
            

            # Determine the number of decimal places to try avoiding frame boundary errors given python rounding issues
            fr = Decimal(universal_frame_rate())
            precision = Decimal(1) / fr
            precision_str = str(precision)
            getcontext().prec = len(precision_str.split('.')[-1])

            start = time.perf_counter()

            best_path = paint_paths(reaction)



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



        # Sort of a migration here. If there's a short filler at the beginning, instead merge it with 
        # first path (and extend appropriately)
        first_segment = metadata['best_path'][0]
        if len(first_segment) == 6:
            (rs,re,bs,be,filler,__) = first_segment
        else: 
            (rs,re,bs,be,filler) = first_segment
        if filler and (re-rs) / sr < 4:
            metadata['best_path'].pop(0)
            metadata['best_path'][0][0] -= (re-rs)
            metadata['best_path'][0][2] = bs



        reaction.update(metadata)

        if not os.path.exists(output_file) and conf["output_alignment_video"]:
            conf['load_reaction'](reaction['channel'])

            react_video = reaction.get('video_path')
            base_video = conf.get('base_video_path')

            reaction_sample_rate = Decimal(sr)
            best_path_converted = [ ( Decimal(s[0]) / reaction_sample_rate, Decimal(s[1]) / reaction_sample_rate, Decimal(s[2]) / reaction_sample_rate, Decimal(s[3]) / reaction_sample_rate, s[4]) for s in reaction['best_path'] ]

            trim_and_concat_video(reaction, react_video, best_path_converted, base_video, output_file, extend_by = extend_by, use_fill = conf.get('include_base_video', True))
        

    return output_file







