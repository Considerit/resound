import os

from utilities import trim_and_concat_video, prepare_reactions, extract_audio, compute_precision_recall, universal_frame_rate, download_and_parse_reactions, is_close
from cross_expander import cross_expander_aligner
from face_finder import create_reactor_view
from backchannel_isolator import process_reactor_audio
from compositor import compose_reactor_compilation
from decimal import Decimal, getcontext

import cProfile
import pstats

ground_truth = {
    "CAN HE RAP THO？! ｜ Ren - The Hunger knox-truncated.mp4": [(0.0, 12.6), (80, 89), (123, 131), (156, 160), (173, 176), (189, 193), (235, 239), (247, 254.5), (286, 290), (342, 346), (373, 377), (442, 445), (477, 483), (513, 517), (546, 552), (570, 578), (599, 600), (632, 639), (645, 651), (662, 665), (675, 680), (694, 707), (734, 753)],
    "Rapper REACTS to REN - THE HUNGER anthony ray-truncated.mp4": [(0.5, 75), (604, 613), (658, 680), (724, 737), (760, 781), (1236, 1241)],
    "Black Pegasus.mp4": [(241, 306.4), (7*60+5, 7*60+54), (8*60+2, 10*60 + 14)] # for ren-suicide
}


def compress_segments(match_segments, sr, segment_combination_threshold):
    compressed_subsequences = []

    idx = 0 
    segment_groups = []
    current_group = []
    current_filler = match_segments[0][4]
    for current_start, current_end, current_base_start, current_base_end, filler in match_segments:
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

            if (start - current_end) / sr <= segment_combination_threshold:
                # This subsequence is continuous with the current one, extend it
                print("***COMBINING SEGMENT", current_end, start, (start - current_end), (start - current_end) / sr   )
                current_end = end
                current_base_end = base_end

                result = (current_base_end - current_base_start) / sr * universal_frame_rate()
                print(f"is new segment whole? Is {current_base_end - current_base_start} [{(result)}] divisible by frame rate? {is_close(result, round(result))} ")
            else:
                # This subsequence is not continuous, add the current one to the list and start a new one
                compressed_segment = (current_start, current_end, current_base_start, current_base_end, filler)
                if not filler:
                    assert( is_close(current_end - current_start, current_base_end - current_base_start) )
                compressed_subsequences.append( compressed_segment )
                current_start, current_end, current_base_start, current_base_end, _ = start, end, base_start, base_end, filler

        # Add the last subsequence
        compressed_subsequences.append((current_start, current_end, current_base_start, current_base_end, filler))
        # print(f"{end - start} vs {base_end - base_start}"      )
        if not filler:
            assert( is_close(current_end - current_start, current_base_end - current_base_start) )

    # compressed_subsequences = match_segments
    return compressed_subsequences

def create_aligned_reaction_video(react_video_ext, output_file: str, react_video, base_video, base_audio_data, base_audio_path, options):

    gt = ground_truth.get(os.path.basename(react_video) )
    # if not gt: 
    #     return

    options.setdefault("step_size", 1)
    options.setdefault("min_segment_length_in_seconds", 3)
    options.setdefault("first_match_length_multiplier", 1.5)
    options.setdefault("reverse_search_bound", options['first_match_length_multiplier'] * options['min_segment_length_in_seconds'])
    options.setdefault("segment_end_backoff", 20000)
    options.setdefault("segment_combination_threshold", .3)
    options.setdefault("peak_tolerance", .7)
    options.setdefault("expansion_tolerance", .7)


    segment_combination_threshold = options['segment_combination_threshold']
    del options['segment_combination_threshold']


    # Extract the reaction audio
    reaction_audio_data, reaction_sample_rate, reaction_audio_path = extract_audio(react_video)




    # Determine the number of decimal places to try avoiding frame boundary errors given python rounding issues
    fr = Decimal(universal_frame_rate())
    precision = Decimal(1) / fr
    precision_str = str(precision)
    getcontext().prec = len(precision_str.split('.')[-1])


    print(f"\n*******{options}")
    uncompressed_sequences = cross_expander_aligner(base_audio_data, reaction_audio_data, sr=reaction_sample_rate, **options)
    
    print('Uncompressed Sequences:')
    for sequence in uncompressed_sequences:
        print(f"\t{'*' if sequence[4] else ''}base: {float(sequence[2])}-{float(sequence[3])}  reaction: {float(sequence[0])}-{float(sequence[1])}      equal? {(sequence[1] - sequence[0])}=={(sequence[3] - sequence[2])} [{(sequence[1] - sequence[0]) - (sequence[3] - sequence[2])}]  whole? {(sequence[1] - sequence[0]) * fr}")



    sequences = compress_segments(uncompressed_sequences, segment_combination_threshold=segment_combination_threshold, sr=reaction_sample_rate)

    reaction_sample_rate = Decimal(reaction_sample_rate)

    final_sequences =          [ ( Decimal(s[0]) / reaction_sample_rate, Decimal(s[1]) / reaction_sample_rate, Decimal(s[2]) / reaction_sample_rate, Decimal(s[3]) / reaction_sample_rate, s[4]) for s in sequences ]

    print("\nsequences:")

    for sequence in final_sequences:
        print(f"\t{'*' if sequence[4] else ''}base: {float(sequence[2])}-{float(sequence[3])}  reaction: {float(sequence[0])}-{float(sequence[1])}      equal? {(sequence[1] - sequence[0])}=={(sequence[3] - sequence[2])} [{(sequence[1] - sequence[0]) - (sequence[3] - sequence[2])}]  whole? {(sequence[1] - sequence[0]) * fr}")

    # for i, sequence in enumerate(final_sequences):
    #     result = Decimal((sequence[1] - sequence[0]) * fr) 
    #     # assert( is_close(sequence[1] - sequence[0], sequence[3] - sequence[2])   ) # test if base sequence is equal to reaction sequence
        
    #     # if i < len(final_sequences) - 1:
    #     #     assert( is_close(result, Decimal(round(result))) ) # test if is a whole number


    if gt:
        compute_precision_recall(final_sequences, gt, tolerance=1.5)



    # Trim and align the reaction video
    trim_and_concat_video(react_video, final_sequences, base_video, output_file, react_video_ext)
    return output_file


def handle_reaction_video(output_dir: str, react_video, base_video, base_audio_data, base_audio_path, options):


    react_video_name, react_video_ext = os.path.splitext(react_video)
    output_file = os.path.join(output_dir, os.path.basename(react_video_name) + f"-CROSS-EXPANDER{react_video_ext}")

    # if '40' not in react_video_name:
    #     return

    print("processing ", react_video_name)
    # Create the output video file name


    if not os.path.exists(output_file):
        create_aligned_reaction_video(react_video_ext, output_file, react_video, base_video, base_audio_data, base_audio_path, options)

    _,_,aligned_reaction_audio_path = extract_audio(output_file)
    isolated_commentary = process_reactor_audio(aligned_reaction_audio_path, base_audio_path)
    faces = create_reactor_view(output_file, base_video, replacement_audio=isolated_commentary, show_facial_recognition=False)

    return faces





from moviepy.editor import VideoFileClip

def create_reaction_compilations(song_def:dict, output_dir: str = 'aligned', include_base_video = True, options = {}):
    song = song_def['title']
    song_directory = os.path.join('Media', song)
    reactions_dir = 'reactions'

    full_output_dir = os.path.join(song_directory, output_dir)
    if not os.path.exists(full_output_dir):
       # Create a new directory because it does not exist
       os.makedirs(full_output_dir)

    print("Processing directory", song_directory, reactions_dir, "Outputting to", output_dir)

    base_video, react_videos = prepare_reactions(song_directory)


    # Extract the base audio and get the sample rate
    base_audio_data, _, base_audio_path = extract_audio(base_video)


    reaction_dir = os.path.join(song_directory, reactions_dir)

    failed_reactions = []
    reactors = []
    for i, react_video in enumerate(react_videos):
        # if 'Cliff' not in react_video:
        #     continue

        try:
            # profiler = cProfile.Profile()
            # profiler.enable()



            faces = handle_reaction_video(full_output_dir, react_video, base_video, base_audio_data, base_audio_path, options)
            
            if len(faces) > 1:
                outputs = [{'file': f, 'group': i + 1} for f in faces]
            else:
                outputs = [{'file': f} for f in faces]

            reactors.extend(outputs)
            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
            # stats.print_stats()

        except Exception as e: 
            traceback.print_exc()
            print(e)
            failed_reactions.append((react_video, e))

    reaction_videos = []
    for reactor in reactors: 
        input_file = reactor['file']

        featured = False
        for featured_r in song_def['featured']:
          if featured_r in input_file:
            featured = True
            break

        reaction = {
            'key': input_file,
            'clip': VideoFileClip(input_file),
            'orientation': get_orientation(input_file),
            'group': reactor.get('group', None),
            'featured': featured
        }
        reaction_videos.append(reaction)

    base_video_for_compilation = VideoFileClip(base_video)
    compose_reactor_compilation(song_def, base_video_for_compilation, reaction_videos, os.path.join(song_directory, f"{song} (compilation).mp4"))

    return failed_reactions

def get_orientation(input_file):
    base_name, ext = os.path.splitext(input_file)
    parts = base_name.split('-')
    orientation = parts[-1] if len(parts) > 3 else 'center'
    return orientation


import traceback
if __name__ == '__main__':

    suicide = {
        'title': "Ren - Suicide",
        'include_base_video': True,
        'featured': ['ThatSingerReactions', 'Rosalie', 'JohnReavesLive']
    }

    fire = {
        'title': "Ren - Fire",
        'include_base_video': False,
        'featured': ['Johnnie Calloway', 'Anthony Ray']
    }

    hunger = {
        'title': "Ren - The Hunger",
        'include_base_video': True,
        'featured': ['h8tful', 'jamel', '_QlkLhbCeNo']
    }

    genesis = {
        'title': "Ren - Genesis",
        'include_base_video': True,
        'featured': ['Jamel', 'J Rizzle', 'That singer reacts']
    }

    # download_and_parse_reactions("Ren - Suicide")

    songs = [suicide, fire, hunger, genesis]

    failures = []
    for song in songs: 
        failed = create_reaction_compilations(song, output_dir = "crossed-backoff-0", options={'segment_end_backoff': 0, 'segment_combination_threshold': 0})
        if(len(failed) > 0):
            failures.append((song, failed)) 





    print(f"\n\nDone! {len(failures)} songs did not finish")

    for song, failed in failures:
        print(f"\n\n {len(failed)} Failures for song {song}")
        for react_video, e in failed:
            print(f"\n***{react_video} failed with:")
            print(e)







