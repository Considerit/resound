import os
import copy

from utilities import prepare_reactions, extract_audio, download_and_parse_reactions
from cross_expander import create_aligned_reaction_video
from face_finder import create_reactor_view
from backchannel_isolator import process_reactor_audio
from compositor import compose_reactor_compilation

import cProfile
import pstats






def handle_reaction_video(song:dict, output_dir: str, react_video, base_video, base_audio_data, base_audio_path, options, extend_by=15):


    react_video_name, react_video_ext = os.path.splitext(react_video)
    output_file = os.path.join(output_dir, os.path.basename(react_video_name) + f"-CROSS-EXPANDER{react_video_ext}")

    # if '40' not in react_video_name:
    #     return

    print("processing ", react_video_name)
    # Create the output video file name


    create_aligned_reaction_video(song, react_video_ext, output_file, react_video, base_video, base_audio_data, base_audio_path, options, extend_by=extend_by)

    if not options["isolate_commentary"]:
        return []

    _,sr,aligned_reaction_audio_path = extract_audio(output_file)
    isolated_commentary = process_reactor_audio(aligned_reaction_audio_path, base_audio_path, extended_by=extend_by, sr=sr)
    
    if not options["create_reactor_view"]:
        return []

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

            faces = handle_reaction_video(song_def, full_output_dir, react_video, base_video, base_audio_data, base_audio_path, copy.deepcopy(options))
            
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

    if len(reaction_videos) > 0 and options['create_compilation']:
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
        'featured': ['ThatSingerReactions', 'Rosalie', 'JohnReavesLive'],
        'ground_truth': {
            "Black Pegasus.mp4": [(241, 306.4), (7*60+5, 7*60+54), (8*60+2, 10*60 + 14)],
            "That’s Not Acting Either.mp4": [(75, 129), (192, 282), (7*60+30, 9*60 + 7)]            
        }

    }

    fire = {
        'title': "Ren - Fire",
        'include_base_video': False,
        'featured': ['Johnnie Calloway', 'Anthony Ray']
    }

    hunger = {
        'title': "Ren - The Hunger",
        'include_base_video': True,
        'featured': ['h8tful', 'jamel', '_QlkLhbCeNo'],
        'ground_truth': {
            "CAN HE RAP THO？! ｜ Ren - The Hunger knox-truncated.mp4": [(0.0, 12.6), (80, 89), (123, 131), (156, 160), (173, 176), (189, 193), (235, 239), (247, 254.5), (286, 290), (342, 346), (373, 377), (442, 445), (477, 483), (513, 517), (546, 552), (570, 578), (599, 600), (632, 639), (645, 651), (662, 665), (675, 680), (694, 707), (734, 753)],
            "Rapper REACTS to REN - THE HUNGER anthony ray-truncated.mp4": [(0.5, 75), (604, 613), (658, 680), (724, 737), (760, 781), (1236, 1241)],
            "REN-HUNGER [REACTION] h8tful jay-truncated.mp4": [(0, 12.75), (28, 36), (49, 51), (88, 90), (104, 112), (135, 140), (160, 178), (195, 200), (227, 238), (254, 260), (284, 298), (319, 330), (355, 361), (371, 408)],

        }

    }

    genesis = {
        'title': "Ren - Genesis",
        'include_base_video': True,
        'featured': ['Jamel', 'J Rizzle', 'That singer reacts']
    }


    # download_and_parse_reactions("Ren - Suicide")

    songs = [hunger, genesis, fire, suicide]



    fired = {
        'title': "Ren - Fired",
        'include_base_video': False,
        'featured': ['Johnnie Calloway', 'Anthony Ray']
    }

    # songs = [suicide]

    output_dir = "cheetah"
    # output_dir = "processed"


    options = {
        "output_alignment_metadata": True,
        "output_alignment_video": True,
        "isolate_commentary": True,
        "create_reactor_view": True,
        "create_compilation": True
    }



    failures = []
    for song in songs: 
        failed = create_reaction_compilations(song, output_dir = output_dir, options=options)
        if(len(failed) > 0):
            failures.append((song, failed)) 





    print(f"\n\nDone! {len(failures)} songs did not finish")

    for song, failed in failures:
        print(f"\n\n {len(failed)} Failures for song {song}")
        for react_video, e in failed:
            print(f"\n***{react_video} failed with:")
            print(e)







