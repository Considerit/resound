import os
import copy

from utilities import prepare_reactions, extract_audio
from inventory import download_and_parse_reactions, get_manifest_path
from cross_expander import create_aligned_reaction_video
from face_finder import create_reactor_view
from backchannel_isolator import process_reactor_audio
from compositor import compose_reactor_compilation

import cProfile
import pstats






def handle_reaction_video(song:dict, output_dir: str, react_video, base_video, base_audio_data, base_audio_path, options, extend_by=15):


    react_video_name, react_video_ext = os.path.splitext(react_video)
    output_file = os.path.join(output_dir, os.path.basename(react_video_name) + f"-CROSS-EXPANDER.mp4")

    # if '40' not in react_video_name:
    #     return

    print("processing ", react_video_name)
    # Create the output video file name

    create_aligned_reaction_video(song, react_video_ext, output_file, react_video, base_video, base_audio_data, base_audio_path, options, extend_by=extend_by)

    if not options["isolate_commentary"]:
        return []

    _,sr,aligned_reaction_audio_path = extract_audio(output_file, preserve_silence=True)

    isolated_commentary = process_reactor_audio(output_dir, aligned_reaction_audio_path, base_audio_path, extended_by=extend_by, sr=sr)
    
    if not options["create_reactor_view"]:
        return []

    faces = create_reactor_view(output_file, base_video, replacement_audio=isolated_commentary, show_facial_recognition=False)

    return faces





from moviepy.editor import VideoFileClip

def create_reaction_compilations(song_def:dict, output_dir: str = 'aligned', include_base_video = True, options = {}):


    try:
        song = f"{song_def['artist']} - {song_def['song']}"
        song_directory = os.path.join('Media', song)
        reactions_dir = 'reactions'
        failed_reactions = []

        compilation_path = os.path.join(song_directory, f"{song} (compilation).mp4")
        if os.path.exists(compilation_path):
          print("Compilation already exists", compilation_path)
          return []



        full_output_dir = os.path.join(song_directory, output_dir)
        if not os.path.exists(full_output_dir):
           # Create a new directory because it does not exist
           os.makedirs(full_output_dir)



        print("Processing directory", song_directory, reactions_dir, "Outputting to", output_dir)

        lock_file = os.path.join(full_output_dir, 'locked')
        if os.path.exists( lock_file  ):
            print("...Skipping because another process is already working on this video")
            return []

        lock = open(lock_file, 'w')
        lock.write(f"yo")
        lock.close()


        if options.get('download_and_parse'):
            download_and_parse_reactions(song_def['artist'], song_def['song'], song_def['search'])

        manifest = open(get_manifest_path(song_def['artist'], song_def['song']), "r")


        base_video, react_videos = prepare_reactions(song_directory)


        # Extract the base audio and get the sample rate
        base_audio_data, _, base_audio_path = extract_audio(base_video)


        reaction_dir = os.path.join(song_directory, reactions_dir)

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

            print(f"\tQueuing up {input_file}")

            reaction = {
                'key': input_file,
                'clip': VideoFileClip(input_file),
                'orientation': get_orientation(input_file),
                'group': reactor.get('group', None),
                'featured': featured
            }
            reaction_videos.append(reaction)

        if len(reaction_videos) > 0 and options['create_compilation']:
            compose_reactor_compilation(song_def, base_video, reaction_videos, compilation_path, fast_only=song_def.get('fast_only', False))
    except KeyboardInterrupt as e:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        else:
            print("Could not find lockfile to clean up")
        raise(e)

    except Exception as e:
        traceback.print_exc()
        print(e)
        

    if os.path.exists(lock_file):
        os.remove(lock_file)
    return failed_reactions

def get_orientation(input_file):
    base_name, ext = os.path.splitext(input_file)
    parts = base_name.split('-')
    orientation = parts[-1] if len(parts) > 3 else 'center'
    return orientation


import traceback
if __name__ == '__main__':


    suicide = {
        'include_base_video': True,
        'featured': ['ThatSingerReactions', 'Rosalie Reacts', 'JohnReavesLive', 'Dicodec'],
        'ground_truth': {
            "Black Pegasus": [(241, 306.4), (7*60+5, 7*60+54), (8*60+2, 10*60 + 14)],
            "That’s Not Acting Either": [(75, 129), (192, 282), (7*60+30, 9*60 + 7)]            
        },
        'song': 'Suicide',
        'artist': 'Ren',
        'search': ['Suicide', 'Su!cIde', 'Su!cide', 'suicide']

    }

    fire = {
        'include_base_video': False,
        'featured': ['Johnnie Calloway Sr', 'Anthony Ray Reacts', 'DuaneTV', "SheaWhatNow"],
        'song': 'Fire',
        'artist': 'Ren',
        'search': 'Fire'
    }

    hunger = {
        'include_base_video': True,
        'featured': ['H8TFUL JAY', 'Stevie Knight', 'Jamel_AKA_Jamal', 'Knox Hill', "TheWolfJohnson", "Lilly Jane Reacts", "ThatSingerReactions"],
        'ground_truth': {
            "Knox Hill": [(0.0, 12.6), (80, 89), (123, 131), (156, 160), (173, 176), (189, 193), (235, 239), (247, 254.5), (286, 290), (342, 346), (373, 377), (442, 445), (477, 483), (513, 517), (546, 552), (570, 578), (599, 600), (632, 639), (645, 651), (662, 665), (675, 680), (694, 707), (734, 753)],
            "RAP CATALOG by Anthony Ray": [(0.5, 75), (604, 613), (658, 680), (724, 737), (760, 781), (1236, 1241)],
            "H8TFUL JAY": [(0, 12.75), (28, 36), (49, 51), (88, 90), (104, 112), (135, 140), (160, 178), (195, 200), (227, 238), (254, 260), (284, 298), (319, 330), (355, 361), (371, 408)],

        },
        'song': 'The Hunger',
        'artist': 'Ren',
        'search': 'Hunger'
    }

    genesis = {
        'include_base_video': True,
        'featured': ['Jamel_AKA_Jamal', 'J Rizzle', 'ThatSingerReactions', "K-RayTV", "Black Pegasus"],
        'song': 'Genesis',
        'artist': 'Ren',
        'search': 'Genesis'
    }


    humble = {
        'include_base_video': True,
        'featured': [],
        'song': 'Humble',
        'artist': 'Ren',
        'search': 'Humble',
        'fast_only': True
    }

    ocean = {
        'include_base_video': False,
        'featured': ["RAP CATALOG by Anthony Ray", "Black Pegasus"],
        'song': 'Ocean',
        'artist': 'Ren',
        'search': 'Ocean',
        'fast_only': True
    }

    diazepam = {
        'include_base_video': True,
        'featured': ["Jamel_AKA_Jamal", "Neurogal MD", "SheaWhatNow", "Sean Staxx"],
        'song': 'Diazepam',
        'artist': 'Ren',
        'search': 'Diazepam'
    }


    crutch = {
        'include_base_video': True,
        'featured': ["Rosalie Reacts", "That\u2019s Not Acting Either", "McFly JP", "Joe E Sparks", "Ian Taylor Reacts", "redheadedneighbor"],
        'song': 'Crutch',
        'artist': 'Ren',
        'search': 'Crutch',
        'fast_only': True
    }

    losing_it = {
        'include_base_video': True,
        'featured': [],
        'song': 'Losing It',
        'artist': 'Ren',
        'search': 'Losing It',
        'fast_only': True
    }

    power = {
        'include_base_video': True,
        'featured': [],
        'song': 'Power',
        'artist': 'Ren',
        'search': 'Power',
        'fast_only': True
    }

    sick_boi = {
        'include_base_video': True,
        'featured': [],
        'song': 'Sick Boi',
        'artist': 'Ren',
        'search': 'Sick Boi',
        'fast_only': True
    }


    watch_world_burn = {
        'include_base_video': True,
        'featured': [],
        'song': 'Watch the World Burn',
        'artist': 'Falling in Reverse',
        'search': 'Watch the World Burn',
        'fast_only': True
    }

    time_will_fly = {
        'include_base_video': True,
        'featured': ["ThatSingerReactions", "Black Pegasus", "redheadedneighbor", "The NEW Reel"],
        'song': 'Time Will Fly',
        'artist': 'Sam Tompkins',
        'search': 'Time Will Fly'
    }

    cats_in_the_cradle = {
        'include_base_video': True,
        'featured': [],
        'song': "Cat's in the Cradle",
        'artist': 'Harry Chapin',
        'search': "Cat's in the Cradle",
        'fast_only': True
    }

    handy = {
        'include_base_video': True,
        'featured': ["BrittReacts", "The Matthews Fam", "Jamel_AKA_Jamal", "ScribeCash"],
        'song': 'Handy',
        'artist': 'Weird Al',
        'search': 'Handy'
    }

    foil = {
        'include_base_video': True,
        'featured': ["BrittReacts", "The Matthews Fam", "Jamel_AKA_Jamal", "ScribeCash"],
        'song': 'Foil',
        'artist': 'Weird Al',
        'search': 'Foil',
        'fast_only': True
    }

    pentiums = {
        'include_base_video': True,
        'featured': ["BrittReacts", "The Matthews Fam", "Jamel_AKA_Jamal", "ScribeCash"],
        'song': "It's All About The Pentiums",
        'artist': 'Weird Al',
        'search': 'All About The Pentiums',
        'fast_only': True
    }


    wreck_of_fitzgerald = {
        'include_base_video': True,
        'featured': [],
        'song': 'Wreck of The Edmund Fitzgerald',
        'artist': 'Gordon Lightfoot',
        'search': 'Wreck of The Edmund Fitzgerald',
        'fast_only': True
    }

    this_is_america = {
        'include_base_video': True,
        'featured': [],
        'song': 'This is America',
        'artist': 'Childish Gambino',
        'search': 'This is America',
        'fast_only': True
    }


    finished = [time_will_fly]
    songs = [handy, suicide, hunger, fire, genesis, watch_world_burn, foil, cats_in_the_cradle, power, losing_it, sick_boi, this_is_america, wreck_of_fitzgerald, pentiums, diazepam, ocean, crutch ]


    output_dir = "cheetah"


    options = {
        "output_alignment_metadata": True,
        "output_alignment_video": True,
        "isolate_commentary": True,
        "create_reactor_view": True,
        "create_compilation": True,
        "download_and_parse": True
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







