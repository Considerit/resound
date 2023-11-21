from decouple import config as decouple_config
api_key = decouple_config("Youtube_API_KEY")


import requests
import time
import json 
import os
import subprocess
import copy
import glob 

from utilities import conf, conversion_audio_sample_rate as sr
from utilities.utilities import extract_audio



from pyyoutube import Client, PyYouTubeException
client = Client(api_key=api_key)

RATE_LIMIT_PAUSE = 10  # Pause for 10 seconds if rate limit is hit




def handle_rate_limiting():
    print("Rate limit hit. Pausing for a while...")
    time.sleep(RATE_LIMIT_PAUSE)


def process_reaction(song, artist, search, item, reactors, reactions, test):
    # print(item['snippet'])
    reactor_name = item['snippet']['channelTitle']
    channel_id = item['snippet']['channelId']
    
    if (not test or not test(item['snippet']['title'])):
        # print(f"Bad result: {item['snippet']['title']}", item['snippet']['title'].lower())
        return
    
    if isinstance(item['id'], str):
        video_id = item['id']
    else: 
        if not 'videoId' in item['id']:
            # print(f"{reactor_name} doesn't have a videoid")
            return
        video_id = item['id']['videoId'] 

    
    if video_id in reactions:
        # print(f"duplicate title {item['snippet']['title']}")
        return


    # print(item)
    reactions[video_id] = {
        'song': song,
        'reactor': reactor_name,
        'release_date': item['snippet']['publishedAt'],
        'title': item['snippet']['title'],
        'description': item['snippet']['description'],
        'thumbnails': item['snippet']['thumbnails'],
        'id': video_id,
        'download': False
    }


    print(f"*** ADDED: {reactor_name}  -- {item['snippet']['title']}")
    if reactor_name in reactors:
        print(f"duplicate reactor for {reactor_name}")
    else:
        channel_data = client.channels.list(channel_id=channel_id, return_json=True)
        reactor = {
            'name': reactor_name,
            'icon': channel_data['items'][0]['snippet']['thumbnails']['default']['url']
        }
        reactors[reactor_name] = reactor

def search_reactions(artist, song, search, reactors, reactions, test, page_token=None):
    try:
        if isinstance(search, str):
            song_string = f'"{search}"'
        else:
            song_string = '({})'.format(' OR '.join(['"{}"'.format(s) for s in search]))
        
        # print(song_string)
        params = {
            'q': f'allintitle: {artist} {song_string}',
            'key': api_key,
            'part': 'snippet',
            'maxResults': 50
        }
        if page_token:
            params['pageToken'] = page_token

        search_results = client.search.list(**params, return_json=True)
        # print("results:", search_results)
        for item in search_results['items']:
            process_reaction(song, artist, search, item, reactors, reactions, test)

        # Check for next page and fetch it
        next_page_token = search_results.get('nextPageToken')
        if next_page_token:
            time.sleep(5)  # Pause to avoid hitting rate limits
            search_reactions(artist, song, search, reactors, reactions, test, next_page_token)
    except PyYouTubeException as e:
        if e.status_code == 429:  # Rate Limit Exceeded
            handle_rate_limiting()
            search_reactions(artist, song, search, reactors, reactions, test, page_token)  # Retry after pausing
        else:
            print(f"Error fetching video data: {e}")



def search_for_song(artist, song, search):
    try:
        if isinstance(search, str):
            song_string = f'"{search}"'
        else:
            song_string = '({})'.format(' OR '.join(['"{}"'.format(s) for s in search]))
        
        params = {
            'q': f'allintitle: {search}',
            'key': api_key,
            'part': 'snippet',
            'maxResults': 10
        }
        search_results = client.search.list(**params, return_json=True)
        
        for r in search_results['items']:
            print(r)

        item = search_results['items'][0]
                        
        song = {
            'song': song,
            'artist': item['snippet']['channelTitle'],
            'channel_id': item['snippet']['channelId'],
            'release_date': item['snippet']['publishedAt'],
            'title': item['snippet']['title'],
            'description': item['snippet']['description'],
            'thumbnails': item['snippet']['thumbnails'],
            'id': item['id']['videoId'],
            'download': True
        }

        return song


    except PyYouTubeException as e:
        if e.status_code == 429:  # Rate Limit Exceeded
            handle_rate_limiting()
            search_for_song(artist, song, search)  # Retry after pausing
        else:
            print(f"Error fetching video data: {e}")



# pentatonix
# reactors_to_include=["Roddy Rod", "OfficialDrizzy_Tayy", "Dynasty M&G", "Dian Feb", "BRYBRY", "riesha reacts", "ThatSingerReactions", "THA ANTHONY SHOW", "Dee Omar", "Hassan Ahmed", "JSwithMeeakz", "The dreadheaded oreo", "Zach Archer",  "Jerod M", "Zhen Yao Yin", "DWIDS-TV",  "Rob Reactor",   "Matts Reacts!", "It's me Barry", "The Tide Pool",  "RedTop Reactions", "The Br3ak Room", "G.O.T Games", "BROTHER", "Behind The Curve",   "Wolliofficial", "MAProductions", "Tea Time With Travis", "Reactions by D", "Jacob Restituto",  "Carmen Reacts", "Jimmy Reacts!",  "Meaningfullyvacant"]

# Robyn
# reactors_to_include=["Anton Reacts", "Empress", "HBK Luke", "RedTop Reactions", "Produce At Home", "Jerod M", "PancakeMarshmellowDude"]

reactors_to_include=["The Charismatic Voice", "BARS & BARBELLS", "Flawd TV", "H8TFUL JAY", "Peter Barber", "Chris Liepe", "TrevReacts", 'Neurogal MD', 'Duane Reacts', 'Doug Helvering', 'DuaneTV', 'SheaWhatNow', 'Kso Big Dipper', 'Lee Reacts', 'redheadedneighbor', 'Black Pegasus', 'Knox Hill', 'Jamel_AKA_Jamal', 'ThatSingerReactions', "That\u2019s Not Acting Either", 'Dicodec', 'BrittReacts', "UNCLE MOMO", "RAP CATALOG by Anthony Ray", "Anthony Ray Reacts", "Kyker2Funny", "Ian Taylor Reacts", "Joe E Sparks", "Cliff Beats", "Rosalie Elliott", 'Ellierose Reacts', 'MrLboyd Reacts']
def create_manifest(song_def, artist, song_title, manifest_file, song_search, search, test = None):
    
    print("CREATING MANIFEST")
    global reactors_to_include
    reactors = {}
    reactions = {}

    if os.path.exists(manifest_file):
        jsonfile = open(manifest_file)
        manifest = json.load(jsonfile)
        jsonfile.close()

        migrated_reactions = {}

        for title, reaction in manifest["reactions"].items():
            id = reaction.get('id')
            if id not in migrated_reactions or reaction.get('download', False):
                migrated_reactions[id] = reaction

        manifest["reactions"] = migrated_reactions



    else:
        manifest = {
            "main_song": None,
            "reactions": {},
            "reactors": {}
        }

    if isinstance(search, str):
        search = [search]

    if manifest['main_song'] is None:
        manifest['main_song'] = search_for_song(artist, song_title, song_search)


    if test is None: 
        def test(title):
            if isinstance(search, str):
                song_present = search.lower() in title.lower()
            else:
                song_present = any(s.lower() in title.lower() for s in search)

            artist_present = artist.lower() in title.lower()
            return artist_present and song_present

    reactor_search_terms = copy.copy(search)
    for s in search:
        for r in reactors_to_include:
            if r not in manifest['reactors']:
                reactor_search_terms.append(f"{s} {r}")


    search_reactions(artist, song_title, reactor_search_terms, manifest["reactors"], manifest["reactions"], test)

    if song_def.get('include_videos', False):
        def passes_muster(x):
            return True

        to_include = song_def.get('include_videos')
        for videoid in to_include:
            if videoid not in manifest['reactions']:
                resp = client.videos.list(video_id=videoid)
                if len(resp.items) > 0:
                    item = resp.items[0].to_dict()
                    process_reaction(song_title, artist, search, item, manifest["reactors"], manifest["reactions"], passes_muster)
                else:
                    raise(Exception(f"COULD NOT FIND {videoid}"))

            manifest['reactions'][videoid]['download'] = True

    manifest_json = json.dumps(manifest, indent = 4) 
    jsonfile = open(manifest_file, "w")
    jsonfile.write(manifest_json)
    jsonfile.close()



def get_manifest_path(artist, song):
    song_directory = os.path.join('Media', f"{artist} - {song}")
    
    if not os.path.exists(song_directory):
       # Create a new directory because it does not exist
       os.makedirs(song_directory)

    manifest_file = os.path.join(song_directory, "manifest.json")
    return manifest_file


def download_and_parse_reactions(song_def, artist, song, song_search, search, force=False):

    from backchannel_isolator.track_separation import separate_vocals

    song_directory = os.path.join('Media', f"{artist} - {song}")
    
    if not os.path.exists(song_directory):
       # Create a new directory because it does not exist
       os.makedirs(song_directory)

    manifest_file = get_manifest_path(artist, song)
    if not os.path.exists(manifest_file) or force:
        create_manifest(song_def, artist, song, manifest_file, song_search, search, conf.get('search_tester', None))

    song_data = json.load(open(manifest_file))

    song_file = os.path.join(song_directory, f"{artist} - {song}")
    
    if not os.path.exists(song_file + '.mp4') and not os.path.exists(song_file + '.webm'):
        v_id = song_data["main_song"]["id"]

        cmd = f"yt-dlp -o \"{song_file + '.webm'}\" https://www.youtube.com/watch\?v\={v_id}\;"
        # print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    else: 
        print(f"{song_file} exists")


    full_reactions_path = os.path.join(song_directory, 'reactions')
    if not os.path.exists(full_reactions_path):
       # Create a new directory because it does not exist
       os.makedirs(full_reactions_path)

    reaction_inventory = get_selected_reactions(song_data)

    for channel, reaction in reaction_inventory.items():
        v_id = reaction["id"]

        output = os.path.join(full_reactions_path, channel + '.webm')
        extracted_output = os.path.join(full_reactions_path, channel + '.mp4')

        if not os.path.exists(output) and not os.path.exists(extracted_output):
            cmd = f"yt-dlp -o \"{output}\" https://www.youtube.com/watch\?v\={v_id}\;"
            # print(cmd)

            try:
                subprocess.run(cmd, shell=True, check=True)
            except:
                print(f"Failed to download {output} {cmd}")


        # Get all reaction video files
        mkv_videos = glob.glob(os.path.join(full_reactions_path, "*.mkv"))

        # Convert all mkv videos to mp4 in the same directory and then delete the mkv videos

        for mkv_video in mkv_videos:
            print(f"HANDLING MKV {mkv_video}")
            # Strip existing video extension from filename
            base_name = os.path.basename(mkv_video)
            file_name_without_ext = os.path.splitext(base_name)[0]
            # If the stripped filename still has an extension, remove that too
            if any(ext in file_name_without_ext for ext in ['.webm', '.mp4', '.mkv']):
                file_name_without_ext = os.path.splitext(file_name_without_ext)[0]

            output_file = os.path.join(reaction_dir, file_name_without_ext + '.mp4')

            print(f'OUTPUT FILE {output_file}')
            ffmpeg.input(mkv_video).output(output_file).run()
            os.remove(mkv_video)






        if os.path.exists(extracted_output):
            reaction_file = extracted_output
        else: 
            reaction_file = output


        vocal_path_filename = 'vocals-post-high-passed.wav'
        separation_path = os.path.splitext(reaction_file)[0]
        vocals_path = os.path.join(separation_path, vocal_path_filename)
        if not os.path.exists( vocals_path ):
            reaction_audio_data, __, reaction_audio_path = extract_audio(reaction_file)
            separate_vocals(separation_path, reaction_audio_path, vocal_path_filename, duration=len(reaction_audio_data)/float(sr))




def get_selected_reactions(song_data, filter_by_downloaded=True):
    reaction_inventory = {}
    for _, reaction in song_data["reactions"].items():
        key = reaction['reactor']

        if reaction.get("download") or (not filter_by_downloaded and key not in reaction_inventory):    
            if key in reaction_inventory:
                key = reaction['reactor'] + '_' + reaction["id"] # handle multiple reactions for a single channel
            reaction_inventory[key] = reaction
    return reaction_inventory


def generate_description_text(artist, song):
    manifest_file = get_manifest_path(artist, song)
    song_data = json.load(open(manifest_file))

    reactions = get_selected_reactions(song_data)

    for channel, reaction in reactions.items():
        print(f"\t{channel}: https://youtube.com/watch?v={reaction.get('id')}")


if __name__ == '__main__':

    generate_description_text("Ren", "Fire")



    manifest_file = get_manifest_path("Ren", "Money Game Part 1")
    song_data = json.load(open(manifest_file))

    downloaded_m1 = get_selected_reactions(song_data, True)
    available_m1 = get_selected_reactions(song_data, False)


    manifest_file = get_manifest_path("Ren", "Money Game Part 2")
    song_data = json.load(open(manifest_file))

    downloaded_m2 = get_selected_reactions(song_data, True)
    available_m2 = get_selected_reactions(song_data, False)


    manifest_file = get_manifest_path("Ren", "Money Game Part 3")
    song_data = json.load(open(manifest_file))

    downloaded_m3 = get_selected_reactions(song_data, True)
    available_m3 = get_selected_reactions(song_data, False)


    reactors_used = {}
    for inv in [downloaded_m1, downloaded_m2, downloaded_m3]:
        for k,r in inv.items():
            reactors_used[k] = True

    for i, inv in enumerate([(downloaded_m1,available_m1), (downloaded_m2,available_m2), (downloaded_m3,available_m3)]):
        (used, avail) = inv
        print(f"Money Game Part {i+1} -- including {len(used.keys())} reactors of {len(avail.keys())} available")
        for k in reactors_used.keys():
            if k not in used and k in avail:
                print(f"\tConsider including {k} / https://www.youtube.com/watch?v={avail[k]['id']}")


    


