from decouple import config
api_key = config("Youtube_API_KEY")


import requests
import time
import json 
import os
import subprocess
from pyyoutube import Client, PyYouTubeException


from utilities import prepare_reactions


client = Client(api_key=api_key)
RATE_LIMIT_PAUSE = 10  # Pause for 10 seconds if rate limit is hit

def handle_rate_limiting():
    print("Rate limit hit. Pausing for a while...")
    time.sleep(RATE_LIMIT_PAUSE)


def process_reaction(song, artist, search, item, reactors, reactions):

    reactor_name = item['snippet']['channelTitle']
    channel_id = item['snippet']['channelId']
    

    if isinstance(song, str):
        song_present = search.lower() in item['snippet']['title'].lower()
    else:
        song_present = any(s.lower() in item['snippet']['title'].lower() for s in search)

    artist_present = artist.lower() in item['snippet']['title'].lower()
    
    if not song_present or not artist_present:
        print(f"Bad result: {item['snippet']['title']}", item['snippet']['title'].lower())
        return
    
    if item['snippet']['title'] in reactions:
        print(f"duplicate title {item['snippet']['title']}")
        return
    
    if not 'videoId' in item['id']:
        print(f"{reactor_name} doesn't have a videoid")
        return

    reactions[item['snippet']['title']] = {
        'song': song,
        'reactor': reactor_name,
        'release_date': item['snippet']['publishedAt'],
        'title': item['snippet']['title'],
        'description': item['snippet']['description'],
        'thumbnails': item['snippet']['thumbnails'],
        'id': item['id']['videoId'],
        'download': False
    }

    if reactor_name in reactors:
        print(f"duplicate reactor for {reactor_name}")
    else:
        channel_data = client.channels.list(channel_id=channel_id, return_json=True)
        reactor = {
            'name': reactor_name,
            'icon': channel_data['items'][0]['snippet']['thumbnails']['default']['url']
        }
        reactors[reactor_name] = reactor

def search_reactions(artist, song, search, reactors, reactions, page_token=None):
    try:
        if isinstance(search, str):
            song_string = f'"{search}"'
        else:
            song_string = '({})'.format(' OR '.join(['"{}"'.format(s) for s in search]))
        
        params = {
            'q': f'allintitle: {artist} {song_string}',
            'key': api_key,
            'part': 'snippet',
            'maxResults': 50
        }
        if page_token:
            params['pageToken'] = page_token

        search_results = client.search.list(**params, return_json=True)
        print("results:", search_results)
        for item in search_results['items']:
            process_reaction(song, artist, search, item, reactors, reactions)

        # Check for next page and fetch it
        next_page_token = search_results.get('nextPageToken')
        if next_page_token:
            time.sleep(5)  # Pause to avoid hitting rate limits
            search_reactions(artist, song, search, reactors, reactions, next_page_token)
    except PyYouTubeException as e:
        if e.status_code == 429:  # Rate Limit Exceeded
            handle_rate_limiting()
            search_reactions(artist, song, search, reactors, reactions, page_token)  # Retry after pausing
        else:
            print(f"Error fetching video data: {e}")



def search_for_song(artist, song, search):
    try:
        if isinstance(search, str):
            song_string = f'"{search}"'
        else:
            song_string = '({})'.format(' OR '.join(['"{}"'.format(s) for s in search]))
        
        params = {
            'q': f'allintitle: {artist} {song_string}',
            'key': api_key,
            'part': 'snippet',
            'maxResults': 1
        }
        search_results = client.search.list(**params, return_json=True)
        
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





def create_manifest(artist, song_title, manifest_file, search):
    reactors = {}
    reactions = {}

    song_data = search_for_song(artist, song_title, search)
    search_reactions(artist, song_title, search, reactors, reactions)

    manifest = {
        "main_song": song_data,
        "reactions": reactions, 
        "reactors": reactors
    }

    manifest_json = json.dumps(manifest, indent = 4) 
    jsonfile = open(manifest_file, "w")
    jsonfile.write(manifest_json)
    jsonfile.close()





def download_and_parse_reactions(artist, song, search):
    song_directory = os.path.join('Media', f"{artist} - {song}")
    
    if not os.path.exists(song_directory):
       # Create a new directory because it does not exist
       os.makedirs(song_directory)

    manifest_file = os.path.join(song_directory, "manifest.json")
    if not os.path.exists(manifest_file):
        create_manifest(artist, song, manifest_file, search)

    song_data = json.load(open(manifest_file))

    song_file = os.path.join(song_directory, f"{artist} - {song}")
    
    if not os.path.exists(song_file + '.mp4') and not os.path.exists(song_file + '.webm'):
        v_id = song_data["main_song"]["id"]

        cmd = f"yt-dlp -o \"{song_file + '.webm'}\" https://www.youtube.com/watch\?v\={v_id}\;"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    else: 
        print(f"{song_file} exists")


    full_reactions_path = os.path.join(song_directory, 'reactions')
    if not os.path.exists(full_reactions_path):
       # Create a new directory because it does not exist
       os.makedirs(full_reactions_path)

    reaction_inventory = {}
    for _, reaction in song_data["reactions"].items():
        if reaction.get("download"):    
            key = reaction['reactor']
            if key in reaction_inventory:
                key = reaction['reactor'] + '_' + reaction["id"] # handle multiple reactions for a single channel
            reaction_inventory[key] = reaction

    for name, reaction in reaction_inventory.items():
        v_id = reaction["id"]
        output = os.path.join(full_reactions_path, name + '.webm')
        extracted_output = os.path.splitext(output)[0] + '.mp4'

        if not os.path.exists(output) and not os.path.exists(extracted_output) and not os.path.exists(os.path.join(full_reactions_path, 'tofix', reaction['reactor'] + '.mp4')):
            cmd = f"yt-dlp -o \"{output}\" https://www.youtube.com/watch\?v\={v_id}\;"
            print(cmd)

            try:
                subprocess.run(cmd, shell=True, check=True)
            except:
                print(f"Failed to download {output} {cmd}")

    prepare_reactions(song_directory)




