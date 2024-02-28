import requests
import time
import json 
import os
import subprocess
import copy
import glob 
from datetime import datetime, timedelta
import webbrowser



from decouple import config as decouple_config


api_key = decouple_config("Youtube_API_KEY")

from pyyoutube import Client, PyYouTubeException


youtube_oauth_client_id = decouple_config("youtube_oauth_client_id")
youtube_client_secret = decouple_config("youtube_client_secret")


try: 
    access_token = decouple_config("youtube_access_token")
    last_updated_str = decouple_config("youtube_access_token_last_updated")
    last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
except:
    last_updated = None


if last_updated is None or datetime.now() - last_updated > timedelta(days=1):
    # if the access token expires, you need to delete the access token line in the .env file,
    #   and make sure redirect_url is None, to reset the process.
    redirect_url = "https://localhost/?state=Python-YouTube&code=4/0AeaYSHB5qk5e3b3865h4Ru866Gp1xNqreapoaMxvFYm2o_vJSNzhDNzDFcNXMLnFgKW7bg&scope=profile%20https://www.googleapis.com/auth/youtube%20https://www.googleapis.com/auth/userinfo.profile"

    client = Client(client_id=youtube_oauth_client_id, client_secret=youtube_client_secret)


    if redirect_url is not None:

        existing_values = {}
        with open('.env', 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):  # Ignore empty lines and comments
                    key, value = line.split('=', 1)  # Split on the first equals sign
                    existing_values[key] = value

        try: 
            access_token = client.generate_access_token(authorization_response=redirect_url).access_token

            existing_values["youtube_access_token"] = access_token
            existing_values["youtube_access_token_last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Write everything back to the .env file
            with open(".env", "w") as file:
                for key, value in existing_values.items():
                    file.write(f"{key}={value}\n")

            print("Your .env file should now be updated.")
        except:
            redirect_url = None


    if redirect_url is None:
        print("We need to generate a valid access token")

        # Get authorization url
        x=client.get_authorize_url()
        print(x[0])
        # Click url and give permissions.
        # Copy the redirected url. Note that that this will look like an error page, possibly blank.

        webbrowser.open(x[0])
        raise(Exception(f"Need to regenerate your access token. Access this url in a browser, then set redirect_url above to the resulting url, and rerun: \n{x[0]}"))


client = Client(access_token=access_token)





# client = Client(api_key=api_key)

RATE_LIMIT_PAUSE = 10  # Pause for 10 seconds if rate limit is hit

channel_id = decouple_config("youtube_channel")  # resound

def get_or_create_playlist(title, description):
    playlist_id = find_playlist_by_title(title)

    if playlist_id is not None:
        return playlist_id

    try:
        new_playlist = client.playlists.insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": description
                },
                "status": {
                    "privacyStatus": "public"  # Can be public, private, or unlisted
                }
            }
        )
        return new_playlist.id  # Return new playlist ID
    except PyYouTubeException as e:
        print(f"An error occurred: {e}")

    return None

def find_playlist_by_title(playlist_title):
    try:
        response = client.playlists.list(channel_id=channel_id, count=None)
        for playlist in response.items:
            print(playlist.snippet.title.lower(), playlist.id)
            if playlist.snippet.title.lower() == playlist_title.lower():
                return playlist.id
        return None
    except PyYouTubeException as e:
        print(f"An error occurred: {e}")
        return None




def get_existing_video_ids(playlist_id):


    try:
        existing_videos = []
        got_first = False
        next_page_token = None
        while not got_first or next_page_token:
            if next_page_token:
                result = client.playlistItems.list(playlist_id=playlist_id, part="contentDetails", pageToken=next_page_token, max_results=50)                
            else:
                result = client.playlistItems.list(playlist_id=playlist_id, part="contentDetails", max_results=50)

            for item in result.items:
                existing_videos.append(item.contentDetails.videoId)

            next_page_token = result.nextPageToken
            print(f"Got {len(existing_videos)} items, next page is {next_page_token}")
            got_first = True

        return existing_videos
    except PyYouTubeException as e:
        print(f"An error occurred: {e}")
        return []



existing_ids = []



def add_videos_to_playlist(playlist_id, video_ids):
    global existing_ids
    existing_ids += get_existing_video_ids(playlist_id)
    print(existing_ids)
    for video_id in video_ids:
        if video_id not in existing_ids:
            try:
                body = {
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id
                        }
                    }
                }

                client.playlistItems.insert(
                    part="snippet",
                    body=body
                )
                existing_ids.append(video_id)
                print(f"Added video {video_id} to playlist.")

            except PyYouTubeException as e:
                print(f"Error adding video {video_id}: {e}")




if __name__=='__main__':
    from inventory.inventory import get_manifest_path


    playlist_title = "Reactions to Ren's Fred Again Mash Up"
    playlist_description = "Reactions to Ren's Fred Again Mash Up that were included in this channel's respective Reaction Concert."

    playlist_id = get_or_create_playlist(playlist_title, playlist_description)
    print(playlist_id)

    # print(get_existing_video_ids(playlist_id))

    # if playlist_id is not None:

    song1 = 'Ren - Fred Again Mash Up'
    # song2 = 'Ren - Ocean'

    playlist_songs = [
        song1,
        # song2
    ]
    songs = []
    for song in playlist_songs:
        parts = song.split(' - ')
        songs.append( get_manifest_path(parts[0], parts[1])  )

    for manifest_file in songs:

        song_data = json.load(open(manifest_file))
        reaction_data = song_data['reactions']


        song_ids = [ s.get('id') for s in reaction_data.values() if s.get('download') ]

        song_ids.sort(key=lambda x: x[0])
        # for channel, id in song_ids:
        #     print("https://www.youtube.com/watch?v=" + id, channel)

        add_videos_to_playlist(playlist_id, song_ids)


    def fill_in_for_chiefaberach(artist, song, found_playlist):
        missing_title = f"Missing {artist} - {song} Reactions for Chiefaberach"
        found_videos = get_existing_video_ids(found_playlist)
        song_data = json.load(open(get_manifest_path(artist, song)))
        reactions = song_data['reactions']

        missing = []
        print('EXISTING!')

        for vid in found_videos:
            print(vid)

        for vid, reaction in reactions.items():
            if vid not in found_videos:
                missing.append(vid)

        print('MISSING!')

        playlist_description = "Chief, these are reactions that I think are missing from your playlist."
        playlist_id = get_or_create_playlist(missing_title, playlist_description)

        for vid in missing:
            print(f"[{vid}] {reactions[vid]['reactor']} - {reactions[vid]['title']}")
        add_videos_to_playlist(playlist_id, missing)


    def hi_ren_chiefaberach_gap():
        found_playlist = "PLvQ9PT7Tdzm0oWcg8u_v77mnqNIjZ3lMo"
        artist = 'Ren'
        song = 'Hi Ren - BP'        
        fill_in_for_chiefaberach(artist, song, found_playlist)

    found_playlist = "PLvQ9PT7Tdzm3NTWF9CzxbXvihZ-wS1uyx"
    artist = 'Ren'
    song = 'Fred Again Mash Up'        
    fill_in_for_chiefaberach(artist, song, found_playlist)


