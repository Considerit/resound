import requests
import time
import json 
import os
import subprocess
import copy
import glob 



from decouple import config as decouple_config


api_key = decouple_config("Youtube_API_KEY")

from pyyoutube import Client, PyYouTubeException


youtube_oauth_client_id = decouple_config("youtube_oauth_client_id")
youtube_client_secret = decouple_config("youtube_client_secret")


try: 
    access_token = decouple_config("youtube_access_token")

except:
    # if the access token expires, you need to delete the access token line in the .env file,
    #   and make sure redirect_url is None, to reset the process.
    redirect_url = None


    client = Client(client_id=youtube_oauth_client_id, client_secret=youtube_client_secret)

    if redirect_url is None:
        print("We need to generate a valid access token")

        # Get authorization url
        x=client.get_authorize_url()
        # ('https://accounts.google.com/o/oauth2/v2/auth?response_type=code&client_id=id&redirect_uri=https%3A%2F%2Flocalhost%2F&scope=scope&state=PyYouTube&access_type=offline&prompt=select_account', 'PyYouTube')
        print(x[0])
        # Click url and give permissions.
        # Copy the redirected url. Note that that this will look like an error page, possibly blank.
        raise(Exception(f"Need to regenerate your access token. Access this url in a browser, then set redirect_url above to the resulting url, and rerun: \n{x[0]}"))

    else: 
        access_token = client.generate_access_token(authorization_response=redirect_url).access_token
        subprocess.run(
            f"echo \"\nyoutube_access_token={access_token}\" >> ./.env", shell=True
        )
        # AccessToken(access_token='token', expires_in=3599, token_type='Bearer')
        print("Your .env file should now be updated.")


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
    # existing_ids += get_existing_video_ids(playlist_id)
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


    # playlist_title = "Reactions to Sam Tompkins' TIME WILL FLY, HERO, and HERO (live)"
    # playlist_description = "Reactions to Sam Tompkins' TIME WILL FLY, HERO, and HERO (live) that were included in this channel's respective Reaction Symphony."

    # playlist_id = get_or_create_playlist(playlist_title, playlist_description)
    # print(playlist_id)

    # print(get_existing_video_ids(playlist_id))

    # if playlist_id is not None:

    # time_will_fly = 'Sam Tompkins - Time Will Fly'
    # hero = 'Sam Tompkins - Hero'
    # hero_live = 'Sam Tompkins - Hero live'

    # playlist_songs = [
    #     time_will_fly,
    #     hero,
    #     hero_live
    # ]
    # songs = []
    # for song in playlist_songs:
    #     parts = song.split(' - ')
    #     songs.append( get_manifest_path(parts[0], parts[1])  )

    # for manifest_file in songs:

    #     song_data = json.load(open(manifest_file))
    #     reaction_data = song_data['reactions']


    #     song_ids = [ s.get('id') for s in reaction_data.values() if s.get('download') ]

    #     song_ids.sort(key=lambda x: x[0])
    #     # for channel, id in song_ids:
    #     #     print("https://www.youtube.com/watch?v=" + id, channel)

    #     add_videos_to_playlist(playlist_id, song_ids)



    def hi_ren_chiefaberach_gap():
        hi_ren_chiefaberach = "PLvQ9PT7Tdzm0oWcg8u_v77mnqNIjZ3lMo"
        aberach_vids = get_existing_video_ids(hi_ren_chiefaberach)
        song_data = json.load(open(get_manifest_path('Ren', 'Hi Ren - BP')))
        reactions = song_data['reactions']

        missing = []
        print('EXISTING!')

        for vid in aberach_vids:
            print(vid)

        for vid, reaction in reactions.items():
            if vid not in aberach_vids:
                missing.append(vid)

        print('MISSING!')

        playlist_title = "Missing Hi Ren Reactions for Chiefaberach"
        playlist_description = "Chief, these are reactions that I think are missing from your playlist."
        playlist_id = get_or_create_playlist(playlist_title, playlist_description)

        for vid in missing:
            print(f"[{vid}] {reactions[vid]['reactor']} - {reactions[vid]['title']}")
        add_videos_to_playlist(playlist_id, missing)

    hi_ren_chiefaberach_gap()