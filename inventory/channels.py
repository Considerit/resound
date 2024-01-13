from decouple import config as decouple_config
api_key = decouple_config("Youtube_API_KEY")

import json 
import os
import random
import time

from inventory.youtubesearch import YoutubeSearch

from pyyoutube import Client, PyYouTubeException
client = Client(api_key=api_key)


def get_reactors_inventory():
    reactors_inventory_file = os.path.join('Media', f"reactor_inventory.json")
    if os.path.exists(reactors_inventory_file):
        reactors_inventory = json.load(open(reactors_inventory_file))
    else:
        reactors_inventory = {}

    return reactors_inventory

def write_reactors_inventory(reactors_inventory):
    reactors_inventory_file = os.path.join('Media', f"reactor_inventory.json")
    reactors_inventory_json = json.dumps(reactors_inventory, indent = 4) 
    jsonfile = open(reactors_inventory_file, "w")
    jsonfile.write(reactors_inventory_json)
    jsonfile.close()

def update_channel(channel):
    ri = get_reactors_inventory()

    ri[channel.get('channelId')] = channel

    write_reactors_inventory(ri)

def get_channel(channelId):
    ri = get_reactors_inventory()

    return ri[channelId]



def refresh_reactors_inventory():

    reactors_inventory = get_reactors_inventory()


    list_subfolders_with_paths = [os.path.join(f.path, 'manifest.json') for f in os.scandir('Media') if f.is_dir()]
    list_subfolders_with_paths = [f for f in list_subfolders_with_paths if os.path.exists(f)]


    # update song inclusions
    for song_manifest in list_subfolders_with_paths:

        song = os.path.basename(song_manifest[0:len(song_manifest) - len("/manifest.json")])
        # artist, song = song.split(' - ')
        # filter_and_augment_manifest(artist, song)

        reactions = json.load(open(song_manifest))['reactions']

        included_in = {}
        mentioned_in = {}
        for reaction in reactions.values():
            channel_id = reaction['channelId']

            mentioned_in[channel_id] = True
            if reaction.get('download', False):
                included_in[channel_id] = True

            if channel_id not in reactors_inventory:
                reactors_inventory[channel_id] = {}

        for channel_id, channel_info in reactors_inventory.items():
            if 'included_in' not in channel_info:
                channel_info['included_in'] = []

            if channel_id in included_in and song not in channel_info['included_in']:
                channel_info['included_in'].append(song)
            elif channel_id not in included_in and song in channel_info['included_in']:
                channel_info['included_in'].remove(song)

            if 'mentioned_in' not in channel_info:
                channel_info['mentioned_in'] = []

            if channel_id in mentioned_in and song not in channel_info['mentioned_in']:
                channel_info['mentioned_in'].append(song)
            elif channel_id not in mentioned_in and song in channel_info['mentioned_in']:
                channel_info['mentioned_in'].remove(song)



    # update channel info with statistics
    for channel_id, channel_info in reactors_inventory.items():
        if 'subscriberCount' in channel_info:
            continue
        info = getChannelInfo(channel_id)
        channel_info.update(info)
        print(f"{channel_info['title']}: {info['subscriberCount']}")

        write_reactors_inventory(reactors_inventory)

    for channel_id, channel_info in reactors_inventory.items():
        channel_info['channelId'] = channel_id
        if 'aliases' in channel_info:
            channel_info['aliases'] = [alias for alias in channel_info['aliases'] if alias != channel_info['title']]
            if len(channel_info['aliases']) == 0:
                del channel_info['aliases']

    write_reactors_inventory(reactors_inventory)


    add_reactor_notations()


def get_recommended_channels(include_eligible=True):
    reactors_inventory = get_reactors_inventory()
    recommended = [r for r in reactors_inventory.values() \
                      if r.get('auto', None) == 'include' or \
                         (include_eligible and r.get('auto', None) == 'eligible')]

    recommended.sort( key=lambda x: x.get('title') )

    return recommended


def getChannelInfo(channel_id):

    params = {
        'channel_id': channel_id,
        'key': api_key,
        'part': 'snippet,statistics',
        'maxResults': 1
    }

    item = client.channels.list(**params, return_json=True)['items'][0]

    result = {
        "title":       item['snippet']['title'],    
        "description": item['snippet']['description'],
        'publishedAt': item['snippet']['publishedAt'],
        "customUrl":   item['snippet']['customUrl'],
        "viewCount":   int(item['statistics']['viewCount']),
        "subscriberCount": int(item['statistics']['subscriberCount']),
        "videoCount":  int(item['statistics']['videoCount']),
        "included_in": []
    }
    return result



from inventory.reactor_notations import annotations

def add_reactor_notations():
    reactors_inventory = get_reactors_inventory()

    later = False
    for __, channel_info in reactors_inventory.items():


        if 'notes' not in channel_info:
            channel_info['notes'] = {}


        for note, search, test in annotations:
            if note in channel_info['notes']:
                continue

            print(f"Adding note for {channel_info.get('title')} ({note})")


            items = YoutubeSearch(f'{note}', max_results=3, channel=channel_info).to_dict()
            

            if items is None:
                print("\tFAILED")
                continue

            if len(items) > 0:

                channel_info['notes'][note] = []

                for item in items: 
                    title = item['title']

                    if search.lower() in title.lower() and test(title):
                        channel_info['notes'][note].append(title)

                if len(channel_info['notes'][note]) == 0:
                    channel_info['notes'][note] = None
                else: 
                    print(f"\tAdded {channel_info['notes'][note]}")


            else:
                channel_info['notes'][note] = None



            write_reactors_inventory(reactors_inventory)

            





if __name__ == '__main__':

    refresh_reactors_inventory()

    


