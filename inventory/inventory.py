from decouple import config as decouple_config

api_key = decouple_config("Youtube_API_KEY")


import requests
import time
import json
import os
import subprocess
import copy
import glob
import ffmpeg


from utilities import conversion_audio_sample_rate as sr
from utilities.utilities import check_and_fix_fps

from inventory.channels import (
    get_recommended_channels,
    refresh_reactors_inventory,
    update_channel,
    get_channel,
)

import yt_dlp

from inventory.youtubesearch import YoutubeSearch

from pyyoutube import Client, PyYouTubeException

from library import generate_youtube_search_string, search_defs

client = Client(api_key=api_key)

RATE_LIMIT_PAUSE = 10  # Pause for 10 seconds if rate limit is hit


def handle_rate_limiting():
    print("Rate limit hit. Pausing for a while...")
    time.sleep(RATE_LIMIT_PAUSE)


def process_reaction(song, artist, search, item, reactions, test):
    # print(item['snippet'])
    reactor_name = item["snippet"]["channelTitle"]
    channel_id = item["snippet"]["channelId"]

    if isinstance(item["id"], str):
        video_id = item["id"]
    else:
        if not "videoId" in item["id"]:
            recommended_channels = (
                channel.get("channelId") for channel in get_recommended_channels()
            )
            if channel_id not in recommended_channels:
                print(f"{reactor_name} doesn't have a videoid AND is not a recommended channel")
            return False
        video_id = item["id"]["videoId"]

    if not test or not test(item["snippet"]["title"]):
        print(
            f"Bad result: [{video_id}] {item['snippet']['title']}",
            item["snippet"]["title"].lower(),
        )
        return False

    if video_id in reactions:
        # print(f"duplicate title {item['snippet']['title']}")
        return True

    # print(item)
    reactions[video_id] = {
        "song": song,
        "reactor": reactor_name,
        "release_date": item["snippet"]["publishedAt"],
        "title": item["snippet"]["title"],
        "description": item["snippet"]["description"],
        "thumbnails": item["snippet"]["thumbnails"],
        "id": video_id,
        "download": False,
        "channelId": channel_id,
    }

    print(f"\n*** ADDED: {reactor_name}  -- {item['snippet']['title']}\n")

    return True


def search_reactions(
    artist,
    song,
    query,
    search,
    reactions,
    test,
    search_channel_id=None,
    page_token=None,
    sort="relevance",
):
    try:
        if isinstance(search, str):
            song_string = f'"{search}"'
        else:
            song_string = "({})".format(" OR ".join(['"{}"'.format(s) for s in search]))

        # print(song_string)
        params = {
            # 'q': f'allintitle: {artist} {song_string}',
            "q": query,  # f"{artist} {song_string} reacts|reaction",
            "key": api_key,
            "part": "snippet",
            "maxResults": 50,
            "order": sort,
        }
        if search_channel_id:
            params["channelId"] = search_channel_id
        if page_token:
            params["pageToken"] = page_token

        search_results = client.search.list(**params, return_json=True)
        next_page_token = search_results.get("nextPageToken", None)
        print(
            f"\n{params['q']} found {len(search_results['items'])} results, next token = {next_page_token}\n"
        )
        # print("results:", search_results)
        found_at_least_one_relevant = True  # False
        for item in search_results["items"]:
            found_at_least_one_relevant = (
                process_reaction(song, artist, search, item, reactions, test)
                or found_at_least_one_relevant
            )

        # Check for next page and fetch it
        if next_page_token and found_at_least_one_relevant:
            time.sleep(5)  # Pause to avoid hitting rate limits
            search_reactions(
                artist,
                song,
                query,
                search,
                reactions,
                test,
                search_channel_id,
                next_page_token,
                sort,
            )
    except PyYouTubeException as e:
        if e.status_code == 429:  # Rate Limit Exceeded
            handle_rate_limiting()
            search_reactions(
                artist,
                song,
                query,
                search,
                reactions,
                test,
                search_channel_id,
                page_token,
                sort,
            )  # Retry after pausing
        else:
            print(f"Error fetching video data: {e}")


def search_recommended_channels(artist, song, search, reactions, test, manifest, force=False):
    channels = get_recommended_channels()  # include_all=True)

    song_string = f'"{artist} {song} {search[0]}"'

    print("\tSearching recommended channels")
    for idx, channel in enumerate(channels):
        if not force and channel.get("searched_for", {}).get(song_string, False):
            continue

        print(f"\t\t[{100 * idx / len(channels):.1f}%] Checking {channel.get('title')}")

        items = YoutubeSearch(song_string, max_results=3, channel=channel).to_dict()

        if "searched_for" not in channel:
            channel["searched_for"] = {}
        channel["searched_for"][song_string] = True
        update_channel(channel)

        if not items or len(items) == 0:
            # print(f"\tNo results for {channel.get('title')}")
            continue

        for item in items:
            if not test or not test(item["title"], is_channel_search=True):
                print(f"\t\t\tBad result: {item['title']}")
                continue

            if item["id"] in reactions:
                # print(f"\tduplicate title {item['title']}")
                continue

            print(f"\t\t\t*** New reaction found: {item['title']}")
            views = item["views"]
            if isinstance(views, str):
                views = int(item["views"].replace(",", "").replace(" views", ""))

            reaction = {
                "song": song,
                "reactor": item["channel"],
                "channelId": item["channelId"],
                "title": item["title"],
                "description": item["long_desc"],
                "thumbnails": item["thumbnails"],
                "id": item["id"],
                "duration": item["duration"],
                "views": views,
                "download": False,
            }

            reactions[item["id"]] = reaction
            save_reactions_manifest(manifest, artist, song)


def search_for_song(artist, song, search):
    try:
        if isinstance(search, str):
            song_string = search
        else:
            song_string = "({})".format(" OR ".join(['"{}"'.format(s) for s in search]))

        item = YoutubeSearch(search, max_results=1).to_dict()[0]
        song = {
            "song": song,
            "artist": item["channel"],
            "channelId": item["channelId"],
            "title": item["title"],
            "description": item["long_desc"],
            "thumbnails": item["thumbnails"],
            "id": item["id"],
            "download": True,
        }

        return song

    except PyYouTubeException as e:
        if e.status_code == 429:  # Rate Limit Exceeded
            handle_rate_limiting()
            search_for_song(artist, song, search)  # Retry after pausing
        else:
            print(f"Error fetching video data: {e}")


def get_reactions_manifest(artist, song):
    manifest_file = get_manifest_path(artist, song)

    if os.path.exists(manifest_file):
        jsonfile = open(manifest_file)
        manifest = json.load(jsonfile)
        jsonfile.close()

    else:
        manifest = {
            "main_song": None,
            "reactions": {},
        }

    return manifest


def save_reactions_manifest(manifest, artist, song):
    manifest_file = get_manifest_path(artist, song)

    manifest_json = json.dumps(manifest, indent=4)
    jsonfile = open(manifest_file, "w")
    jsonfile.write(manifest_json)
    jsonfile.close()


def create_manifest(song_def, artist, song_title, song_search, search, test=None):
    print("CREATING MANIFEST")
    reactions = {}

    manifest = get_reactions_manifest(artist, song_title)

    if isinstance(search, str):
        search = [search]

    if manifest["main_song"] is None:
        manifest["main_song"] = search_for_song(artist, song_title, song_search)

    song_key = f"{song_def['artist']} - {song_def['song']}"
    query = generate_youtube_search_string(search_defs.get(song_key))
    print(query)

    for sort in ["relevance", "date"]:  # , "rating", "viewCount", "title"]:
        search_reactions(
            artist,
            song_title,
            query,
            search,
            manifest["reactions"],
            test,
            song_def.get("search_channel_id", None),
            sort=sort,
        )

    save_reactions_manifest(manifest, artist, song_title)

    if not song_def.get("search_channel_id", None):
        search_recommended_channels(
            artist,
            song_title,
            search,
            manifest["reactions"],
            test,
            manifest,
            force=False,
        )

    save_reactions_manifest(manifest, artist, song_title)

    if (
        song_def.get("include_videos", False) or song_def.get("playlist_id", False)
    ) and not song_def.get("skip_searching_recommended"):

        def passes_muster(x):
            return True

        to_include = song_def.get("include_videos", [])
        if song_def.get("playlist_id", False):
            from inventory.playlists import get_existing_video_ids

            to_include += get_existing_video_ids(song_def.get("playlist_id"))

        print("TRYING TO GET INCLUDED VIDEOS", to_include)
        for videoid in to_include:
            if videoid not in manifest["reactions"]:
                resp = client.videos.list(video_id=videoid)
                if len(resp.items) > 0:
                    item = resp.items[0].to_dict()
                    print(f"\t{videoid} FOUND")
                    process_reaction(
                        song_title,
                        artist,
                        search,
                        item,
                        manifest["reactions"],
                        passes_muster,
                    )
                    assert videoid in manifest["reactions"]
                else:
                    print("COULD NOT FIND {videoid}")
            else:
                print(f"\t{videoid} Already in manifest")

            # manifest["reactions"][videoid]["download"] = True

    save_reactions_manifest(manifest, artist, song_title)

    refresh_reactors_inventory()

    filter_and_augment_manifest(artist, song_title, force=True)


def get_manifest_path(artist, song):
    song_directory = os.path.join("Media", f"{artist} - {song}")

    if not os.path.exists(song_directory):
        # Create a new directory because it does not exist
        os.makedirs(song_directory)

    manifest_file = os.path.join(song_directory, "manifest.json")
    return manifest_file


def download_and_parse_reactions(
    song_def, artist, song, song_search, search, refresh_manifest=False
):
    from utilities import conf

    song_directory = os.path.join("Media", f"{artist} - {song}")

    if not os.path.exists(song_directory):
        # Create a new directory because it does not exist
        os.makedirs(song_directory)

    manifest_file = get_manifest_path(artist, song)
    if refresh_manifest or not os.path.exists(manifest_file):
        create_manifest(
            song_def, artist, song, song_search, search, conf.get("search_tester", None)
        )

    song_data = json.load(open(manifest_file))

    download_song(song_directory, artist, song)
    download_included_reactions(song_directory, artist, song)


def download_song(song_directory, artist, song):
    song_file = os.path.join(song_directory, f"{artist} - {song}")

    if not os.path.exists(song_file + ".mp4") and not os.path.exists(song_file + ".webm"):
        song_data = get_reactions_manifest(artist, song)
        v_id = song_data["main_song"]["id"]

        cmd = f"yt-dlp -o \"{song_file + '.webm'}\" https://www.youtube.com/watch\?v\={v_id}\;"
        # print(cmd)
        subprocess.run(cmd, shell=True, check=True)

    else:
        print(f"{song_file} exists")

    weird_output = os.path.join(song_file + ".webm.mp4")
    if os.path.exists(weird_output):
        os.rename(weird_output, song_file + ".mp4")

    matching_files = glob.glob(f"{song_file}.*")
    if any(f.endswith((".mp4", ".webm")) for f in matching_files):
        check_and_fix_fps(matching_files[0])


def download_included_reactions(song_directory, artist, song):
    full_reactions_path = os.path.join(song_directory, "reactions")
    if not os.path.exists(full_reactions_path):
        # Create a new directory because it does not exist
        os.makedirs(full_reactions_path)

    reaction_inventory = get_selected_reactions(artist, song)

    for channel, reaction in reaction_inventory.items():
        v_id = reaction["id"]

        output = os.path.join(full_reactions_path, channel + ".webm")
        extracted_output = os.path.join(full_reactions_path, channel + ".mp4")

        weird_output = os.path.join(full_reactions_path, channel + ".webm.mp4")
        if os.path.exists(weird_output):
            os.rename(weird_output, extracted_output)

        if not os.path.exists(output) and not os.path.exists(extracted_output):
            cmd = f'yt-dlp -o "{output}" -S vcodec:h264,res,acodec:m4a https://www.youtube.com/watch\?v\={v_id}\;'
            # print(cmd)

            try:
                subprocess.run(cmd, shell=True, check=True)
            except:
                print(f"Failed to download {output} {cmd}")

            weird_output = os.path.join(full_reactions_path, channel + ".webm.mp4")
            if os.path.exists(weird_output):
                os.rename(weird_output, extracted_output)

        # check_and_fix_fps(extracted_output)

        # Get all reaction video files
        mkv_videos = glob.glob(os.path.join(full_reactions_path, "*.mkv"))

        # Convert all mkv videos to mp4 in the same directory and then delete the mkv videos

        for mkv_video in mkv_videos:
            print(f"HANDLING MKV {mkv_video}")
            # Strip existing video extension from filename
            base_name = os.path.basename(mkv_video)
            file_name_without_ext = os.path.splitext(base_name)[0]
            # If the stripped filename still has an extension, remove that too
            if any(ext in file_name_without_ext for ext in [".webm", ".mp4", ".mkv"]):
                file_name_without_ext = os.path.splitext(file_name_without_ext)[0]

            output_file = os.path.join(full_reactions_path, file_name_without_ext + ".mp4")

            print(f"OUTPUT FILE {output_file}")
            ffmpeg.input(mkv_video).output(output_file).run()
            os.remove(mkv_video)

        if os.path.exists(extracted_output):
            reaction_file = extracted_output
        else:
            reaction_file = output


def get_selected_reactions(artist, song, filter_by_downloaded=True):
    song_data = get_reactions_manifest(artist, song)

    reaction_inventory = {}
    for _, reaction in song_data["reactions"].items():
        key = reaction["reactor"]

        if reaction.get("download") or (not filter_by_downloaded and key not in reaction_inventory):
            if key in reaction_inventory:
                key = (
                    reaction["reactor"] + "_" + reaction["id"]
                )  # handle multiple reactions for a single channel
                if reaction.get("file_prefix", False) != key:
                    reaction["file_prefix"] = key
                    save_reactions_manifest(song_data, artist, song)

            reaction_inventory[key] = reaction
    return reaction_inventory


def generate_description_text(artist, song):
    manifest = get_reactions_manifest(artist, song)

    reactions = get_selected_reactions(artist, song)

    for channel, reaction in reactions.items():
        print(f"\t{channel}: https://youtube.com/watch?v={reaction.get('id')}")


def prepare_title(title):
    title = title.encode("ascii", "ignore")
    title = title.decode()

    return (
        title.replace("&quot;", "")
        .replace("-", " - ")
        .replace("&#39;", "'")
        .replace("“", "")
        .replace("”", "")
        .replace('"', "")
    )


def filter_and_augment_manifest(artist, song, force=False):
    from utilities import conf

    manifest_file = get_manifest_path(artist, song)
    song_data = json.load(open(manifest_file))

    f = os.path.join(f"library/{artist} - {song}.json")
    song_def = json.load(open(f))

    song = song_data["main_song"]
    if force or not song.get("duration", False):
        search = f"{artist} {prepare_title(song.get('title'))}"
        print(search)
        result = YoutubeSearch(search, max_results=1).to_dict()[0]

        song["duration"] = result["duration"]
        song["views"] = int(result["views"].replace(",", "").replace(" views", ""))

    duration = song.get("duration")
    song_duration = int(duration.split(":")[0]) * 60 + int(duration.split(":")[1])

    reaction_inventory = song_data["reactions"]

    print("Filtering and augmenting manifest")

    to_delete = []

    test = conf.get("search_tester", None)

    channels = get_recommended_channels(include_eligible=True)
    reaction_channels = {}
    for ch in channels:
        reaction_channels[ch.get("channelId")] = ch

    for idx, (vid, reaction) in enumerate(reaction_inventory.items()):
        if reaction.get("id") not in song_def.get("include_videos", []) and not test(
            reaction.get("title"), reaction.get("channelId") in reaction_channels
        ):
            print("DELETING", reaction.get("id"), song_def.get("include_videos", []))
            to_delete.append(reaction.get("id"))
            continue

        if force or not reaction.get("duration", False):
            search = f"{prepare_title(reaction.get('title'))} \"{reaction.get('id')}\""

            print(
                f"\t[{100 * idx / len(reaction_inventory.keys()):.1f}%] {search}",
                end="\r",
            )

            # search = f"\"{reaction.get('reactor')}\" \"{artist}\"  \"{song.get('title')}\"  "

            results = YoutubeSearch(
                search, max_results=5, channel=get_channel(reaction.get("channelId"))
            ).to_dict()
            if results is None:
                continue

            result = None
            for r in results:
                if r["id"] == reaction.get("id"):
                    result = r
                    break

            if result is None:
                search = f"\"{reaction.get('id')}\""
                try:
                    results = YoutubeSearch(search, max_results=3).to_dict()

                    result = None
                    for r in results:
                        if r["id"] == reaction.get("id"):
                            result = r
                            break

                    if result == None:
                        print(search)
                        print(result)
                        print(reaction)
                        continue
                except:
                    print(search)
                    print(result)
                    print(reaction)
                    continue

            assert result

            duration = reaction["duration"] = result["duration"]
            try:
                reaction["views"] = int(result["views"].replace(",", "").replace(" views", ""))
            except:
                reaction["views"] = -1

            if isinstance(duration, str):
                minutes = hours = seconds = 0
                ts = duration.split(":")
                if len(ts) == 3:
                    hours, minutes, seconds = ts
                elif len(ts) == 2:
                    minutes, seconds = ts
                else:
                    seconds = ts[0]
                hours = int(hours) * 60 * 60
                minutes = int(minutes) * 60
                seconds = int(seconds)

            else:
                hours = 0
                minutes = 0
                seconds = duration

            reaction_duration = minutes + seconds
            if song_duration > reaction_duration - 5:
                print(
                    "DELETING for DURATION",
                    reaction.get("id"),
                    song_duration,
                    reaction_duration,
                )
                to_delete.append(vid)

    for vid in to_delete:
        print("FILTERING", vid)
        del reaction_inventory[vid]

    song_data["reactions"] = reaction_inventory

    manifest_json = json.dumps(song_data, indent=4)

    jsonfile = open(manifest_file, "w")
    jsonfile.write(manifest_json)
    jsonfile.close()


def migrate_reactions():
    list_subfolders_with_paths = [
        os.path.join(f.path, "manifest.json") for f in os.scandir("Media") if f.is_dir()
    ]
    list_subfolders_with_paths = [f for f in list_subfolders_with_paths if os.path.exists(f)]

    for song_manifest in list_subfolders_with_paths:
        song_data = json.load(open(song_manifest))
        reaction_inventory = song_data["reactions"]

        to_delete = []

        for ___, reaction in reaction_inventory.items():
            # migration logic here
            if "views" in reaction:
                reaction["views"] = int(reaction["views"])
            continue

        for vid in to_delete:
            print("FILTERING", vid)
            del reaction_inventory[vid]

        song_data["reactions"] = reaction_inventory

        manifest_json = json.dumps(song_data, indent=4)

        jsonfile = open(song_manifest, "w")
        jsonfile.write(manifest_json)
        jsonfile.close()


def reaction_stats(song):
    song_manifest = os.path.join("Media", song, "manifest.json")
    song_data = json.load(open(song_manifest))
    reaction_inventory = song_data["reactions"]
    reactions = list(reaction_inventory.values())
    reactions.sort(key=lambda x: -x.get("views", 0))
    num_views = 0
    for reaction in reactions:
        num_views += reaction.get("views", 0)
        print(f"{reaction.get('views', 0)} - {reaction.get('reactor')}")

    return {"count": len(reaction_inventory.keys()), "views": num_views}


if __name__ == "__main__":
    # migrate_reactions()

    # generate_description_text("Ren", "Hi Ren")

    stats = reaction_stats("Ren - Fred Again Mash Up")
    print(stats)

    # manifest_file = get_manifest_path("Ren", "Money Game Part 1")
    # song_data = json.load(open(manifest_file))

    # downloaded_m1 = get_selected_reactions(song_data, True)
    # available_m1 = get_selected_reactions(song_data, False)

    # manifest_file = get_manifest_path("Ren", "Money Game Part 2")
    # song_data = json.load(open(manifest_file))

    # downloaded_m2 = get_selected_reactions(song_data, True)
    # available_m2 = get_selected_reactions(song_data, False)

    # manifest_file = get_manifest_path("Ren", "Money Game Part 3")
    # song_data = json.load(open(manifest_file))

    # downloaded_m3 = get_selected_reactions(song_data, True)
    # available_m3 = get_selected_reactions(song_data, False)

    # reactors_used = {}
    # for inv in [downloaded_m1, downloaded_m2, downloaded_m3]:
    #     for k,r in inv.items():
    #         reactors_used[k] = True

    # for i, inv in enumerate([(downloaded_m1,available_m1), (downloaded_m2,available_m2), (downloaded_m3,available_m3)]):
    #     (used, avail) = inv
    #     print(f"Money Game Part {i+1} -- including {len(used.keys())} reactors of {len(avail.keys())} available")
    #     for k in reactors_used.keys():
    #         if k not in used and k in avail:
    #             print(f"\tConsider including {k} / https://www.youtube.com/watch?v={avail[k]['id']}")
