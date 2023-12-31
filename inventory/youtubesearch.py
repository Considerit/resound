# Modified from: https://github.com/joetats/youtube_search/blob/master/youtube_search/__init__.py

import requests
import urllib.parse
import json
import time



# https://www.youtube.com/@jonathanhiller_/search?query=ren


class YoutubeSearch:
    def __init__(self, search_terms: str, max_results=None, channel=None):
        self.search_terms = search_terms
        self.max_results = max_results
        if channel is None:
            self.url = "https://youtube.com/results?search_query"
        else:
            #self.url = f"https://youtube.com/{channel.get('customUrl')}/search?query"
            self.url = f"https://www.youtube.com/channel/{channel.get('channelId')}/search?query"

        self.videos = self._search()




    def _fetch(self, url):
        response = None
        try: 
            response = requests.get(url, timeout=5).text
            time.sleep(1.5)
        except Exception as e:
            print("Problem with the request, pausing for 10 sec")
            time.sleep(10)
            print('...ok trying again')

            try: 
                response = requests.get(url).text
            except Exception as e:
                print("Got ANOTHER connection reset, pausing for 90 sec")
                time.sleep(60)
                print('...ok trying again')
                response = requests.get(url).text

        return response

    def _search(self):
        encoded_search = urllib.parse.quote_plus(self.search_terms)
        
        url = self.url + f"={encoded_search}" #f"{BASE_URL}/results?search_query={encoded_search}"


        response = self._fetch(url)
        i = 0 
        while "ytInitialData" not in response:
            response = self._fetch(url)
            if i > 5:
                print("...looped too many times", url)
                return None
            i += 1



        results = self._parse_html(response, url)

        if self.max_results is not None and results and len(results) > self.max_results:
            return results[: self.max_results]
        return results

    def _parse_html(self, response, url):
        results = []



        start = (
            response.index("ytInitialData")
            + len("ytInitialData")
            + 3
        )
        end = response.index("};", start) + 1
        json_str = response[start:end]
        data = json.loads(json_str)

        head = data["contents"].get("twoColumnSearchResultsRenderer", None)

        if head:
            for contents in head["primaryContents"]["sectionListRenderer"]["contents"]:
                if "itemSectionRenderer" in contents:
                    for video in contents["itemSectionRenderer"]["contents"]:
                        res = {}
                        if "videoRenderer" in video.keys():
                            video_data = video.get("videoRenderer", {})
                            
                            # print("\n\nVIDEO DATA")
                            # for k,v in video_data.items():
                            #     print(k, v)
                            res["id"] = video_data.get("videoId", None)
                            res["thumbnails"] = [thumb.get("url", None) for thumb in video_data.get("thumbnail", {}).get("thumbnails", [{}]) ]
                            res["title"] = video_data.get("title", {}).get("runs", [[{}]])[0].get("text", None)
                            res["long_desc"] = video_data.get("descriptionSnippet", {}).get("runs", [{}])[0].get("text", None)
                            res["channel"] = video_data.get("longBylineText", {}).get("runs", [[{}]])[0].get("text", None)
                            res["channelId"] = video_data.get("longBylineText", {}).get("runs", [[{}]])[0].get('navigationEndpoint', {}).get("browseEndpoint", {}).get('browseId', None)

                            res["duration"] = video_data.get("lengthText", {}).get("simpleText", 0)
                            res["views"] = video_data.get("viewCountText", {}).get("simpleText", 0)
                            res["publish_time"] = video_data.get("publishedTimeText", {}).get("simpleText", 0)
                            res["url_suffix"] = video_data.get("navigationEndpoint", {}).get("commandMetadata", {}).get("webCommandMetadata", {}).get("url", None)
                            
                            # for k,v in res.items():
                            #     print(k,v)

                            results.append(res)


            if results:
                return results

        if not head:

          try: 
              head = [t for t in data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"] if t.get('expandableTabRenderer', False)]
              if not head or len(head) == 0:
                print(url)
                return None
              head = head[0]['expandableTabRenderer']
              for contents in head["content"]["sectionListRenderer"]["contents"]:
                if "itemSectionRenderer" in contents:
                    for video in contents["itemSectionRenderer"]["contents"]:
                        res = {}
                        if "videoRenderer" in video.keys():
                            video_data = video.get("videoRenderer", {})
                            
                            # print("\n\nVIDEO DATA")
                            # for k,v in video_data.items():
                            #     print(k, v)
                            res["id"] = video_data.get("videoId", None)
                            res["thumbnails"] = [thumb.get("url", None) for thumb in video_data.get("thumbnail", {}).get("thumbnails", [{}]) ]
                            res["title"] = video_data.get("title", {}).get("runs", [[{}]])[0].get("text", None)
                            res["long_desc"] = video_data.get("descriptionSnippet", {}).get("runs", [{}])[0].get("text", None)
                            res["channel"] = video_data.get("longBylineText", {}).get("runs", [[{}]])[0].get("text", None)
                            res["channelId"] = video_data.get("longBylineText", {}).get("runs", [[{}]])[0].get('navigationEndpoint', {}).get("browseEndpoint", {}).get('browseId', None)

                            res["duration"] = video_data.get("lengthText", {}).get("simpleText", 0)
                            res["views"] = video_data.get("viewCountText", {}).get("simpleText", 0)
                            res["publish_time"] = video_data.get("publishedTimeText", {}).get("simpleText", 0)
                            res["url_suffix"] = video_data.get("navigationEndpoint", {}).get("commandMetadata", {}).get("webCommandMetadata", {}).get("url", None)
                            
                            # for k,v in res.items():
                            #     print(k,v)

                            results.append(res)

              if results:
                return results
          except Exception as e:
            print("EXCEPTION!", e)
            raise e
            return None 


        # print('')
        # print(json.dumps(head))
        # print('')

        return results

    def to_dict(self, clear_cache=True):
        result = self.videos
        if clear_cache:
            self.videos = ""
        return result

    def to_json(self, clear_cache=True):
        result = json.dumps({"videos": self.videos})
        if clear_cache:
            self.videos = ""
        return result