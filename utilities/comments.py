from itertools import islice
from youtube_comment_downloader import *

from utilities import save_object_to_file, read_object_from_file


def download_comments(vids, fname):
    downloader = YoutubeCommentDownloader()

    all_comments = []
    for vid in vids:
        comments = downloader.get_comments_from_url(f'https://www.youtube.com/watch?v={vid}', sort_by=SORT_BY_POPULAR)
        all_comments += comments

    save_object_to_file(fname, all_comments)
    return all_comments

def simplify_comments(comments, fname, include_replies=False, comment_length_min=100, vote_minimum=10):



    simplified_comments = [ f"{c['author']}: \"{c['text']}\"  (votes: {c['votes']})"  for c in comments if ( 'K' in c['votes'] or int(c["votes"]) >= vote_minimum) and len(c["text"]) > comment_length_min and (include_replies or not c["reply"]) ]

    save_object_to_file(fname, simplified_comments)

if __name__ == '__main__':


    # resound vids: vids = ['8WHf-jJkx78', 'B4bstHZsz0k', 'PLmHQPpOUHw', 'FUNhMfdsblE', '4I48W3XwsEw']
    # chalk outlines:
    vids = ['35yALr_opeg']
    fname = 'chalk_outlines_comments.json'

    # download_comments(vids, fname)

    comments = read_object_from_file(fname)

    comments = simplify_comments(comments, "chalk_outlines_comments.txt", include_replies=False)
