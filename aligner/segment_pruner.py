import os, math
from utilities import conversion_audio_sample_rate as sr
from utilities import conf

from scipy.signal import correlate
import numpy as np
import matplotlib.pyplot as plt

from aligner.scoring_and_similarity import (
    get_segment_mfcc_cosine_similarity_score,
)

from aligner.path_finder import prune_cache, near_song_end, near_song_beginning


def prune_unreachable_segments(reaction, segments, allowed_spacing, prune_links=False):
    segments = [s for s in segments if not s.get("pruned", False)]

    starting_segment_num = len(segments)
    segments.sort(key=lambda x: x["end_points"][1])  # sort by reaction_end

    joinable_segments = {}

    for segment in segments:
        segment_id = segment["key"]
        joins = joinable_segments[segment_id] = get_joinable_segments(
            reaction, segment, segments, allowed_spacing, prune_links=prune_links
        )

        at_end = segment["at_end"] = near_song_end(segment, allowed_spacing)

        if len(joins) == 0 and not at_end:
            # if prune_links:
            #     print("PRUNING BECAUSE NO JOINS FOUND", segment_id)
            #     song_length = len(conf.get('song_audio_data'))
            #     print(song_length/sr, (song_length - segment['end_points'][3])/sr, allowed_spacing/sr)

            segment["pruned"] = True
            del joinable_segments[segment_id]
            if prune_cache:
                prune_cache["unreachable"] += 1

    # segments = [s for s in segments if not s.get("pruned", False)]

    # clean up segments that can't reach the end
    pruned_another = True
    while pruned_another:
        segments = [s for s in segments if not s.get("pruned", False)]

        pruned_another = False
        for segment in segments:
            segment_id = segment["key"]
            if segment_id in joinable_segments:
                joins = joinable_segments[segment_id] = [
                    s for s in joinable_segments[segment_id] if not s.get("pruned", False)
                ]

                if len(joins) == 0 and not segment.get("at_end", False):
                    segment["pruned"] = True
                    del joinable_segments[segment_id]
                    pruned_another = True
                    if prune_cache:
                        prune_cache["unreachable"] += 1

                    # if prune_links:
                    #     print("PRUNING BECAUSE CANT REACH END", segment_id)

    # now clean up all segments that can't be reached from the start
    pruned_another = True

    while pruned_another:
        pruned_another = False

        segments_reached = {}
        for segment_id, joins in joinable_segments.items():
            joins = joinable_segments[segment_id] = [s for s in joins if not s.get("pruned", False)]
            for s in joins:
                segments_reached[s["key"]] = True

        segments = [s for s in segments if not s.get("pruned", False)]

        for segment in segments:
            segment_id = segment["key"]
            if segment_id not in segments_reached and not near_song_beginning(
                segment, allowed_spacing
            ):
                segment["pruned"] = True
                if segment_id in joinable_segments:
                    del joinable_segments[segment_id]
                pruned_another = True
                if prune_cache:
                    prune_cache["unreachable"] += 1

                # if prune_links:
                #     print("PRUNING BECAUSE CANT REACH FROM STARET", segment_id)

    segments = [s for s in segments if not s.get("pruned", False)]

    # for k,v in joinable_segments.items():
    #     print(k,v)

    print(
        f"Pruned unreachable segments: {len(segments)} remaining of {starting_segment_num} to start"
    )

    return segments, joinable_segments


# returns all segments that a given segment could jump to
def get_joinable_segments(reaction, segment, all_segments, allowed_spacing, prune_links):
    if segment.get("pruned", True):
        return []

    ay, by, ax, bx = segment["end_points"]
    candidates = []
    for candidate in all_segments:
        if candidate.get("pruned", False):
            continue

        cy, dy, cx, dx = candidate["end_points"]

        if ay == cy and by == dy and ax == cx and bx == dx:
            continue

        on_top = cy - cx > ay - ax  # solving for b in y=ax+b, where a=1
        over_each_other = bx + allowed_spacing >= cx - allowed_spacing and ax < dx

        extends_past = dx > bx  # Most of the time this is correct, but in the future we
        # might want to support link joins that aren't from the end
        # of one segment.

        if on_top and over_each_other and (extends_past or not prune_links):
            candidates.append(candidate)

    if prune_links:
        candidates = prune_poor_links(reaction, segment, candidates)

    return candidates


def prune_poor_segments(
    reaction, segments, min_segments_at_checkpoint=4, quality_threshold=0.8, length_threshold=0.5
):
    segments = [s for s in segments if not s.get("pruned")]
    unpruned_to_start = len(segments)

    checkpoints = [i for i in range(0, conf.get("song_length"), sr)]

    segments_at_checkpoint = {}

    for checkpoint in checkpoints:
        segments_at_checkpoint[checkpoint] = []
        for segment in segments:
            if segment["end_points"][2] <= checkpoint <= segment["end_points"][3]:
                segments_at_checkpoint[checkpoint].append(segment)

    for checkpoint, ck_segments in segments_at_checkpoint.items():
        # print("Checkpoint", checkpoint / sr)
        # ck_segments = [s for s in ck_segments if not s.get("pruned")]
        if len(ck_segments) < min_segments_at_checkpoint:
            continue

        max_quality = -1
        max_length = -1
        # max_time = -1
        # min_time = 999999999999999999999999999999999
        for s in ck_segments:
            if "mfcc_cosine_score" not in s:
                s["mfcc_cosine_score"] = get_segment_mfcc_cosine_similarity_score(
                    reaction, s["end_points"]
                )
                s["length"] = s["end_points"][3] - s["end_points"][2]

            if max_quality < s["mfcc_cosine_score"]:
                max_quality = s["mfcc_cosine_score"]
            if max_length < s["length"]:
                max_length = s["length"]
            # if max_time < s["end_points"][1]:
            #     max_time = s["end_points"][1]
            # if min_time > s["end_points"][0]:
            #     min_time = s["end_points"][0]

        for s in ck_segments:
            if (
                s["mfcc_cosine_score"]
                < quality_threshold * max_quality
                # and s["length"] < length_threshold * max_length
                # and s["end_points"][0] > min_time
                # and s["end_points"][1] < max_time
            ):
                covering_segments = []

                for s2 in ck_segments:
                    if s == s2 or s2["pruned"]:
                        continue
                    if (
                        s2["end_points"][0] <= s["end_points"][0]
                        and s2["end_points"][1] >= s["end_points"][1]
                    ):
                        covering_segments.append(s2)

                # print("\tprune candidate:", s["end_points"][1] / sr, len(covering_segments))

                max_covering_score = -1
                for cs in covering_segments:
                    cs_reaction_start, cs_reaction_end, cs_music_start, cs_music_end = cs[
                        "end_points"
                    ]
                    covering_segment = [
                        cs_reaction_start + (cs_reaction_start - s["end_points"][0]),
                        cs_reaction_end - (cs_reaction_end - s["end_points"][1]),
                        s["end_points"][2],
                        s["end_points"][3],
                    ]
                    covering_score = get_segment_mfcc_cosine_similarity_score(
                        reaction, covering_segment
                    )
                    if covering_score > max_covering_score:
                        max_covering_score = covering_score

                    # print("\t\tCovering score: ", covering_score, s["mfcc_cosine_score"])

                if len(covering_segments) > 0 and s["mfcc_cosine_score"] < max_covering_score * (
                    quality_threshold + (1 - quality_threshold) / 2
                ):
                    s["pruned"] = True

            # print(
            #     f"\t{'*' if s['pruned'] else ' '} length={int(s['length'] / max_length * 100)}  quality={int(s['mfcc_cosine_score'] / max_quality * 100)}"
            # )

    unpruned_segments = [s for s in segments if not s.get("pruned")]
    print(
        f"\tpruning poor segments resulted in {len(unpruned_segments)} segments, down from {unpruned_to_start}"
    )

    return unpruned_segments


def get_link_score(reaction, link, reaction_start, base_start):
    link_segment = copy.copy(link["end_points"])

    filler = 0
    if link_segment[0] > reaction_start:  # if we need filler
        filler = link_segment[0] - reaction_start
        # link_segment[0] += filler
        # link_segment[2] += filler

    link_segment[0] = max(reaction_start, link_segment[0])
    link_segment[2] = max(base_start, link_segment[2])

    filler_penalty = 1 - filler / (link_segment[3] - link_segment[2])
    score_link = filler_penalty * get_segment_mfcc_cosine_similarity_score(reaction, link_segment)
    return score_link


def prune_poor_links(reaction, segment, links):
    # Don't follow some branches when a prior branch looks substantially better
    # and extends further into the future

    link_prune_threshold = 0.9

    good_links = []

    # ...for the link
    reaction_start = segment["end_points"][1]
    base_start = segment["end_points"][3]

    # sort links by reverse reaction_end
    # links.sort( key=lambda x: x['end_points'][1], reverse=True )

    for i, link in enumerate(links):
        prune = False
        score_link = get_link_score(reaction, link, reaction_start, base_start)
        link_intercept = link["end_points"][0] - link["end_points"][2]

        for j, pruning_link in enumerate(links):
            if j == i:
                continue

            pruning_link_intercept = pruning_link["end_points"][0] - pruning_link["end_points"][2]

            if link_intercept < pruning_link_intercept:  # only prune links that come later
                continue

            if pruning_link["end_points"][3] < link["end_points"][3]:
                continue

            assert pruning_link["end_points"][3] > base_start

            # Now we want to compare the score of pruning_link over the scope
            # of link, to link, and if it is greater by a certain threshold, prune link.
            # One tricky thing is that we might need filler for one or both of them.

            score_pruning_link = get_link_score(reaction, pruning_link, reaction_start, base_start)

            if score_pruning_link * link_prune_threshold > score_link:
                prune = True
                prune_cache["poor_link"] += 1
                break

        if not prune:
            good_links.append(link)

    return good_links
