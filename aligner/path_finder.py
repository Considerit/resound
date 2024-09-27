import time, os, copy

from utilities import conversion_audio_sample_rate as sr
from utilities import conf

from prettytable import PrettyTable
from aligner.visualize_alignment import GENERATE_FULL_ALIGNMENT_VIDEO
from aligner.scoring_and_similarity import (
    print_path,
    path_alignment_score,
    get_segment_score,
)

from aligner.align_by_audio.pruning_search import is_path_quality_poor

from utilities import print_profiling

location_cache = {}
best_score_cache = {}
prune_cache = {}


def initialize_paint_caches():
    location_cache.clear()
    best_score_cache.clear()
    best_score_cache.update({"best_overall_path": None, "best_overall_score": None})
    prune_cache.clear()
    prune_cache.update(
        {
            "location": 0,
            "best_score": 0,
            "poor_path": 0,
            "poor_link": 0,
            "neighbor": 0,
            "unreachable": 0,
            "segment_quality": 0,
        }
    )


def construct_all_paths(
    reaction,
    segments,
    joinable_segment_map,
    allowed_spacing,
    allowed_spacing_on_ends,
    chunk_size,
    segments_by_key,
):
    paths = []
    song_length = conf.get("song_length")
    partial_paths = []

    start_time = time.perf_counter()
    time_of_last_best_score = start_time

    def complete_path(path):
        nonlocal time_of_last_best_score

        completed_path = copy.deepcopy(path)

        fill_len = song_length - path[-1][3]
        if fill_len > 0:
            completed_path.append(
                [
                    path[-1][1],
                    path[-1][1] + fill_len,
                    path[-1][3],
                    path[-1][3] + fill_len,
                    True,
                    None,
                ]
            )

        score = path_alignment_score(path, reaction, segments_by_key)
        if (
            best_score_cache["best_overall_path"] is None
            or best_score_cache["best_overall_score"] < score
        ):
            best_score_cache["best_overall_path"] = path
            best_score_cache["best_overall_score"] = score

            print("\nNew best score!")
            print_path(path, reaction, segments_by_key)
            time_of_last_best_score = time.perf_counter()

        # if score > 0.9 * best_score_cache["best_overall_score"]:
        paths.append(completed_path)

        if GENERATE_FULL_ALIGNMENT_VIDEO:
            splay_paint(
                reaction,
                segments,
                stroke_alpha=1,
                stroke_color="gray",
                stroke_linewidth=1,
                show_live=False,
                # best_path=best_score_cache.get("best_overall_path", None),
                paths=[completed_path],
                chunk_size=chunk_size,
                id=f"path-completed-{hash( tuple (  [tuple(t) for t in completed_path]       )     )}",
            )

    for c in segments:
        if c.get("pruned", False):
            continue

        if near_song_beginning(c, allowed_spacing_on_ends):
            reaction_start, reaction_end, base_start, base_end = c["end_points"]
            start_path = [[reaction_start, reaction_end, base_start, base_end, False, c["key"]]]
            if base_start > 0:
                start_path.insert(
                    0,
                    [
                        reaction_start - base_start,
                        reaction_start,
                        0,
                        base_start,
                        True,
                        None,
                    ],
                )

            score = path_alignment_score(start_path, reaction, segments_by_key)
            partial_paths.append([[start_path, c], score])

            if near_song_end(c, allowed_spacing_on_ends):  # in the case of one long completion
                complete_path(start_path)

    i = 0

    was_prune_eligible = False
    backlog = []
    while len(partial_paths) > 0:
        i += 1
        prune_eligible = (
            time.perf_counter() - start_time > 2 * 60
        )  # True #len(partial_paths) > 100 #or len(paths) > 10000

        if time.perf_counter() - time_of_last_best_score > 30 * 60:
            return paths

        if time.perf_counter() - time_of_last_best_score < 15 * 60:
            threshold_base = 0.7
        else:
            threshold_base = 0.85

        if i < 50000:
            sort_every = 2500
        elif i < 200000:
            sort_every = 15000
        elif i < 500000:
            sort_every = 45000
        else:
            sort_every = 75000

        if (len(partial_paths) < 10 or i % sort_every == 4900) and len(backlog) > 0:
            partial_paths.extend(backlog)
            backlog.clear()

            if prune_eligible and not was_prune_eligible:
                print("\nENABLING PRUNING\n")
                iii = len(partial_paths) - 1

                for partial, score in reversed(partial_paths):
                    if False and is_path_quality_poor(reaction, partial[0]):
                        prune_cache["poor_path"] += 1
                        partial_paths.pop(iii)
                    else:
                        partial_path = []
                        for segment in partial[0]:
                            partial_path.append(segment)
                            should_prune, score = should_prune_path(
                                reaction,
                                partial_path,
                                song_length,
                                segments_by_key,
                                threshold_base=threshold_base,
                            )
                            if should_prune:
                                partial_paths.pop(iii)
                                break
                    iii -= 1
                was_prune_eligible = True
                continue

        if i % 10 == 1 or len(partial_paths) > 1000:
            partial_paths.sort(key=lambda x: x[1], reverse=True)
            if len(partial_paths) > 200:
                backlog.extend(partial_paths[200:])
                del partial_paths[200:]

        if i % 1000 == 999:
            # print_prune_data()
            print_profiling()

        partial_path, score = partial_paths.pop(0)

        # print(len(partial_paths), end='\r')

        should_prune, score = should_prune_path(
            reaction,
            partial_path[0],
            song_length,
            segments_by_key,
            score,
            threshold_base=threshold_base,
        )

        if False and GENERATE_FULL_ALIGNMENT_VIDEO:
            splay_paint(
                reaction,
                segments,
                stroke_alpha=1,
                stroke_color="gray",
                stroke_linewidth=1,
                show_live=False,
                # best_path=best_score_cache.get("best_overall_path", None),
                paths=[partial_path[0]],
                chunk_size=chunk_size,
                id=f"path-test-{i}",
            )

        if should_prune and prune_eligible:
            continue

        print(
            f"{(time.perf_counter() - time_of_last_best_score) / 60:.1f}",
            len(partial_paths) + len(backlog),
            len(paths),
            len(partial_path[0]),
            end="\r",
        )

        if partial_path[1]["key"] in joinable_segment_map:
            next_partials = branch_from(
                reaction,
                partial_path,
                joinable_segment_map[partial_path[1]["key"]],
                allowed_spacing,
            )

            for partial in next_partials:
                path, last_segment = partial
                if near_song_end(last_segment, allowed_spacing_on_ends):
                    complete_path(path)

                should_prune, score = should_prune_path(
                    reaction, path, song_length, segments_by_key, threshold_base=threshold_base
                )
                if not should_prune and prune_eligible and is_path_quality_poor(reaction, path):
                    prune_cache["poor_path"] += 1
                    should_prune = True

                if (
                    (not should_prune)
                    and last_segment["key"] in joinable_segment_map
                    and len(joinable_segment_map[last_segment["key"]]) > 0
                ):
                    partial_paths.append([partial, score])

    return paths


def should_prune_path(reaction, path, song_length, segments_by_key, score=None, threshold_base=0.7):
    reaction_end = path[-1][1]

    score = path_alignment_score(path, reaction, segments_by_key)

    # prune based on path length
    path_length = len(path)
    if (
        path_length > 20
        and best_score_cache["best_overall_path"]
        and path_length > 2 * len(best_score_cache["best_overall_path"])
    ):
        return True, score

    prune_for_location = should_prune_for_location(
        reaction, path, song_length, score, threshold_base=threshold_base
    )

    if prune_for_location:
        prune_cache["location"] += 1
        return True, score

    return False, score


def should_prune_for_location(reaction, path, song_length, score, threshold_base=0.7):
    global location_cache

    last_segment = path[-1]
    location_key = str(last_segment)
    segment_quality_threshold = 0.750

    has_bad_segment = False
    for segment in path:
        if not segment[4]:
            # NOTE: I changed this from segment[-1] because I think that was
            #       the original intention (score if not filler), but the
            #       format of segment has since changed.
            segment_score = get_segment_score(reaction, segment)
            if segment_score < segment_quality_threshold:
                has_bad_segment = True
                break

    location_prune_threshold = threshold_base  # .7 + .25 * last_segment[3] / song_length
    if has_bad_segment:
        location_prune_threshold = 0.99

    if location_key in location_cache:
        best_score = location_cache[location_key]
        if score >= best_score:
            location_cache[location_key] = score
        else:
            prunable = location_prune_threshold * best_score > score
            if prunable:
                # print('\npruned!', best_score[2], score[2])
                return True
    else:
        location_cache[location_key] = score

    return False


def branch_from(reaction, partial_path, joinable_segments, allowed_spacing):
    branches = []

    current_path, last_segment = partial_path

    reaction_end = current_path[-1][1]
    base_end = current_path[-1][3]

    b_current = reaction_end - base_end  # as in, y = ax + b

    for candidate in joinable_segments:
        (
            candidate_reaction_start,
            candidate_reaction_end,
            candidate_base_start,
            candidate_base_end,
        ) = candidate["end_points"]

        distance = base_end - candidate_base_start
        if candidate_base_end > base_end and distance > -allowed_spacing:
            branch = copy.copy(current_path)

            if distance < 0:
                branch.append(
                    [
                        reaction_end,
                        reaction_end - distance,
                        base_end,
                        base_end - distance,
                        True,
                        None,
                    ]
                )
                filled = -distance
            else:
                filled = 0

            b_candidate = candidate_reaction_end - candidate_base_end

            branch.append(
                [
                    reaction_end + filled + b_candidate - b_current,
                    candidate_reaction_end,
                    base_end + filled,
                    candidate_base_end,
                    False,
                    candidate["key"],
                ]
            )

            branches.append([branch, candidate])

    return branches


def near_song_beginning(segment, allowed_spacing):
    return segment["end_points"][2] < allowed_spacing


def near_song_end(segment, allowed_spacing):
    song_length = conf.get("song_length")
    return (
        song_length - segment["end_points"][3] < 3 * allowed_spacing
    )  # because of weird ends of songs, let it be farther away


def print_prune_data():
    x = PrettyTable()
    x.border = False
    x.align = "r"
    x.field_names = ["\t", "Prune type", "Count"]

    for k, v in prune_cache.items():
        x.add_row(["\t", k, v])

    print(x)
