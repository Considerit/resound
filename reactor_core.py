import os
import copy
import glob
import json

from prettytable import PrettyTable

from utilities import prepare_reactions, extract_audio, conf, make_conf, unload_reaction
from utilities import conversion_audio_sample_rate as sr
from inventory import (
    download_and_parse_reactions,
    get_manifest_path,
    filter_and_augment_manifest,
)
from face_finder.face_finder import create_reactor_view, get_face_files
from backchannel_isolator import (
    isolate_reactor_backchannel,
    get_reactor_backchannel_path,
)
from compositor import create_reaction_concert
from compositor.asides import create_asides

from aligner import create_alignment_for_reaction_video, write_aligned_reaction_video
from aligner.scoring_and_similarity import ground_truth_overlap

import cProfile
import pstats

from utilities import print_profiling
from utilities.locks import (
    is_locked,
    request_lock,
    free_lock,
    free_all_locks,
    other_locks,
)


def remove_unneeded_files(song_def):
    song = f"{song_def['artist']} - {song_def['song']}"
    song_directory = os.path.join("Media", song)

    print(f"Cleaning up {song}...")

    wav_files = glob.glob(
        f"{song_directory}/reactions/**/vocals-post-high-passed.wav", recursive=True
    )
    wav_files += glob.glob(f"{song_directory}/reactions/**/accompaniment.wav", recursive=True)
    wav_files += glob.glob(f"{song_directory}/reactions/**/vocals.wav", recursive=True)

    # Delete each .wav file
    for wav_file in wav_files:
        # if 'isolated_backchannel' in wav_file:
        #     continue

        if f"/reactions/{song}" in wav_file:
            continue

        try:
            os.remove(wav_file)
            print(f"\tDeleted: {wav_file}")
        except Exception as e:
            print(f"Error occurred while deleting file {wav_file}: {e}")


def clean_up(song_def, on_ice=False):
    song = f"{song_def['artist']} - {song_def['song']}"
    song_directory = os.path.join("Media", song)

    print(f"Cleaning up {song}...")

    wav_files = glob.glob(f"{song_directory}/**/*.wav", recursive=True)

    # Delete each .wav file
    for wav_file in wav_files:
        # if 'isolated_backchannel' in wav_file:
        #     continue

        try:
            os.remove(wav_file)
            print(f"\tDeleted: {wav_file}")
        except Exception as e:
            print(f"Error occurred while deleting file {wav_file}: {e}")

    if on_ice:
        return

    mp4_files = glob.glob(f"{song_directory}/bounded/*CROSS-EXPANDER*.mp4", recursive=True)
    mp4_files = mp4_files + glob.glob(f"{song_directory}/bounded/*-aside-*.mp4", recursive=True)
    mp4_files = mp4_files + glob.glob(f"{song_directory}/reactions/*.mp4", recursive=True)

    for mp4 in mp4_files:
        # if 'cropped' in mp4:
        #     continue

        if "Resound" in mp4:
            continue

        try:
            os.remove(mp4)
            print(f"\tDeleted: {mp4}")
        except Exception as e:
            print(f"Error occurred while deleting file {mp4}: {e}")

    webm_files = glob.glob(f"{song_directory}/**/*.webm", recursive=True)

    for webm in webm_files:
        try:
            os.remove(webm)
            print(f"\tDeleted: {webm}")
        except Exception as e:
            print(f"Error occurred while deleting file {webm}: {e}")


def reaction_fully_processed(reaction):
    return len(get_face_files(reaction)) > 0 and os.path.exists(
        get_reactor_backchannel_path(reaction)
    )


def handle_reaction_video(reaction, jobs, extend_by=15):
    # if '40' not in reaction['channel']:
    #     return

    # print("processing ", reaction['channel'])
    # Create the output video file name
    if "create_alignment" in jobs:
        create_alignment_for_reaction_video(reaction)

    if "output_alignment_video" in jobs:
        write_aligned_reaction_video(reaction, extend_by=extend_by)

    if "isolate_commentary" not in jobs:
        return []

    reaction["backchannel_audio"] = isolate_reactor_backchannel(reaction, extended_by=extend_by)

    if "create_reactor_view" not in jobs:
        return []

    # backchannel_audio is used by create_reactor_view to replace the audio track of the reactor trace
    reaction["reactors"], __ = create_reactor_view(reaction)

    if reaction["asides"]:
        create_asides(reaction)


from moviepy.editor import VideoFileClip


def create_reaction_compilation(
    song_def: dict,
    progress,
    output_dir: str = "aligned",
    include_base_video=True,
    extend_by=12,
    options={},
    jobs=[],
):
    job_sequence = jobs
    jobs = {j: max_workers for j, max_workers in jobs}
    failed_reactions = []

    try:
        make_conf(song_def, options, output_dir)

        if is_locked("compilation"):
            return []

        song_directory = conf.get("song_directory")

        compilation_path = conf.get("compilation_path")

        compilation_exists = os.path.exists(compilation_path)

        if request_lock("downloading"):
            print("Processing directory", song_directory, "Outputting to", output_dir)

            if conf.get("refresh_manifest", False) or (
                not compilation_exists and (conf.get("download_and_parse", False))
            ):
                download_and_parse_reactions(
                    song_def,
                    song_def["artist"],
                    song_def["song"],
                    song_def.get(
                        "song_search",
                        f"{song_def.get('artist')} {song_def.get('song')}",
                    ),
                    song_def["search"],
                    refresh_manifest=conf.get("refresh_manifest", False),
                )

                print(
                    "Filtering and augmenting manifest",
                    song_directory,
                    "Outputting to",
                    output_dir,
                )
                filter_and_augment_manifest(
                    song_def["artist"], song_def["song"], force=False
                )  # set force=True to update view counts

        else:
            print(
                f"...Skipping {song_def['song']} because another process is already working on this video"
            )
            return []

        free_lock("downloading")

        if conf.get("only_manifest", False):
            return []

        conf.get("load_reactions")()

        all_reactions = list(conf.get("reactions").keys())
        all_reactions.sort()

        for i, channel in enumerate(all_reactions):
            reaction = conf.get("reactions").get(channel)

            print_profiling()

            if not request_lock(channel):
                continue

            try:
                # profiler = cProfile.Profile()
                # profiler.enable()

                if not reaction_fully_processed(reaction):
                    handle_reaction_video(reaction, jobs=jobs, extend_by=extend_by)

                # profiler.disable()
                # stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
                # stats.print_stats()

            except Exception as e:
                traceback.print_exc()
                print(e)
                traceback_str = traceback.format_exc()
                failed_reactions.append((reaction.get("channel"), e, traceback_str))
                conf["remove_reaction"](reaction.get("channel"))
                if conf.get("break_on_exception"):
                    raise (e)

            log_progress(progress)

            unload_reaction(channel)
            free_lock(channel)

        print_progress(progress)
        compilation_exists = os.path.exists(compilation_path)
        print("COMP EXISTS?", compilation_exists, compilation_path)
        if not compilation_exists and "create_compilation" in jobs and request_lock("compilation"):
            for channel in all_reactions:
                reaction = conf.get("reactions").get(channel)
                reaction["backchannel_audio"] = get_reactor_backchannel_path(reaction)
                # backchannel_audio is used by create_reactor_view to replace the audio track of the reactor trace
                reaction["reactors"], __ = create_reactor_view(reaction)
                if reaction["asides"]:
                    create_asides(reaction)

            create_reaction_concert(extend_by=extend_by)
            free_lock("compilation")

    except KeyboardInterrupt as e:
        free_all_locks()
        raise (e)

    except Exception as e:
        free_all_locks()
        traceback.print_exc()
        print(e)
        if conf.get("break_on_exception"):
            raise (e)

    free_all_locks()
    return failed_reactions


def log_progress(progress):
    key = conf.get("song_key")

    if key not in progress:
        progress[key] = {}

    for i, (channel, reaction) in enumerate(conf.get("reactions").items()):
        if reaction.get("best_path"):
            if reaction.get("ground_truth"):
                overlap = f"{ground_truth_overlap(reaction.get('best_path'), reaction.get('ground_truth')):.1f}%"
            else:
                overlap = "-"

            target_score = reaction.get("target_score", None)
            best_observed_ground_truth = "-"
            best_local_ground_truth = "-"
            if target_score:
                if isinstance(target_score, float):
                    target_score = target_score
                else:
                    if len(target_score) == 2:
                        target_score, best_observed_ground_truth = target_score
                    else:
                        (
                            target_score,
                            best_observed_ground_truth,
                            best_local_ground_truth,
                        ) = target_score

            progress[key][channel] = {
                "best_path": reaction.get("best_path"),
                "best_path_output": reaction.get("best_path_output"),
                "best_path_score": reaction.get("best_path_score"),
                "alignment_duration": reaction.get("alignment_duration"),
                "target_score": target_score,
                "best_observed_ground_truth": best_observed_ground_truth,
                "best_local_ground_truth": best_local_ground_truth,
                "ground_truth": reaction.get("ground_truth", None),
                "ground_truth_overlap": overlap,
            }


def print_progress(progress):
    # for song_key, alignments in progress.items():
    #     for channel, reaction in alignments.items():
    #         if reaction.get('best_path'):
    #             print(f"************* best path for {channel} / {song_key} ****************")
    #             print(reaction.get('best_path_output'))

    x = PrettyTable()
    x.field_names = [
        "Song",
        "Channel",
        "Duration",
        "Score",
        "Best Seen Score",
        "Ground Truth",
        "Local Best Ground Truth",
        "Best Seen Ground Truth",
    ]
    x.align = "r"

    print("****************")
    print(f"Score Summary")

    for song_key, alignments in progress.items():
        for channel, reaction in alignments.items():
            if reaction.get("best_path"):
                x.add_row(
                    [
                        song_key,
                        channel,
                        f"{reaction.get('alignment_duration'):.1f}",
                        f"{reaction.get('best_path_score')[0]:.3f}",
                        reaction.get("target_score", None) or "-",
                        reaction.get("ground_truth_overlap"),
                        f"{reaction.get('best_local_ground_truth')}%",
                        f"{reaction.get('best_observed_ground_truth')}%",
                    ]
                )
            else:
                x.add_row(
                    [
                        song_key,
                        channel,
                        "-",
                        "-",
                        "-",
                        reaction.get("target_score", None) or "-",
                    ]
                )
    print(x)


results_output_dir = "bounded"


def load_songs(lst):
    loaded = []
    for s in lst:
        f = os.path.join(f"library/{s}.json")
        print(f)
        defn = json.load(open(f))
        loaded.append(defn)
    return loaded


import traceback

if __name__ == "__main__":
    from library import (
        songs,
        drafts,
        refresh_manifest,
        finished,
        put_on_ice,
        process_transcripts,
    )

    songs = load_songs(songs)
    drafts = load_songs(drafts)
    refresh_manifest = load_songs(refresh_manifest)
    finished = load_songs(finished)
    put_on_ice = load_songs(put_on_ice)

    process_transcripts = load_songs(process_transcripts)

    progress = {}

    for song in finished:
        clean_up(song)

    for song in put_on_ice:
        clean_up(song, on_ice=True)

    manifest_options = {
        "only_manifest": True,
        "refresh_manifest": False,
        "download_and_parse": True,
    }

    failures = []
    for song in refresh_manifest:
        print(f"Updating manifest for {song.get('song')}")
        failed = create_reaction_compilation(
            song, progress, output_dir=results_output_dir, options=manifest_options
        )
        if len(failed) > 0:
            failures.append((song, failed))
        conf["free_conf"]()

    options = {
        "save_alignment_metadata": True,
        "download_and_parse": False,
        "draft": True,
        "break_on_exception": False,
        "skip_asides": False,
    }

    jobs = [
        ["create_alignment", 6],
        ["output_alignment_video", 1],
        ["isolate_commentary", 1],
        ["create_reactor_view", 1],
        ["create_compilation", 1],
    ]

    failures = []
    for song in drafts:
        failed = create_reaction_compilation(
            song,
            progress,
            output_dir=results_output_dir,
            options=options,
            jobs=jobs,
        )
        if len(failed) > 0:
            failures.append((song, failed))

    options["draft"] = False
    failures = []
    for song in songs:
        failed = create_reaction_compilation(
            song,
            progress,
            output_dir=results_output_dir,
            options=options,
            jobs=jobs,
        )
        if len(failed) > 0:
            failures.append((song, failed))

    for song in process_transcripts:
        make_conf(song, options, results_output_dir)
        conf.get("load_reactions")()

        from transcription.transcribe_reaction import process_transcripts

        process_transcripts()

    print(f"\n\nDone! {len(failures)} songs did not finish")

    for song, failed in failures:
        print(f"\n\n {len(failed)} Failures for song {song}")
        for react_video, e, trace in failed:
            print(f"\n***{react_video} failed with:")
            print(trace)
            print(e)
            print("*****")
