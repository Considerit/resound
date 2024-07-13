import numpy as np
import os
import math

from moviepy.editor import ImageClip, VideoFileClip, CompositeVideoClip
from moviepy.editor import concatenate_videoclips

from moviepy.video.VideoClip import VideoClip
from moviepy.audio.AudioClip import AudioClip

from moviepy.video.fx import fadeout

from utilities import (
    extract_audio,
    conf,
    conversion_frame_rate,
    conversion_audio_sample_rate as sr,
)


PAUSE_AFTER_ASIDE = 0.3


def create_asides(reaction):
    from face_finder import create_reactor_view

    reaction["aside_clips"] = {}
    # print(f"Asides: {reaction.get('channel')} has {len(reaction['asides'])}")

    all_asides = reaction["asides"]

    def consolidate_adjacent_asides(all_asides):
        groups = []
        all_asides.sort(key=lambda x: x[0])  # sort by start value of the aside

        # group_when_closer_than = 0.1  # put asides into the same group if one's start value
        #                               # is less than group_when_closer_than greater than the
        #                               # end value of the previous aside

        # Initialize the first group
        current_group = []

        # Group asides
        for aside in all_asides:
            aside = start, end, insertion_point, rewind = get_aside_config(aside)

            # If current_group is empty or the start of the current aside is close enough to the end of the last aside in the group
            if (
                not current_group or insertion_point == current_group[-1][2]
            ):  # < group_when_closer_than:
                current_group.append(aside)
            else:
                # Current aside starts a new group
                groups.append(current_group)
                current_group = [aside]

        # Add the last group if it's not empty
        if current_group:
            groups.append(current_group)

        consolidated_asides = []
        for group in groups:
            start = group[0][0]
            end = group[-1][1]

            aside_conf = {
                "range": (start, end),
                "insertion_point": group[-1][2],
                "rewind": group[-1][3],
                "keep_segments": [(g[0] - start, g[1] - start) for g in group],
            }
            consolidated_asides.append(aside_conf)

        return consolidated_asides

    def get_aside_config(aside):
        if len(aside) == 4:
            start, end, insertion_point, rewind = aside
        else:
            start, end, insertion_point = aside
            rewind = 3

        return start, end, insertion_point, rewind

    consolidated_asides = consolidate_adjacent_asides(all_asides)

    for i, aside in enumerate(consolidated_asides):
        insertion_point = aside["insertion_point"]
        rewind = aside["rewind"]
        keep_segments = aside["keep_segments"]
        keep_segments.sort(key=lambda x: x[0])

        # first, isolate the correct part of the reaction video
        aside_video_clip = os.path.join(
            conf.get("temp_directory"), f"{reaction.get('channel')}-aside-{i}.mp4"
        )

        if not os.path.exists(aside_video_clip):
            react_video = VideoFileClip(reaction.get("video_path"), has_mask=True)
            start = aside["range"][0]
            subclips = [
                react_video.subclip(start + aside[0], start + aside[1])
                for aside in keep_segments
            ]
            aside_clip = concatenate_videoclips(subclips).set_fps(30)

            # aside_clip.write_videofile(aside_video_clip, codec="h264_videotoolbox", audio_codec="aac", ffmpeg_params=['-q:v', '60'])

            aside_clip.write_videofile(
                aside_video_clip,
                codec="libx264",
                ffmpeg_params=["-crf", "18", "-preset", "slow"],
                audio_codec="aac",
            )

            react_video.close()
            aside_clip.close()

        # do face detection on it
        aside_reactor_views, __ = create_reactor_view(
            reaction, aside_video=aside_video_clip, show_facial_recognition=False
        )

        for reactor_view in aside_reactor_views:
            reactor_view["duration"] = sum((b - a for (a, b) in keep_segments))
            reactor_view["aside-num"] = i

        reaction["aside_clips"][insertion_point] = (aside_reactor_views, rewind)


def create_full_video_from_spec(spec, name, key2video):
    video_segments = []

    for segment in spec:
        still_frame = segment.get("still-frame", None)

        clip = key2video[segment["key"]]
        main_duration = clip.duration

        if still_frame is not None:
            duration = segment["duration"]
            frame = clip.get_frame(still_frame)
            clip = ImageClip(frame, duration=duration)

        else:
            start = segment["start"]

            end = segment["end"]
            if end == "*":
                end = main_duration

            if end <= start:
                continue

            clip = clip.subclip(start, end)

            if segment.get("rewind_clip", False):
                print("rewind clip", segment)
                rewind_icon = (
                    ImageClip(os.path.join("compositor", "rewind.png"))
                    .set_duration(1)
                    .resize((100, 100))
                )
                rewind_icon = rewind_icon.fadeout(0.5)
                rewind_icon = rewind_icon.set_position(("center", "center"))
                duration = segment["duration"]

                composite_rewind_clip = CompositeVideoClip(
                    [clip, rewind_icon]
                )  # results in video with just black background
                f = os.path.join(
                    conf.get("temp_directory"),
                    f"COMPOSITE_REWIND_CLIP-{start}-{end}-{duration}.mp4",
                )
                if not os.path.exists(f):
                    composite_rewind_clip.write_videofile(
                        f, codec="h264_videotoolbox", ffmpeg_params=["-q:v", "40"]
                    )
                clip = VideoFileClip(f).resize(clip.size)

        video_segments.append(clip)

    full_video = concatenate_videoclips(video_segments)
    return full_video


def adjust_audible_segments(
    audible_segments, name, split_point, duration, audio_factor
):
    new_segments = []

    if duration == 0:
        return

    for segment in audible_segments[name]:
        start, end, current_audio_factor = segment
        if end < split_point:
            # print("\t\t\t\t", f"BEFORE! start (s): {start/sr}, duration (s): {duration/sr}")

            new_segments.append(segment)
        elif start >= split_point:
            # print("\t\t\t\t", f"start (s): {start/sr}, duration (s): {duration/sr}, new start (s): {(start+duration)/sr}, new start (sr): {start+duration}")
            new_segments.append(
                [start + duration, end + duration, current_audio_factor]
            )
        elif start < split_point and end > split_point:
            # print("\t\t\t\t", f"SPLITTING! start (s): {start/sr}, duration (s): {duration/sr}")

            if split_point - start > 0:
                before = [start, split_point, current_audio_factor]
                new_segments.append(before)
                # print("\t\t\t\t\t", f"BEFORE... start (s): {start/sr} - SPLIT (s): {split_point / sr}, duration (s): {(split_point - start)/sr}")

            if end - split_point > 0:
                after = [split_point + duration, end + duration, current_audio_factor]
                new_segments.append(after)
                # print("\t\t\t\t\t", f"AFTER... start (s): {(split_point + duration)/sr} - SPLIT (s): {(end + duration) / sr}, duration (s): {(end-split_point)/sr}")

    if audio_factor != 0:
        insertion = [split_point, split_point + duration, 1]
        new_segments.append(insertion)

    new_segments.sort(key=lambda x: x[0])
    audible_segments[name] = new_segments


def create_full_audio_from_spec(spec, name, key2audio, audible_segments):
    # print("*************")
    # print(f"CREATING FULL AUDIO FOR {name}")

    main_audio = key2audio["main"]

    is_reaction = name in audible_segments

    audio_segments = []

    # if is_reaction:
    #     print(f"BEFORE AUDIBLE SEGMENTS FOR {name}")
    #     for segment in audible_segments[name]:
    #         print(f"\t  start={segment[0] / sr}  end={segment[1] / sr}  factor={segment[2]}")

    #     for segment in spec:
    #         print(segment)

    for segment in reversed(spec):
        split_point = int(segment.get("split_point", segment.get("start", 0)) * sr)

        still_frame = segment.get("still-frame", None)
        if still_frame is not None:
            duration = int(sr * segment["duration"])
            audio = np.zeros((duration, main_audio.shape[1]))
            assert duration > 0
            if is_reaction:
                # print(f"STILL FRAME FOR {segment['duration']} at {split_point / sr}")
                adjust_audible_segments(
                    audible_segments=audible_segments,
                    name=name,
                    split_point=split_point,
                    duration=duration,
                    audio_factor=0,
                )

        else:
            audio_src = key2audio[segment["key"]]

            start = int(sr * segment["start"])

            end = segment["end"]
            if end == "*":
                end = audio_src.shape[0]
            else:
                end = int(sr * end)

            if end <= start:
                continue

            audio = audio_src[start:end, :]

            if is_reaction:
                duration = int(sr * segment.get("duration", 0))
                assert (duration > 0, end, start)

                if segment["key"] == "main":
                    # scaling_factors.append(  audio_scaling_factors[name][start:end]   )

                    # print(f"MAIN FOR {duration / sr} at {split_point / sr}")

                    adjust_audible_segments(
                        audible_segments, name, split_point, duration, 0
                    )

                elif segment["key"].startswith("aside"):  # from an aside
                    # print(f"ASIDE FOR {duration / sr} at {split_point / sr}")
                    adjust_audible_segments(
                        audible_segments, name, split_point, duration, 1
                    )

                else:
                    raise Exception("Unsupported key", segment)

        audio_segments.append(audio)

    audio_segments.reverse()
    full_audio = np.concatenate(audio_segments)

    # if is_reaction:
    #     print(f"AUDIBLE SEGMENTS FOR {name}")
    #     for s in audible_segments[name]:
    #         print("\t", f"{s[0] / sr} - {s[1] / sr}", s)
    return full_audio


def create_full_media_specification_with_asides(base_video_duration):
    full_spec = {
        "base": [{"key": "main", "start": 0, "end": "*"}],
    }

    all_asides = []
    for name, reaction in conf.get("reactions").items():
        full_spec[name] = [{"key": "main", "start": 0, "end": "*"}]

        if reaction.get("aside_clips", None):
            for insertion_point, (aside_clips, rewind) in reaction.get(
                "aside_clips"
            ).items():
                all_asides.append(
                    [insertion_point, aside_clips, reaction.get("channel"), rewind]
                )

    if len(all_asides) == 0:
        return full_spec

    all_asides.sort(key=lambda x: x[0], reverse=True)

    def split_at(
        name,
        split_point,
        duration,
        insert="still-frame",
        pause_after=None,
        start=None,
        end=None,
    ):
        spec = full_spec[name]
        new_spec = []
        active = spec.pop(0)

        if pause_after is None:
            pause_after = PAUSE_AFTER_ASIDE

        # Splice the aside into the current reaction clip
        before = active.copy()
        before["end"] = split_point
        after = active.copy()
        after["start"] = split_point

        if before["end"] - before["start"] > 0:
            new_spec.append(before)

        if insert == "still-frame":
            duration += pause_after

            assert duration is not None and duration > 0
            splicing_in = {
                "key": "main",
                "still-frame": max(0, split_point - 0.1),
                "duration": duration,
                "split_point": split_point,
            }
            new_spec.append(splicing_in)

        elif insert == "rewind_clip":
            duration += pause_after

            assert start is not None and end is not None
            if end == "*" or end - start > 0:
                splicing_in = {
                    "key": "main",
                    "rewind_clip": True,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "split_point": split_point,
                }
                new_spec.append(splicing_in)

        else:
            assert start is not None and end is not None

            if end == "*" or end - start > 0:
                splicing_in = {
                    "key": insert,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "split_point": split_point,
                }
                new_spec.append(splicing_in)

            if pause_after > 0:
                short_pause = {
                    "key": insert,
                    "still-frame": 0,
                    "duration": pause_after,
                    "split_point": split_point,
                }
                new_spec.append(short_pause)

        if (
            "end" not in after
            or after["end"] == "*"
            or "before" not in after
            or after["end"] - after["before"] > 0
        ):
            new_spec.append(after)

        full_spec[name] = new_spec + spec

    for i, (insertion_point, aside_clips, channel, rewind) in enumerate(all_asides):
        duration = aside_clips[0]["duration"]

        if duration <= 0:
            continue

        print(f"\tAside at {insertion_point} of {duration} seconds for {channel}")

        if not rewind or base_video_duration < insertion_point:
            rewind = 0

        rewind = min(insertion_point, rewind)

        if rewind > 0:
            split_at(
                name="base",
                split_point=insertion_point,
                duration=rewind,
                insert="rewind_clip",
                pause_after=0,
                start=insertion_point - rewind,
                end=insertion_point,
            )

        split_at(name="base", split_point=insertion_point, duration=duration)

        for name, reaction in conf.get("reactions").items():
            if rewind > 0:
                split_at(
                    name=name,
                    split_point=insertion_point,
                    duration=rewind,
                    insert="main",
                    pause_after=0,
                    start=insertion_point - rewind,
                    end=insertion_point,
                )

            if channel == name:
                aside_id = aside_clips[0]["aside-num"]
                split_at(
                    name=name,
                    split_point=insertion_point,
                    insert=f"aside-{aside_id}",
                    start=0,
                    end="*",
                    duration=duration,
                )

            else:
                split_at(name=name, split_point=insertion_point, duration=duration)

    # print("FULL SPEC")
    # for k,v in full_spec.items():
    #     print( f"\t{k}" )
    #     t = 0
    #     for i, vv in enumerate(v):
    #         duration = vv.get('duration', None)
    #         if duration is None:
    #             if vv['end'] == '*':
    #                 duration = 0
    #             else:
    #                 duration = vv['end'] - vv['start']

    #         if duration is None:
    #             print(vv)

    #         print( f"\t\t |{t} - {t + duration}| {vv.get('key')}: {vv.get('still-frame', '')} split_point = {vv.get('split_point', '')} duration = {duration}")

    #         t += duration

    return full_spec


# Any of the reaction clips can have any number of "asides". An aside is a bonus
# video clip spliced into a specific point in the respective reaction video clip.
# When an aside is active, only the respective video clip is playing, and all the
# other clips are paused. When a clip is paused because an aside is playing, the
# previous frame is replicated until the aside is finished, and no audio is played.


def incorporate_asides_video(base_video, video_background):
    full_spec = create_full_media_specification_with_asides(base_video.duration)
    base_video = create_full_video_from_spec(
        full_spec["base"], "base", {"main": base_video}
    )

    if video_background is not None:
        video_background = create_full_video_from_spec(
            full_spec["base"], "base", {"main": video_background}
        )

    for name, reaction in conf.get("reactions").items():
        # print(f"\t\tCreating video clip for {name}")
        reactors = reaction.get("reactors")
        print(f"Making video for {name}")

        for idx, reactor in enumerate(reactors):
            reactor_clip = VideoFileClip(reactor["path"])

            videos = {"main": reactor_clip}

            for insertion_point, (views, rewind) in reaction.get(
                "aside_clips", {}
            ).items():
                view = views[idx]
                # for view in views:
                videos[f"aside-{view['aside-num']}"] = VideoFileClip(
                    view["path"]
                ).without_audio()

            reactor["clip"] = create_full_video_from_spec(full_spec[name], name, videos)

    print("Done incorporating asides video")
    return base_video, video_background


from compositor.mix_audio import adjust_gain_for_loudness_match


def incorporate_asides_audio(base_video, base_audio_clip, audible_segments):
    full_spec = create_full_media_specification_with_asides(base_video.duration)

    base_audio_clip = create_full_audio_from_spec(
        full_spec["base"], "base", {"main": base_audio_clip}, audible_segments
    )

    print("CREATING ASIDE AUDIO CLIPS")
    for name, reaction in conf.get("reactions").items():
        print(f"\t\tCreating audio clip for {name}")
        reactors = reaction.get("reactors")

        audio_files = {"main": reaction["mixed_audio"]}

        for insertion_point, (views, rewind) in reaction.get("aside_clips", {}).items():
            for view in views:
                aside_audio, __, __ = extract_audio(
                    view["path"],
                    convert_to_mono=False,
                    keep_file=False,
                    preserve_silence=True,
                )
                aside_audio = adjust_gain_for_loudness_match(aside_audio, name)
                audio_files[f"aside-{view['aside-num']}"] = aside_audio

        reaction["mixed_audio"] = create_full_audio_from_spec(
            full_spec[name], name, audio_files, audible_segments
        )

    return base_audio_clip
