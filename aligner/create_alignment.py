import os
import time

from decimal import Decimal, getcontext

from utilities import universal_frame_rate
from utilities import conf, save_object_to_file, read_object_from_file
from utilities import conversion_audio_sample_rate as sr

from aligner.scoring_and_similarity import (
    path_score,
    print_path,
)
from aligner.create_trimmed_video import trim_and_concat_video
from aligner.align_by_audio import paint_paths
from aligner.align_by_image import build_image_matches


def create_aligned_reaction_video(reaction, extend_by=0, force=False):
    global conf

    output_file = reaction.get("aligned_path")

    if conf.get("create_alignment") or conf.get("alignment_test") or force:
        alignment_metadata_file = reaction.get("alignment_metadata")

        if not os.path.exists(alignment_metadata_file):
            conf["load_reaction"](reaction["channel"])

            # Determine the number of decimal places to try avoiding frame boundary errors given python rounding issues
            fr = Decimal(universal_frame_rate())
            precision = Decimal(1) / fr
            precision_str = str(precision)
            getcontext().prec = len(precision_str.split(".")[-1])

            start = time.perf_counter()

            image_based_segments = None
            image_based_reaction_start = None
            if conf.get("use_image_based_alignment"):
                (
                    image_based_segments,
                    image_based_reaction_start,
                    image_based_reaction_end,
                ) = build_image_matches(reaction)

                if image_based_reaction_start is not None:
                    reaction["start_reaction_search_at"] = reaction.get(
                        "start_reaction_search_at", image_based_reaction_start - sr
                    )

                if image_based_reaction_end is not None:
                    reaction["end_reaction_search_at"] = reaction.get(
                        "end_reaction_search_at", image_based_reaction_end + sr
                    )

            if image_based_reaction_start is None:
                reaction["start_reaction_search_at"] = reaction.get(
                    "start_reaction_search_at", 3 * sr
                )

            best_path = paint_paths(reaction, seed_segments=(image_based_segments or []))

            alignment_duration = (time.perf_counter() - start) / 60  # in minutes
            best_path_score = path_score(best_path, reaction)

            best_path_output = print_path(best_path, reaction).get_string()

            metadata = {
                "best_path_score": best_path_score,
                "best_path": best_path,
                "best_path_output": best_path_output,
                "alignment_duration": alignment_duration,
            }

            if conf.get("save_alignment_metadata"):
                save_object_to_file(alignment_metadata_file, metadata)
        else:
            metadata = read_object_from_file(alignment_metadata_file)

        # Sort of a migration here. If there's a short filler at the beginning, instead merge it with
        # first path (and extend appropriately)
        first_segment = metadata["best_path"][0]
        if len(first_segment) == 6:
            (rs, re, bs, be, filler, __) = first_segment
        else:
            (rs, re, bs, be, filler) = first_segment
        if filler and (re - rs) / sr < 4:
            metadata["best_path"].pop(0)
            metadata["best_path"][0][0] -= re - rs
            metadata["best_path"][0][2] = bs

        reaction.update(metadata)

        if not os.path.exists(output_file) and conf.get("output_alignment_video", False):
            conf["load_reaction"](reaction["channel"])

            react_video = reaction.get("video_path")
            base_video = conf.get("base_video_path")

            reaction_sample_rate = Decimal(sr)
            best_path_converted = [
                (
                    Decimal(s[0]) / reaction_sample_rate,
                    Decimal(s[1]) / reaction_sample_rate,
                    Decimal(s[2]) / reaction_sample_rate,
                    Decimal(s[3]) / reaction_sample_rate,
                    s[4],
                )
                for s in reaction["best_path"]
            ]

            trim_and_concat_video(
                reaction,
                react_video,
                best_path_converted,
                base_video,
                output_file,
                extend_by=extend_by,
                use_fill=conf.get("include_base_video", True),
            )

    return output_file
