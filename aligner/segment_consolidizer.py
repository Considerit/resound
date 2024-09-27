import os
from utilities import conversion_audio_sample_rate as sr
from utilities import conf
from collections import defaultdict


def consolidate_segments(all_segments, neighborhood_size=int(sr / 2 + 1)):
    segments = [s for s in all_segments if not s.get("pruned", False)]

    not_subsumed = []
    to_subsume = []

    def is_neighbor(seg, candidate):
        rs, re, bs, be = seg["end_points"]
        crs, cre, cbs, cbe = candidate["end_points"]
        yint = rs - bs
        candidate_yint = crs - cbs

        return abs(candidate_yint - yint) <= neighborhood_size and max(bs, cbs) <= min(be, cbe)

    for seg in segments:
        intercept = seg["end_points"][0] - seg["end_points"][2]
        if "candidate_intercepts" not in seg:
            seg["candidate_intercepts"] = {st[0] - st[2]: True for st in seg["strokes"]}

        neighbors = [s2 for s2 in segments if seg != s2 and is_neighbor(seg, s2)]

        if len(neighbors) == 0:
            not_subsumed.append(seg)
            continue

        for n in neighbors:
            seg["candidate_intercepts"].update({st[0] - st[2]: True for st in n["strokes"]})

        # check if neighbor should absorb this segment
        bs = seg["end_points"][2]
        be = seg["end_points"][3]

        subsumed_by_neighbor = False
        for j, s2 in enumerate(neighbors):
            if s2 in to_subsume:
                continue

            bs2 = s2["end_points"][2]
            be2 = s2["end_points"][3]

            if bs2 <= bs and be2 >= be:
                subsumed_by_neighbor = bs2 != bs or be2 != be
                break

        if (
            subsumed_by_neighbor
            and seg.get("source", None) == "image-alignment"
            and s2.get("source", None) == "audio-alignment"
        ):
            not_subsumed.append(seg)
            # For image and audio sourced segments that are very closely in agreement,
            # adopt the image sourced one.
            if abs(bs2 - bs) < 3 * sr and abs(be2 - be) < 3 * sr:
                to_subsume.append(s2)
        elif not subsumed_by_neighbor:
            not_subsumed.append(seg)

    filtered_segments = [s for s in not_subsumed if s not in to_subsume]

    if len(filtered_segments) == 0:
        print(f"ERROR CASE: everything subsumed {intercept} {len(segments)}")
        print(len(not_subsumed), len(to_subsume))

        raise Exception()

    print(
        f"\t\tConsolidation resulted in {len(filtered_segments)} segments (from {len(all_segments)})"
    )
    return filtered_segments


def bridge_gaps(all_segments):
    all_segments = [s for s in all_segments if not s.get("pruned", False)]

    merge_thresh = int(sr / 2)

    by_intercept = [(s["end_points"][0] - s["end_points"][2], s) for s in all_segments]
    by_intercept.sort(key=lambda x: x[0])
    merged = {}  # indicies of by_intercept that have already been merged into a different segment

    merge_happened = True
    iteration = 0
    while merge_happened:
        # print(f"Finding merges iteration {iteration}")
        iteration += 1
        merge_happened = False

        for i, (intercept, segment) in enumerate(by_intercept):
            if i in merged or segment.get("source", None) == "image-alignment":
                continue

            for j in range(i + 1, len(by_intercept)):
                if j in merged or by_intercept[j][1].get("source", None) == "image-alignment":
                    continue

                # we're done finding merge candidates if our current intercept is too far out of bounds
                candidate_intercept, candidate_segment = by_intercept[j]
                if candidate_intercept - intercept > merge_thresh:
                    break

                # only subsume if the section to be subsumed is closer to the big segment than its width
                candidate_seg_length = (
                    candidate_segment["end_points"][3] - candidate_segment["end_points"][2]
                )
                dx = min(
                    abs(segment["end_points"][3] - candidate_segment["end_points"][2]),
                    abs(candidate_segment["end_points"][3] - segment["end_points"][2]),
                )
                dy = abs(candidate_intercept - intercept)
                dist_from_segment = (dx**2 + dy**2) ** 0.5
                if dist_from_segment > candidate_seg_length / 2:
                    continue

                # merge!
                merge_happened = True
                seg_length = segment["end_points"][3] - segment["end_points"][2]
                if seg_length > candidate_seg_length:
                    src = candidate_segment
                    dest = segment
                    target_intercept = intercept
                    merged[j] = True
                    # print(f"\tMERGE! {target_intercept / sr}: {j} [{src['end_points'][2]/sr} - {src['end_points'][3]/sr}] => {i} [{dest['end_points'][2]/sr} - {dest['end_points'][3]/sr}]")

                else:
                    src = segment
                    dest = candidate_segment
                    target_intercept = candidate_intercept
                    merged[i] = True
                    # print(f"\tMERGE! {target_intercept / sr}: {i} [{src['end_points'][2]/sr} - {src['end_points'][3]/sr}] => {j} [{dest['end_points'][2]/sr} - {dest['end_points'][3]/sr}]")

                (src_rs, src_re, src_bs, src_be) = src["end_points"]
                (dest_rs, dest_re, dest_bs, dest_be) = dest["end_points"]

                dest["end_points"] = [
                    min(src_bs, dest_bs) + target_intercept,
                    max(src_be, dest_be) + target_intercept,
                    min(src_bs, dest_bs),
                    max(src_be, dest_be),
                ]

                for stroke in src["strokes"]:
                    dest["strokes"].append(stroke)

                if src == segment:
                    break  # we've merged into this one, so stop merging more into it

    consolidated_segments = [s for i, (b, s) in enumerate(by_intercept) if i not in merged]

    print(
        f"\t\tbridge_gaps resulted in {len(consolidated_segments)} segments, down from {len(all_segments)}"
    )
    return consolidated_segments
