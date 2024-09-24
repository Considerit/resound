import os
from utilities import conversion_audio_sample_rate as sr
from utilities import conf


def consolidate_segments(all_segments):
    intercepts = {}
    seen = {}
    intercept_map = {}
    all_segments = [s for s in all_segments if not s.get("pruned", False)]

    for s in all_segments:
        dup_key = (
            f"{s['end_points'][2]}-{s['end_points'][3]}-{s['end_points'][0]}-{s['end_points'][1]}"
        )
        if dup_key in seen:
            continue
        seen[dup_key] = True

        intercept = s["end_points"][0] - s["end_points"][2]
        key = round(intercept / sr / 2)
        if key not in intercepts:
            intercepts[key] = []
            intercept_map[key] = intercept

        intercepts[key].append(s)

        intercept_map[key] = min(intercept_map[key], intercept)

    filtered_segments = []
    for key, segments in intercepts.items():
        intercept = intercept_map[key]
        if len(segments) == 1:
            filtered_segments.append(segments[0])
            continue

        if len(segments) == 0:
            raise (Exception("No segments!", key))

        max_bs = 0
        min_bs = 999999999999999999999999999999999
        not_subsumed = []
        for i, s in enumerate(segments):
            bs = s["end_points"][2]
            be = s["end_points"][3]
            # print('ABSORB:', intercept, bs/sr, be/sr)

            subsumed_by_other = False
            for j, s2 in enumerate(segments):
                if i == j:
                    continue
                bs2 = s2["end_points"][2]
                be2 = s2["end_points"][3]

                if bs2 <= bs and be2 >= be:
                    subsumed_by_other = bs2 != bs or be2 != be

                    break

            if not subsumed_by_other or s.get("source", None) == "image-alignment":
                not_subsumed.append(s)

        if len(not_subsumed) == 0:
            print(f"ERROR CASE: everything subsumed {intercept} {len(segments)}", segments)

        filtered_segments += not_subsumed

    print(
        f"\t\tconsolidate_segments resulted in {len(filtered_segments)} segments, down from {len(all_segments)}"
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
            if i in merged:
                continue

            for j in range(i + 1, len(by_intercept)):
                if j in merged:
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
        f"\t\tconsolidate_segments resulted in {len(consolidated_segments)} segments, down from {len(all_segments)}"
    )
    return consolidated_segments
