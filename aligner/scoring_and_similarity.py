import numpy as np
import librosa
import math

from utilities import conversion_audio_sample_rate as sr
from utilities import conf

from prettytable import PrettyTable

from silence import get_quiet_parts_of_song


################
# Scoring paths


def path_score(path, reaction, segments_by_key, end=None, start=0):
    global path_score_cache
    global path_score_cache_perf

    song_length = conf.get("song_length")
    song_audio_mfcc = conf.get("song_audio_mfcc")
    hop_length = conf.get("hop_length")
    reaction_audio = reaction["reaction_audio_data"]
    reaction_audio_mfcc = reaction["reaction_audio_mfcc"]

    key = f"{song_length} {len(reaction_audio)} {str(path)} {start} {end}"

    if key in path_score_cache:
        path_score_cache_perf["hits"] += 1
        return path_score_cache[key]

    elif "misses" in path_score_cache_perf:
        path_score_cache_perf["misses"] += 1

    # print(f"path score cache hits/misses = {path_score_cache_perf['hits']} / {path_score_cache_perf['misses']}")

    if end is None:
        end = song_length

    duration = 0
    fill = 0

    temporal_center = 0
    total_length = 0

    segment_penalty = 1

    quiet_parts = get_quiet_parts_of_song()

    img_segs = 0

    time_guided_by_image_alignment_during_quiet_parts = 0
    for segment in path:
        (reaction_start, reaction_end, current_start, current_end, is_filler, key) = segment[:6]

        if reaction_start < 0:
            reaction_end += -1 * reaction_start
            reaction_start = 0
        if current_start < 0:
            current_end += -1 * current_start
            current_start = 0

        segment_weight = (current_end - current_start) / (end - start)
        segment_time_center = reaction_start + (reaction_end - reaction_start) / 2
        temporal_center += segment_weight * segment_time_center

        if not is_filler:
            duration += reaction_end - reaction_start
            total_length += round((reaction_end - reaction_start) / hop_length)

        else:
            fill += current_end - current_start
            total_length += round((current_end - current_start) / hop_length)

        if current_end - current_start < sr and not is_filler:
            segment_penalty *= 0.98

        if key is not None:
            full_segment = segments_by_key[key]
            if full_segment.get("source") == "image-alignment":
                img_segs += 1
                seg_key = str((current_start, current_end))
                if "during_quiet" not in full_segment:
                    full_segment["during_quiet"] = {}

                if seg_key not in full_segment["during_quiet"]:
                    quiet_time = 0

                    for qs, qe in quiet_parts:
                        qs *= sr
                        qe *= sr

                        if max(current_start, qs) <= min(current_end, qe):
                            overlap = min(current_end, qe) - max(current_start, qs)
                            quiet_time += overlap

                    full_segment["during_quiet"][seg_key] = quiet_time / sr

                quiet_time = full_segment["during_quiet"][seg_key]

                time_guided_by_image_alignment_during_quiet_parts += quiet_time

    if time_guided_by_image_alignment_during_quiet_parts > 0:
        segment_penalty *= 1 + 0.001 * time_guided_by_image_alignment_during_quiet_parts

        # print(
        #     f"\tYO! Time guided by image alignment = {time_guided_by_image_alignment_during_quiet_parts}s giving bonus of {1 + 0.001 * time_guided_by_image_alignment_during_quiet_parts}x"
        # )

    # Derivation for below:
    #   earliness = |R| / temporal_center
    #   best possible earliness =  |R| / (|B| / 2) (when the first part of the reaction is matched with the full base audio)
    #   normalized earliness = earliness / best_possible_earliness = |B| / (|R| * temporal_center)
    #   ...but I'm going to change the normalization to match being in the middle of the reaction, because sometimes
    #      the reactions are really really long.
    #   middle_earliness = |R| / ( |R| / 2  ) = 2
    #   normalized earliness = earliness / middle_earliness
    # normalized_earliness_score = song_length / (len(reaction_audio) * temporal_center)
    # fill_score = 1 / (1 + abs(duration - song_length) / sr)

    # earliness = len(reaction_audio) / temporal_center
    # earliness = math.log(1 + earliness)

    earliness = song_length / temporal_center

    fill_score = duration / (duration + fill)

    # mfcc_alignment = path_score_by_mfcc_mse_similarity(path, reaction)
    mfcc_alignment = 0
    # cosine_mfcc_alignment = path_score_by_mfcc_cosine_similarity(path, reaction, segments_by_key)

    # alignment = 100 * cosine_mfcc_alignment

    alignment = 100 * path_alignment_score(path, reaction, segments_by_key)

    duration_score = (duration + fill) / (end - start)

    total_score = (
        segment_penalty * duration_score * duration_score * fill_score * earliness * alignment
    )
    if duration == 0:
        total_score = alignment = 0

    by_dict = {
        "total_score": total_score,
        "earliness": earliness,
        "alignment": alignment,
        "fill_score": fill_score,
        "mfcc_alignment": mfcc_alignment,
        "image_segments": img_segs,
        "duration_score": duration_score,
        "segment_penalty": segment_penalty,
    }

    path_score_cache[key] = [
        total_score,
        earliness,
        alignment,
        fill_score,
        mfcc_alignment,
        duration_score,
        segment_penalty,
        by_dict,
    ]
    return path_score_cache[key]


def path_alignment_score(path, reaction, segments_by_key):
    alignment_score = 0

    total_duration = 0
    for sequence in path:
        total_duration += sequence[1] - sequence[0]

    for sequence in path:
        is_fill = len(sequence) > 4 and sequence[4]
        if not is_fill:
            mfcc_score = get_segment_mfcc_cosine_similarity_score(reaction, sequence)
            duration_factor = (sequence[1] - sequence[0]) / total_duration
            image_score = get_image_score_for_segment(reaction, sequence)

            alignment_score += mfcc_score * mfcc_score * image_score * duration_factor

    return alignment_score


# def path_score_by_mfcc_cosine_similarity(path, reaction, segments_by_key):
#     mfcc_sequence_sum_score = 0

#     total_duration = 0
#     for sequence in path:
#         total_duration += sequence[1] - sequence[0]

#     min_sequence_score = 1
#     penalty = 1

#     for sequence in path:
#         is_fill = len(sequence) > 4 and sequence[4]
#         if not is_fill:
#             mfcc_score = get_segment_mfcc_cosine_similarity_score(reaction, sequence)
#             duration_factor = (sequence[1] - sequence[0]) / total_duration
#             mfcc_sequence_sum_score += mfcc_score * mfcc_score * duration_factor

#     return mfcc_sequence_sum_score * penalty


# def path_score_by_raw_cosine_similarity(path, reaction):
#     sequence_sum_score = 0

#     total_duration = 0
#     for sequence in path:
#         reaction_start, reaction_end, current_start, current_end, is_filler = sequence[:5]
#         total_duration += reaction_end - reaction_start

#     for sequence in path:
#         reaction_start, reaction_end, current_start, current_end, is_filler = sequence[:5]

#         if not is_filler:
#             score = get_segment_raw_cosine_similarity_score(reaction, sequence)
#             duration_factor = (current_end - current_start) / total_duration
#             sequence_sum_score += score * duration_factor

#     return sequence_sum_score


# def path_score_by_mfcc_mse_similarity(path, reaction):
#     total_duration = 0
#     for sequence in path:
#         reaction_start, reaction_end, current_start, current_end, is_filler = sequence[:5]
#         total_duration += reaction_end - reaction_start

#     mse_sum = 0
#     for sequence in path:
#         reaction_start, reaction_end, current_start, current_end, is_filler = sequence[:5]
#         duration = reaction_end - reaction_start

#         mse = get_segment_mfcc_mse(reaction, sequence)

#         if math.isnan(mse):
#             mse = 0

#         mse_sum += mse * duration / total_duration

#     similarity = conf.get("n_mfcc") / (1 + mse_sum)

#     return similarity


def get_image_score_for_segment(reaction, segment):
    if "image_alignment_matrix" not in reaction:
        return 1

    image_scores, music_times, reaction_times = reaction["image_alignment_matrix"]

    if type(segment) == dict:
        rt, re, mt, me = segment["end_points"][:4]
    else:
        rt, re, mt, me = segment[:4]

    sample_rate = music_times[1] - music_times[0]
    initial_mt = music_times[0]
    initial_rt = reaction_times[0]

    # Calculate mt_adjustment and rt_adjustment to find the nearest sampled points
    mt_adjustment = round((mt - initial_mt) / sample_rate) * sample_rate - (mt - initial_mt)
    rt_adjustment = round((rt - initial_rt) / sample_rate) * sample_rate - (rt - initial_rt)

    mt += mt_adjustment
    rt += rt_adjustment

    # key = str((mt, rt))
    # assert key in image_scores

    # key = str((mt + sample_rate, rt + sample_rate))
    # assert key in image_scores

    my_scores = []

    not_in = present = 0
    while mt <= me:
        key = str((mt, rt))
        if key in image_scores:
            my_scores.append(image_scores[key][0])
            present += 1
        else:
            not_in += 1
        mt += sample_rate
        rt += sample_rate

    if not_in > 0:
        print(f"\tFound {present} of {not_in + present} pairs")

    mean = np.mean(np.array(my_scores))
    return mean


def get_segment_score(reaction, segment, use_image_scores=True):
    # mse_score = get_segment_mfcc_mse_score(reaction, segment)
    # raw_cosine_score = get_segment_raw_cosine_similarity_score(reaction, segment)
    cosine_score = get_segment_mfcc_cosine_similarity_score(reaction, segment)
    if use_image_scores:
        image_score = get_image_score_for_segment(reaction, segment)
    else:
        image_score = 1

    alignment = image_score * cosine_score  # * mse_score # * raw_cosine_score
    return alignment


def find_best_path(reaction, candidate_paths, segments_by_key=None):
    print(f"Finding the best of {len(candidate_paths)} paths")

    gt = reaction.get("ground_truth")

    assert len(candidate_paths) > 0

    best_score = 0
    best_early_completion_score = 0
    best_similarity = 0
    best_duration = 0
    most_image_segments = 1

    if gt:
        max_gt = 0
        best_gt_path = None
        best_scores = None

    paths_with_scores = []
    for path in candidate_paths:
        scores = path_score(path, reaction, segments_by_key=segments_by_key)

        if scores[0] > best_score:
            best_score = scores[0]
        if scores[1] > best_early_completion_score:
            best_early_completion_score = scores[1]
        if scores[2] > best_similarity:
            best_similarity = scores[2]
        if scores[3] > best_duration:
            best_duration = scores[3]
        if scores[-1]["image_segments"] > most_image_segments:
            most_image_segments = scores[-1]["image_segments"]

        if gt:
            gtpp = ground_truth_overlap(path, gt)
            if gtpp > max_gt:
                max_gt = gtpp
                best_gt_path = path
                best_scores = scores

        paths_with_scores.append([path, scores])

    print(f"\tDone scoring, now processing paths")

    normalized_scores = []
    for path, scores in paths_with_scores:
        if (
            scores[0] > 0.9 * best_score
            or best_early_completion_score == scores[1]
            or best_similarity == scores[2]
            or best_duration == scores[3]
            or most_image_segments == scores[-1]["image_segments"]
        ):  # winnow it down a bit
            total_score = scores[0] / best_score
            completion_score = scores[1] / best_early_completion_score
            similarity_score = scores[2] / best_similarity
            fill_score = scores[3] / best_duration
            image_score = scores[-1]["image_segments"] / most_image_segments

            normalized_scores.append(
                (
                    path,
                    (
                        total_score,
                        completion_score,
                        similarity_score,
                        fill_score,
                        image_score,
                        scores[-1],
                    ),
                )
            )

    normalized_scores.sort(key=lambda x: x[1][0], reverse=True)

    print("Paths by score:")
    for idx, (path, scores) in enumerate(normalized_scores[:20]):
        if gt:
            gtpp = ground_truth_overlap(path, gt)
            gtp = f"Ground Truth: {gtpp}%"
        else:
            gtp = ""
        print(
            f"\tScore={scores[0]}  EarlyThrough={scores[1]}  Similarity={scores[2]} Duration={scores[3]} Img={scores[4]} Mult={scores[-1]['segment_penalty']} {gtp}"
        )
        print_path(path, reaction, segments_by_key)

    if gt:
        print("***** Best Ground Truth Path *****")

        print(
            f"\tScore={best_scores[0]}  EarlyThrough={best_scores[1]}  Similarity={best_scores[2]} Duration={best_scores[3]} {max_gt}"
        )
        print_path(best_gt_path, reaction, segments_by_key)

    return normalized_scores[0][0]


##################
#  Similarity functions
mfcc_weights = None
use_mfcc_weights = True
mfcc_cosine_weights = None


def mse_mfcc_similarity(mfcc1=None, mfcc2=None, verbose=False, mse_only=False):
    # Compute MFCCs for each audio chunk
    if mfcc1 is None:
        mfcc1 = librosa.feature.mfcc(y=audio_chunk1, sr=sr, n_mfcc=20)

    if mfcc2 is None:
        mfcc2 = librosa.feature.mfcc(y=audio_chunk2, sr=sr, n_mfcc=20)

    # Make sure the MFCCs are the same shape
    len1 = mfcc1.shape[1]
    len2 = mfcc2.shape[1]

    if len1 != len2:
        if len2 > len1:
            mfcc2 = mfcc2[:, :len1]
        else:
            mfcc1 = mfcc1[:, :len2]
    len1 = len2 = min(len1, len2)

    squared_errors = (mfcc1 - mfcc2) ** 2

    if use_mfcc_weights:
        global mfcc_weights
        n_mfcc = conf.get("n_mfcc")
        alpha = 0.9  # decay factor
        if mfcc_weights is None:
            mfcc_weights = np.array([[alpha**i] for i in range(n_mfcc)])

        mse = np.mean(mfcc_weights * squared_errors)
    else:
        # Compute mean squared error between MFCCs
        mse = np.mean(squared_errors)

    if mse_only:
        return mse
    else:
        similarity = conf.get("n_mfcc") / (1 + mse)
        return similarity


def mfcc_cosine_similarity(mfcc1=None, mfcc2=None, verbose=False):
    # Make sure the MFCCs are the same shape
    len1 = mfcc1.shape[1]
    len2 = mfcc2.shape[1]

    if len1 != len2:
        if len2 > len1:
            mfcc2 = mfcc2[:, :len1]
        else:
            mfcc1 = mfcc1[:, :len2]

    if len1 < 1 or len2 < 1:
        return 0

    # Compute cosine similarity for each MFCC coefficient dimension
    similarities = [cosine_similarity(mfcc1[i, :], mfcc2[i, :]) for i in range(mfcc1.shape[0])]

    if use_mfcc_weights:
        global mfcc_cosine_weights
        if mfcc_cosine_weights is None or len(mfcc_cosine_weights) != len(similarities):
            alpha = 0.9  # decay factor
            mfcc_cosine_weights = np.array([alpha**i for i in range(mfcc1.shape[0])])

        # Weighted average of similarities using mfcc_cosine_weights
        weighted_similarity = np.dot(similarities, mfcc_cosine_weights) / np.sum(
            mfcc_cosine_weights
        )
    else:
        raise Exception("Not implemented")

    # print(weighted_similarity)
    try:
        return weighted_similarity[0]
    except:
        return weighted_similarity


def raw_cosine_similarity(audio_chunk1=None, audio_chunk2=None, verbose=False):
    # Make sure the MFCCs are the same shape
    len1 = len(audio_chunk1)
    len2 = len(audio_chunk2)

    if len1 != len2:
        if len2 > len1:
            audio_chunk2 = audio_chunk2[:len1]
        else:
            audio_chunk1 = audio_chunk1[:len2]

    if len1 < 1 or len2 < 1:
        return 0

    # Compute cosine similarity for each MFCC coefficient dimension
    similarity = cosine_similarity(audio_chunk1, audio_chunk2)
    if similarity < 0 or math.isnan(similarity):
        similarity = 0
    return similarity


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)


################
# Path to signal


# def create_reaction_mfcc_from_path(path, reaction):
#     # these mfcc variables are the result of calls to librosa.feature.mfcc, thus
#     # they have a shape of (num_mfcc, length of audio track). The length of the
#     # reaction_audio_mfcc track is greater than the length of song_audio_mfcc track.
#     reaction_audio_mfcc = reaction.get("reaction_audio_mfcc")
#     song_audio_mfcc = conf.get("song_audio_mfcc")
#     hop_length = conf.get("hop_length")

#     total_length = 0
#     for reaction_start, reaction_end, current_start, current_end, is_filler in path:
#         if not is_filler:
#             reaction_start = round(reaction_start / hop_length)
#             reaction_end = round(reaction_end / hop_length)
#             total_length += reaction_end - reaction_start
#         else:
#             current_start = round(current_start / hop_length)
#             current_end = round(current_end / hop_length)
#             total_length += current_end - current_start

#     # the resulting combined mfcc (path_mfcc) should have the same number of mfccs
#     # as the input mfccs, and be the length of the reaction_audio_for_path.
#     path_mfcc = np.zeros((song_audio_mfcc.shape[0], total_length))

#     # Now we're going to fill up the path_mfcc incrementally, taking from either
#     # song_audio_mfcc or reaction_audio_mfcc
#     start = 0
#     for reaction_start, reaction_end, current_start, current_end, is_filler in path:
#         if not is_filler:
#             reaction_start = round(reaction_start / hop_length)
#             reaction_end = round(reaction_end / hop_length)

#             length = reaction_end - reaction_start
#             segment = reaction_audio_mfcc[:, reaction_start:reaction_end]
#         else:
#             length = math.floor((current_end - current_start) / hop_length)

#         if length > 0:
#             if not is_filler:
#                 path_mfcc[:, start : start + length] = segment
#             start += length

#     return path_mfcc


###########
# Printing path


def print_path(path, reaction, segments_by_key=None, ignore_score=False):
    gt = reaction.get("ground_truth")
    print("\t\t****************")

    x = PrettyTable()
    x.border = False
    x.align = "r"
    x.field_names = [
        "\t\t",
        "",
        "Base",
        "Reaction",
        "y-int",
        "mfcc cosine",
        "raw cosine",
        "image",
        "source",
    ]

    if gt:
        x.field_names.append("ground truth")
        print(f"\t\tGround Truth Overlap {ground_truth_overlap(path, gt):.1f}%")

    for sequence in path:
        (reaction_start, reaction_end, current_start, current_end, is_filler, key) = sequence[:6]

        if key is None:
            src = "F"
        else:
            full_seg = segments_by_key[key]
            if full_seg["source"] == "image-alignment":
                src = "I"
            else:
                src = "A"

        gt_pr = "-"
        mfcc_cosine_score = raw_cosine_score = image_score = 0

        if not is_filler:
            if not ignore_score:
                mfcc_cosine_score = 1000 * get_segment_mfcc_cosine_similarity_score(
                    reaction, sequence
                )
                raw_cosine_score = 1000 * get_segment_raw_cosine_similarity_score(
                    reaction, sequence
                )
                image_score = 100 * get_image_score_for_segment(reaction, sequence)

                if math.isnan(mfcc_cosine_score):
                    mfcc_cosine_score = -1
                if math.isnan(raw_cosine_score):
                    raw_cosine_score = -1
            if gt:
                total_overlap = 0
                for gt_sequence in gt:
                    total_overlap += calculate_overlap(sequence, gt_sequence)

                gt_pr = f"{100 * total_overlap / (sequence[1] - sequence[0]):.1f}%"

        row = [
            "\t\t",
            "x" if is_filler else "",
            f"{float(current_start)/sr:.3f}-{float(current_end)/sr:.3f}",
            f"{float(reaction_start)/sr:.3f}-{float(reaction_end)/sr:.3f}",
            f"{float(reaction_start - current_start)/sr:.3f}",
            f"{round(mfcc_cosine_score)}",
            f"{round(raw_cosine_score)}",
            f"{round(image_score)}",
            src,
        ]
        if gt:
            row.append(gt_pr)

        x.add_row(row)

    print(x)

    return x


##############
## Initialization

path_score_cache = {}
path_score_cache_perf = {}


def initialize_path_score():
    global path_score_cache
    global path_score_cache_perf

    path_score_cache.clear()
    path_score_cache_perf["hits"] = 0
    path_score_cache_perf["misses"] = 0


#############
##### Segments

segment_raw_cosine_scores = {}
segment_mfcc_cosine_scores = {}
segment_mfcc_mses = {}


def initialize_segment_tracking():
    global segment_mfcc_mses, segment_mfcc_cosine_scores, segment_raw_cosine_scores
    segment_mfcc_mses.clear()
    segment_mfcc_cosine_scores.clear()
    segment_raw_cosine_scores.clear()


def get_path_id(path):
    return ":".join([get_segment_id(segment) for segment in path])


def get_segment_id(segment):
    if len(segment) == 4:
        reaction_start, reaction_end, current_start, current_end = segment
        is_filler = False
    else:
        reaction_start, reaction_end, current_start, current_end, is_filler = segment[:5]

    return f"{reaction_start} {reaction_end} {current_start} {current_end} {is_filler}"


# def get_segment_mfcc_mse_score(reaction, segment):
#     if segment[4]:
#         return 0

#     mse = get_segment_mfcc_mse(reaction, segment)
#     similarity = conf.get("n_mfcc") / (1 + mse)
#     return similarity


# def get_segment_mfcc_mse(reaction, segment):
#     reaction_start, reaction_end, current_start, current_end, is_filler = segment[:5]

#     global segment_mfcc_mses

#     key = get_segment_id(segment)
#     if key not in segment_mfcc_mses:
#         song_audio_mfcc = conf.get("song_audio_mfcc")
#         reaction_audio_mfcc = reaction.get("reaction_audio_mfcc")
#         hop_length = conf.get("hop_length")

#         if is_filler:
#             mfcc_react_chunk = np.zeros((song_audio_mfcc.shape[0], reaction_end - reaction_start))
#         else:
#             mfcc_react_chunk = reaction_audio_mfcc[
#                 :, round(reaction_start / hop_length) : round(reaction_end / hop_length)
#             ]
#         mfcc_song_chunk = song_audio_mfcc[
#             :, round(current_start / hop_length) : round(current_end / hop_length)
#         ]
#         mfcc_score = mse_mfcc_similarity(
#             mfcc1=mfcc_song_chunk, mfcc2=mfcc_react_chunk, mse_only=True
#         )

#         segment_mfcc_mses[key] = mfcc_score

#     return segment_mfcc_mses[key]


def get_segment_mfcc_cosine_similarity_score(reaction, segment, reaction_audio_mfcc=None):
    if len(segment) == 4:
        reaction_start, reaction_end, current_start, current_end = segment
        is_filler = False
    else:
        reaction_start, reaction_end, current_start, current_end, is_filler = segment[:5]

    if is_filler:
        return 0

    global segment_mfcc_cosine_scores

    key = get_segment_id(segment)
    if key not in segment_mfcc_cosine_scores:
        if reaction_audio_mfcc is None:
            reaction_audio_mfcc = reaction.get("reaction_audio_mfcc")

        song_audio_mfcc = conf.get("song_audio_mfcc")
        hop_length = conf.get("hop_length")

        mfcc_react_chunk = reaction_audio_mfcc[
            :, round(reaction_start / hop_length) : round(reaction_end / hop_length)
        ]
        mfcc_song_chunk = song_audio_mfcc[
            :, round(current_start / hop_length) : round(current_end / hop_length)
        ]
        mfcc_score = mfcc_cosine_similarity(mfcc1=mfcc_song_chunk, mfcc2=mfcc_react_chunk)
        if mfcc_score < 0 or math.isnan(mfcc_score):
            mfcc_score = 0
        segment_mfcc_cosine_scores[key] = mfcc_score

    return segment_mfcc_cosine_scores[key]


def get_segment_raw_cosine_similarity_score(reaction, segment):
    if len(segment) == 4:
        reaction_start, reaction_end, current_start, current_end = segment
        is_filler = False
    else:
        reaction_start, reaction_end, current_start, current_end, is_filler = segment[:5]

    if is_filler:
        return 0

    global segment_raw_cosine_scores

    key = get_segment_id(segment)
    if key not in segment_raw_cosine_scores:
        base_audio = conf.get("song_audio_data")
        reaction_audio = reaction.get("reaction_audio_data")

        react_chunk = reaction_audio[reaction_start:reaction_end]
        song_chunk = base_audio[current_start:current_end]
        cosine_score = raw_cosine_similarity(song_chunk, react_chunk)
        segment_raw_cosine_scores[key] = cosine_score

    return segment_raw_cosine_scores[key]


#############
##### Ground truth


def calculate_overlap(interval1, interval2):
    """Calculate overlap between two intervals."""
    return max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))


def ground_truth_overlap(path, gt):
    total_overlap = 0

    for sequence in path:
        if not sequence[4]:
            for gt_sequence in gt:
                total_overlap += calculate_overlap(sequence, gt_sequence)

    # Calculate total duration of both paths
    path_duration = sum(s[1] - s[0] for s in path)
    gt_duration = sum(end - start for start, end in gt)

    # If the sum of both durations is zero, there's no meaningful percentage overlap to return
    if path_duration + gt_duration == 0:
        return 0

    # Calculate the percentage overlap
    return (total_overlap * 2) / (path_duration + gt_duration) * 100  # Percentage overlap
