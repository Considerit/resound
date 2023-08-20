import numpy as np
import librosa
import math

from utilities.audio_processing import audio_percentile_loudness


def get_chunk_score(basics, reaction_start, reaction_end, current_start, current_end):

    base_audio = basics.get('base_audio')
    reaction_audio = basics.get('reaction_audio')

    base_audio_mfcc = basics.get('base_audio_mfcc')
    base_audio_vol_diff = basics.get('song_percentile_loudness')
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    reaction_audio_vol_diff = basics.get('reaction_percentile_loudness')
    sr = basics.get('sr')
    hop_length = basics.get('hop_length')


    chunk = base_audio[current_start:current_end]
    reaction_chunk = reaction_audio[reaction_start:reaction_end]            

    mfcc_react_chunk = reaction_audio_mfcc[:, round(reaction_start / hop_length):round(reaction_end / hop_length)]
    mfcc_song_chunk =      base_audio_mfcc[:, round(current_start / hop_length):round(current_end / hop_length)]

    # voldiff_react_chunk = reaction_audio_vol_diff[round(reaction_start / hop_length):round(reaction_end / hop_length)]
    # voldiff_song_chunk =      base_audio_vol_diff[round(current_start / hop_length):round(current_end / hop_length)]
    
    mfcc_score = mfcc_similarity(sr, mfcc1=mfcc_song_chunk, mfcc2=mfcc_react_chunk)
    # rel_volume_alignment = relative_volume_similarity(sr, vol_diff1=voldiff_song_chunk, vol_diff2=voldiff_react_chunk)

    # alignment = math.log(1 + 100 * mfcc_score) * math.log(1 + rel_volume_alignment)
    alignment = mfcc_score
    return alignment



def calculate_partial_score(current_path, checkpoint_ts, basics):

    modified_path = []
    adjusted_reaction_end = None

    for segment in current_path:
        (reaction_start, reaction_end, current_start, current_end, filler) = segment
        if current_end >= checkpoint_ts:
            if current_start > checkpoint_ts:
                break

            to_trim = current_end - checkpoint_ts
            current_end = checkpoint_ts
            reaction_end -= to_trim
            modified_path.append( (reaction_start, reaction_end, current_start, current_end, filler) )
            adjusted_reaction_end = reaction_end
            break
        else:
            modified_path.append(segment)
            adjusted_reaction_end = reaction_end

    if adjusted_reaction_end is None: 
        print(f"WEIRD! Could not calculate partial score for {checkpoint_ts} {current_path}")
        return None

    # truncate current path back to current_ts
    score = path_score(modified_path, basics, relative_to = checkpoint_ts) 

    return adjusted_reaction_end, score







def mfcc_similarity(sr, audio_chunk1=None, audio_chunk2=None, mfcc1=None, mfcc2=None):

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

    # Compute mean squared error between MFCCs
    mse = np.mean((mfcc1 - mfcc2)**2)

    similarity = len(mfcc1) / (1 + mse)
    return similarity


def relative_volume_similarity(sr, audio_chunk1=None, audio_chunk2=None, vol_diff1=None, vol_diff2=None, hop_length=1):

    # Compute MFCCs for each audio chunk
    if vol_diff1 is None: 
        vol_diff1 = audio_percentile_loudness(audio_chunk1, loudness_window_size=100, percentile_window_size=100, std_dev_percentile=None, hop_length=hop_length)

    if vol_diff2 is None: 
        vol_diff2 = audio_percentile_loudness(audio_chunk2, loudness_window_size=100, percentile_window_size=100, std_dev_percentile=None, hop_length=hop_length)

    # Make sure of same shape
    len1 = vol_diff1.shape[0]
    len2 = vol_diff2.shape[0]

    if len1 != len2:
        # if abs(len1 - len2) > 1:
        #     print("VOL DIFF SHAPES NOT EQUAL", len1, len2)
        #     # assert(abs(len1 - len2) == 1)

        if len2 > len1:
            vol_diff2 = vol_diff2[:len1]
        else:
            vol_diff1 = vol_diff1[:len2]



    # Calculate the absolute difference between the two volume differences
    absolute_difference = np.abs(vol_diff1 - vol_diff2)

    # Calculate the sum of absolute_difference
    difference_magnitude = np.sum(absolute_difference)


    # Max difference is for volume inversion, where abs(vol_diff1 - vol_diff2) is always 100 (%)
    max_difference = 100 * len(vol_diff1)
    return max_difference / difference_magnitude




def create_reaction_mfcc_from_path(path, basics):

    # these mfcc variables are the result of calls to librosa.feature.mfcc, thus
    # they have a shape of (num_mfcc, length of audio track). The length of the 
    # reaction_audio_mfcc track is greater than the length of base_audio_mfcc track. 
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    base_audio_mfcc = basics.get('base_audio_mfcc')
    hop_length = basics.get('hop_length')

    total_length = 0
    for reaction_start, reaction_end, current_start, current_end, is_filler in path:
        if not is_filler:
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)
            total_length += reaction_end - reaction_start
        else:
            current_start = round(current_start / hop_length)
            current_end = round(current_end / hop_length)
            total_length += current_end - current_start



    # the resulting combined mfcc (path_mfcc) should have the same number of mfccs 
    # as the input mfccs, and be the length of the reaction_audio_for_path. 
    path_mfcc = np.zeros((base_audio_mfcc.shape[0], total_length))


    # Now we're going to fill up the path_mfcc incrementally, taking from either
    # base_audio_mfcc or reaction_audio_mfcc
    start = 0
    for reaction_start, reaction_end, current_start, current_end, is_filler in path:

        if not is_filler:
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)

            length = reaction_end - reaction_start
            segment = reaction_audio_mfcc[:,reaction_start:reaction_end]
        else:
            current_start = round(current_start / hop_length)
            current_end = round(current_end / hop_length)

            length = current_end - current_start
            segment = base_audio_mfcc[:,current_start:current_end]

        if length > 0:
            path_mfcc[:,start:start+length] = segment
            start += length

    return path_mfcc

def create_reaction_vol_diff_from_path(path, basics):

    # these mfcc variables are the result of calls to librosa.feature.mfcc, thus
    # they have a shape of (num_mfcc, length of audio track). The length of the 
    # reaction_audio_mfcc track is greater than the length of base_audio_mfcc track. 
    reaction_audio_mfcc = basics.get('reaction_percentile_loudness')
    base_audio_mfcc = basics.get('song_percentile_loudness')
    hop_length = basics.get('hop_length')

    total_length = 0
    for reaction_start, reaction_end, current_start, current_end, is_filler in path:
        if not is_filler:
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)
            total_length += reaction_end - reaction_start
        else:
            current_start = round(current_start / hop_length)
            current_end = round(current_end / hop_length)
            total_length += current_end - current_start



    # the resulting combined mfcc (path_mfcc) should have the same number of mfccs 
    # as the input mfccs, and be the length of the reaction_audio_for_path. 
    path_mfcc = np.zeros(total_length)


    # Now we're going to fill up the path_mfcc incrementally, taking from either
    # base_audio_mfcc or reaction_audio_mfcc
    start = 0
    for reaction_start, reaction_end, current_start, current_end, is_filler in path:


        if not is_filler:
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)

            length = reaction_end - reaction_start
            segment = reaction_audio_mfcc[reaction_start:reaction_end]
        else:
            current_start = round(current_start / hop_length)
            current_end = round(current_end / hop_length)

            length = current_end - current_start
            segment = base_audio_mfcc[current_start:current_end]

        if length > 0:
            path_mfcc[start:start+length] = segment
            start += length

    return path_mfcc




path_score_cache = {}
path_score_cache_perf = {}

def initialize_path_score():
    global path_score_cache
    global path_score_cache_perf

    path_score_cache.clear()
    path_score_cache_perf["hits"] = 0
    path_score_cache_perf["misses"] = 0
    

def path_score(path, basics, relative_to=None): 
    global path_score_cache
    global path_score_cache_perf

    base_audio = basics.get('base_audio')
    base_audio_mfcc = basics.get('base_audio_mfcc')
    base_audio_vol_diff = basics.get('song_percentile_loudness')
    hop_length = basics.get('hop_length')
    reaction_audio = basics.get('reaction_audio')
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    sr = basics.get('sr')

    key = f"{len(base_audio)} {len(reaction_audio)} {str(path)} {relative_to}"
    if key in path_score_cache:
        path_score_cache_perf['hits'] += 1
        return path_score_cache[key]

    path_score_cache_perf['misses'] += 1

    # print(f"path score cache hits/misses = {path_score_cache_perf['hits']} / {path_score_cache_perf['misses']}")

    if relative_to is None:
        relative_to = len(base_audio)



    duration = 0
    fill = 0

    temporal_center = 0
    total_length = 0

    for reaction_start, reaction_end, current_start, current_end, is_filler in path:
        if reaction_start < 0:
            reaction_end += -1 * reaction_start
            reaction_start = 0
        if current_start < 0:
            current_end += -1 * current_start
            current_start = 0

        segment_weight = (current_end - current_start) / relative_to
        segment_time_center = (reaction_start + (reaction_end - reaction_start) / 2)
        temporal_center += segment_weight * segment_time_center

        if not is_filler:
            duration += (reaction_end - reaction_start)
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)
            total_length += reaction_end - reaction_start            

        else:
            fill += current_end - current_start
            current_start = round(current_start / hop_length)
            current_end = round(current_end / hop_length)
            total_length += current_end - current_start



    # Derivation for below:
    #   earliness = |R| / temporal_center
    #   best possible earliness =  |R| / (|B| / 2) (when the first part of the reaction is matched with the full base audio)
    #   normalized earliness = earliness / best_possible_earliness = |B| / (|R| * temporal_center)
    #   ...but I'm going to change the normalization to match being in the middle of the reaction, because sometimes
    #      the reactions are really really long.
    #   middle_earliness = |R| / ( |R| / 2  ) = 2
    #   normalized earliness = earliness / middle_earliness
    # normalized_earliness_score = len(base_audio) / (len(reaction_audio) * temporal_center)
    # fill_score = 1 / (1 + abs(duration - len(base_audio)) / sr)

    # earliness = len(reaction_audio) / temporal_center
    # earliness = math.log(1 + earliness)

    earliness = len(base_audio) / temporal_center

    fill_score = duration / (duration + fill)


    path_mfcc = create_reaction_mfcc_from_path(path, basics)
    # path_vol_diff = create_reaction_vol_diff_from_path(path, basics)

    mfcc_alignment = mfcc_similarity(sr, mfcc1=base_audio_mfcc[:,:total_length], mfcc2=path_mfcc)
    # rel_vol_alignment = relative_volume_similarity(sr, vol_diff1=base_audio_vol_diff[:total_length], vol_diff2=path_vol_diff)

    alignment = 100 * mfcc_alignment #* math.log10(1 + rel_vol_alignment)

    duration_score = (duration + fill) / relative_to

    total_score = duration_score * duration_score * fill_score * fill_score * earliness * alignment
    if duration == 0:
        total_score = alignment = 0

    path_score_cache[key] = [total_score, earliness, alignment, fill_score]
    return path_score_cache[key]


def find_best_path(candidate_paths, basics):
    sr = basics.get('sr')
    base_audio_mfcc = basics.get('base_audio_mfcc')
    base_audio_vol_diff = basics.get('song_percentile_loudness')
    reaction_audio_mfcc = basics.get('reaction_audio_mfcc')
    reaction_audio_vol_diff = basics.get('reaction_percentile_loudness')
    hop_length = basics.get('hop_length')
    gt = basics.get('ground_truth')
    if gt: 
        gt = [ (int(s * sr), int(e * sr)) for (s,e) in gt]



    assert( len(candidate_paths) > 0 )
    
    paths_with_scores = []

    for path in candidate_paths:
        paths_with_scores.append([path, path_score(path, basics)])

    best_score = 0 
    best_early_completion_score = 0
    best_similarity = 0 
    best_duration = 0
    for path, scores in paths_with_scores:
        if scores[0] > best_score:
            best_score = scores[0]
        if scores[1] > best_early_completion_score:
            best_early_completion_score = scores[1]
        if scores[2] > best_similarity:
            best_similarity = scores[2]
        if scores[3] > best_duration:
            best_duration = scores[3]


    for path, scores in paths_with_scores:
        total_score = scores[0] / best_score
        completion_score = scores[1] / best_early_completion_score
        similarity_score = scores[2] / best_similarity
        fill_score = scores[3] / best_duration

        scores[0] = total_score
        scores[1] = completion_score
        scores[2] = similarity_score
        scores[3] = fill_score

        # score = (fill_score + completion_score + similarity_score) / 3
        # scores.insert(0, score)

    paths_by_score = sorted(paths_with_scores, key=lambda x: x[1][0], reverse=True)

    print("Paths by score:")
    for path,scores in paths_by_score:
        if scores[0] > 0.9:
            if gt:
                gtp = f"Ground Truth: {ground_truth_overlap(path, gt)}%"
            else:
                gtp = ""
            print(f"\tScore={scores[0]}  EarlyThrough={scores[1]}  Similarity={scores[2]} Duration={scores[3]} {gtp}")
            for sequence in path:
                reaction_start, reaction_end, current_start, current_end, is_filler = sequence

                mfcc_react_chunk = reaction_audio_mfcc[:, round(reaction_start / hop_length):round(reaction_end / hop_length)]
                mfcc_song_chunk =      base_audio_mfcc[:, round(current_start / hop_length):round(current_end / hop_length)]

                voldiff_react_chunk = reaction_audio_vol_diff[round(reaction_start / hop_length):round(reaction_end / hop_length)]
                voldiff_song_chunk =      base_audio_vol_diff[round(current_start / hop_length):round(current_end / hop_length)]

                if is_filler: 
                    mfcc_react_chunk = mfcc_song_chunk
                    voldiff_react_chunk = voldiff_song_chunk

                mfcc_score = mfcc_similarity(sr, mfcc1=mfcc_song_chunk, mfcc2=mfcc_react_chunk)
                rel_volume_alignment = relative_volume_similarity(sr, vol_diff1=voldiff_song_chunk, vol_diff2=voldiff_react_chunk)
                print(f"\t\t{'*' if is_filler else ''}base: {float(current_start)/sr:.1f}-{float(current_end)/sr:.1f}  reaction: {float(reaction_start)/sr:.1f}-{float(reaction_end)/sr:.1f} [mfcc: {mfcc_score}] [relvol: {rel_volume_alignment}]")


    return paths_by_score[0][0]


def calculate_overlap(interval1, interval2):
    """Calculate overlap between two intervals."""
    return max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))

def ground_truth_overlap(path, gt):
    total_overlap = 0

    for p_interval in path:
        for gt_interval in gt:
            total_overlap += calculate_overlap(p_interval, gt_interval)

    # Calculate total duration of both paths
    path_duration = sum(end - start for start, end, cstart, cend, filler in path)
    gt_duration = sum(end - start for start, end in gt)
    
    # If the sum of both durations is zero, there's no meaningful percentage overlap to return
    if path_duration + gt_duration == 0:
        return 0

    # Calculate the percentage overlap
    return (total_overlap * 2) / (path_duration + gt_duration) * 100  # Percentage overlap