import numpy as np
import librosa
import math


from utilities import conversion_audio_sample_rate as sr
from utilities import conf


from prettytable import PrettyTable






################
# Scoring paths
truncated_path_cache = {}
def truncate_path(current_path, end, start=0): 

    key = f"{str(current_path)}-{end}"
    if key not in truncated_path_cache:


        modified_path = []
        adjusted_reaction_end = None

        for segment in current_path:
            (reaction_start, reaction_end, current_start, current_end, filler) = segment
            if current_end >= end:
                if current_start > end:
                    break

                to_trim = current_end - end
                current_end = end
                reaction_end -= to_trim
                modified_path.append( (reaction_start, reaction_end, current_start, current_end, filler) )
                adjusted_reaction_end = reaction_end
                break
            else:
                modified_path.append(segment)
                adjusted_reaction_end = reaction_end

        if adjusted_reaction_end is None: 
            raise Exception(f"WEIRD! Could not calculate partial score for {checkpoint_ts} {current_path}")

        if start > 0:
            back_adjusted_path = modified_path
            modified_path = []
            for segment in back_adjusted_path:
                (reaction_start, reaction_end, current_start, current_end, filler) = segment

                if current_end > start:
                    if current_start < start:
                        modified_path.append( (reaction_start + (start - current_start), reaction_end, start, current_end, filler) )
                    else:
                        modified_path.append(segment)

        truncated_path_cache[key] = (adjusted_reaction_end, modified_path)

    return truncated_path_cache[key]

def calculate_partial_score(reaction, current_path, end, start=0):
    adjusted_reaction_end, modified_path = truncate_path(current_path, end=end, start=start)
    score = path_score(modified_path, reaction, end = end, start=start)     
    return (adjusted_reaction_end, score, modified_path)

def path_score(path, reaction, end=None, start=0): 
    global path_score_cache
    global path_score_cache_perf

    base_audio = conf.get('song_audio_data')
    song_audio_mfcc = conf.get('song_audio_mfcc')
    hop_length = conf.get('hop_length')
    reaction_audio = reaction['reaction_audio_data']
    reaction_audio_mfcc = reaction['reaction_audio_mfcc']    

    key = f"{len(base_audio)} {len(reaction_audio)} {str(path)} {start} {end}"

    if key in path_score_cache:
        path_score_cache_perf['hits'] += 1
        return path_score_cache[key]

    elif 'misses' in path_score_cache_perf:
        path_score_cache_perf['misses'] += 1

    # print(f"path score cache hits/misses = {path_score_cache_perf['hits']} / {path_score_cache_perf['misses']}")

    if end is None:
        end = len(base_audio)

    duration = 0
    fill = 0

    temporal_center = 0
    total_length = 0

    segment_penalty = 1
    for segment in path:
        reaction_start, reaction_end, current_start, current_end, is_filler = segment
        if reaction_start < 0:
            reaction_end += -1 * reaction_start
            reaction_start = 0
        if current_start < 0:
            current_end += -1 * current_start
            current_start = 0

        segment_weight = (current_end - current_start) / (end-start)
        segment_time_center = (reaction_start + (reaction_end - reaction_start) / 2)
        temporal_center += segment_weight * segment_time_center

        if not is_filler:
            duration += (reaction_end - reaction_start)
            total_length += round((reaction_end - reaction_start) / hop_length)

        else:
            fill += current_end - current_start
            total_length += round((current_end - current_start) / hop_length)

        
        if not is_filler and current_end - current_start < sr:
            segment_penalty *= .98


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

    # mfcc_alignment = path_score_by_mfcc_mse_similarity(path, reaction)
    mfcc_alignment = 0
    cosine_mfcc_alignment = path_score_by_mfcc_cosine_similarity(path, reaction)
    # cosine_raw_alignment = path_score_by_raw_cosine_similarity(path, reaction)

    alignment = 100 * cosine_mfcc_alignment # * mfcc_alignment # * cosine_raw_alignment



    duration_score = (duration + fill) / (end-start)


    total_score = segment_penalty * duration_score * duration_score * fill_score * earliness * alignment
    if duration == 0:
        total_score = alignment = 0

    path_score_cache[key] = [total_score, earliness, alignment, fill_score, mfcc_alignment, cosine_mfcc_alignment]
    return path_score_cache[key]


def path_score_by_mfcc_cosine_similarity(path, reaction):
    mfcc_sequence_sum_score = 0

    total_duration = 0 
    for sequence in path:
        total_duration += sequence[1] - sequence[0]

    min_sequence_score = 1

    for sequence in path:
        if len(sequence) < 5 or not sequence[-1]: 
            mfcc_score = get_segment_mfcc_cosine_similarity_score(reaction, sequence)
            duration_factor = (sequence[1] - sequence[0]) / total_duration
            mfcc_sequence_sum_score    += mfcc_score * duration_factor
            if mfcc_score < min_sequence_score:
                min_sequence_score = mfcc_score

    if min_sequence_score < .750:
        mfcc_sequence_sum_score *= .95
    if min_sequence_score < .500:
        mfcc_sequence_sum_score *= .95
    if min_sequence_score < .350:
        mfcc_sequence_sum_score *= .95

    return mfcc_sequence_sum_score


def path_score_by_raw_cosine_similarity(path, reaction):
    sequence_sum_score = 0

    total_duration = 0 
    for sequence in path:
        reaction_start, reaction_end, current_start, current_end, is_filler = sequence
        total_duration += reaction_end - reaction_start

    for sequence in path:
        reaction_start, reaction_end, current_start, current_end, is_filler = sequence

        if not is_filler: 
            score = get_segment_raw_cosine_similarity_score(reaction, sequence)
            duration_factor = (current_end - current_start) / total_duration
            sequence_sum_score    += score * duration_factor

    return sequence_sum_score



def path_score_by_mfcc_mse_similarity(path, reaction):
    total_duration = 0 
    for sequence in path:
        reaction_start, reaction_end, current_start, current_end, is_filler = sequence
        total_duration += reaction_end - reaction_start

    mse_sum = 0
    for sequence in path:
        reaction_start, reaction_end, current_start, current_end, is_filler = sequence
        duration = reaction_end - reaction_start

        mse = get_segment_mfcc_mse(reaction, sequence)

        if math.isnan(mse):
            mse = 0  

        mse_sum += mse * duration / total_duration

    similarity = conf.get('n_mfcc') / (1 + mse_sum)

    return similarity



def get_chunk_score(reaction, reaction_start, reaction_end, current_start, current_end):
    segment = (reaction_start, reaction_end, current_start, current_end, False)
    mse_score = get_segment_mfcc_mse_score(reaction, segment)
    cosine_score = get_segment_mfcc_cosine_similarity_score(reaction, segment)
    # raw_cosine_score = get_segment_raw_cosine_similarity_score(reaction, segment)

    alignment = cosine_score #* mse_score # * raw_cosine_score
    return alignment



def find_best_path(reaction, candidate_paths):

    print(f"Finding the best of {len(candidate_paths)} paths")

    gt = reaction.get('ground_truth')

    assert( len(candidate_paths) > 0 )
    

    best_score = 0 
    best_early_completion_score = 0
    best_similarity = 0 
    best_duration = 0

    if gt: 
        max_gt = 0
        best_gt_path = None
        best_scores = None



    paths_with_scores = []
    for path in candidate_paths:
        scores = path_score(path, reaction)

        if scores[0] > best_score:
            best_score = scores[0]
        if scores[1] > best_early_completion_score:
            best_early_completion_score = scores[1]
        if scores[2] > best_similarity:
            best_similarity = scores[2]
        if scores[3] > best_duration:
            best_duration = scores[3]

        if gt: 
            gtpp = ground_truth_overlap(path, gt)
            if gtpp > max_gt:
                max_gt = gtpp
                best_gt_path = path
                best_scores = scores

        if scores[0] > 0.9 * best_score or best_early_completion_score==scores[1] or best_similarity==scores[2] or best_duration==scores[3]: # winnow it down a bit
            paths_with_scores.append([path, scores])


    print(f"\tDone scoring, now processing paths") 


    for path, scores in paths_with_scores:
        total_score = scores[0] / best_score
        completion_score = scores[1] / best_early_completion_score
        similarity_score = scores[2] / best_similarity
        fill_score = scores[3] / best_duration

        scores[0] = total_score
        scores[1] = completion_score
        scores[2] = similarity_score
        scores[3] = fill_score

    paths_with_scores.sort(key=lambda x: x[1][0], reverse=True)

    print("Paths by score:")
    for idx, (path,scores) in enumerate(paths_with_scores[:1]):
            
        if gt:
            gtpp = ground_truth_overlap(path, gt)
            gtp = f"Ground Truth: {gtpp}%"
        else:
            gtp = ""
        print(f"\tScore={scores[0]}  EarlyThrough={scores[1]}  Similarity={scores[2]} Duration={scores[3]} {gtp}")
        print_path(path, reaction)

    if gt: 
        print("***** Best Ground Truth Path *****")

        print(f"\tScore={best_scores[0]}  EarlyThrough={best_scores[1]}  Similarity={best_scores[2]} Duration={best_scores[3]} {max_gt}")
        print_path(best_gt_path, reaction)



    return paths_with_scores[0][0]







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
    len1 = len2 = min(len1,len2)

    squared_errors = (mfcc1 - mfcc2)**2
    

    if use_mfcc_weights:
        global mfcc_weights
        n_mfcc = conf.get('n_mfcc')
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
        similarity = conf.get('n_mfcc') / (1 + mse)
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
        weighted_similarity = np.dot(similarities, mfcc_cosine_weights) / np.sum(mfcc_cosine_weights)
    else:
        raise Exception('Not implemented')

    #print(weighted_similarity)
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

def create_reaction_mfcc_from_path(path, reaction):

    # these mfcc variables are the result of calls to librosa.feature.mfcc, thus
    # they have a shape of (num_mfcc, length of audio track). The length of the 
    # reaction_audio_mfcc track is greater than the length of song_audio_mfcc track. 
    reaction_audio_mfcc = reaction.get('reaction_audio_mfcc')
    song_audio_mfcc = conf.get('song_audio_mfcc')
    hop_length = conf.get('hop_length')

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
    path_mfcc = np.zeros((song_audio_mfcc.shape[0], total_length))


    # Now we're going to fill up the path_mfcc incrementally, taking from either
    # song_audio_mfcc or reaction_audio_mfcc
    start = 0
    for reaction_start, reaction_end, current_start, current_end, is_filler in path:

        if not is_filler:
            reaction_start = round(reaction_start / hop_length)
            reaction_end = round(reaction_end / hop_length)

            length = reaction_end - reaction_start
            segment = reaction_audio_mfcc[:,reaction_start:reaction_end]
        else:
            length = math.floor((current_end - current_start) / hop_length)

        if length > 0:
            if not is_filler:
                path_mfcc[:,start:start+length] = segment
            start += length

    return path_mfcc


###########
# Printing path

def print_path(path, reaction, ignore_score=False):
    
    gt = reaction.get('ground_truth')
    print("\t\t****************")
    if gt: 
        print(f"\t\tGround Truth Overlap {ground_truth_overlap(path, gt):.1f}%")

    x = PrettyTable()
    x.border = False
    x.align = "r"
    x.field_names = ["\t\t", "", "Base", "Reaction", "mfcc mse", "mfcc cosine", "raw cosine", "ground truth"]

    for sequence in path:
        reaction_start, reaction_end, current_start, current_end, is_filler = sequence

        gt_pr = "-"
        mfcc_mse_score = mfcc_cosine_score = raw_cosine_score = 0

        if not is_filler: 
            if not ignore_score:
                mfcc_mse_score = 1000 * get_segment_mfcc_mse_score(reaction, sequence)
                mfcc_cosine_score = 1000 * get_segment_mfcc_cosine_similarity_score(reaction, sequence)                
                raw_cosine_score = 1000 * get_segment_raw_cosine_similarity_score(reaction, sequence)

                if math.isnan(mfcc_mse_score):
                    mfcc_mse_score = -1
                if math.isnan(mfcc_cosine_score):
                    mfcc_cosine_score = -1
                if math.isnan(raw_cosine_score):
                    raw_cosine_score = -1
            if gt: 
                total_overlap = 0
                for gt_sequence in gt:
                    total_overlap += calculate_overlap(sequence, gt_sequence)

                gt_pr = f"{100 * total_overlap / (sequence[1] - sequence[0]):.1f}%"

        x.add_row(["\t\t",'x' if is_filler else '', f"{float(current_start)/sr:.1f}-{float(current_end)/sr:.1f}", f"{float(reaction_start)/sr:.1f}-{float(reaction_end)/sr:.1f}", f"{round(mfcc_mse_score)}", f"{round(mfcc_cosine_score)}", f"{round(raw_cosine_score)}", gt_pr ])

    
    print(x)

    return x



##############
## Initialization

path_score_cache = {}
path_score_cache_perf = {}

def initialize_path_score():
    global path_score_cache
    global path_score_cache_perf
    global truncated_path_cache

    path_score_cache.clear()
    path_score_cache_perf["hits"] = 0
    path_score_cache_perf["misses"] = 0

    truncated_path_cache.clear()
    

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
        reaction_start, reaction_end, current_start, current_end, is_filler = segment
    return f"{reaction_start} {reaction_end} {current_start} {current_end} {is_filler}"

def get_segment_mfcc_mse_score(reaction, segment):
    if segment[-1]:
      return 0

    mse = get_segment_mfcc_mse(reaction, segment)
    similarity = conf.get('n_mfcc') / (1 + mse)
    return similarity

def get_segment_mfcc_mse(reaction, segment):
    reaction_start, reaction_end, current_start, current_end, is_filler = segment

    global segment_mfcc_mses

    key = get_segment_id(segment)
    if key not in segment_mfcc_mses:
      
      song_audio_mfcc = conf.get('song_audio_mfcc')
      reaction_audio_mfcc = reaction.get('reaction_audio_mfcc')
      hop_length = conf.get('hop_length')

      if is_filler:
          mfcc_react_chunk = np.zeros((song_audio_mfcc.shape[0], reaction_end - reaction_start))
      else: 
          mfcc_react_chunk = reaction_audio_mfcc[:, round(reaction_start / hop_length):round(reaction_end / hop_length)]
      mfcc_song_chunk =      song_audio_mfcc[:, round(current_start / hop_length):round(current_end / hop_length)]
      mfcc_score = mse_mfcc_similarity(mfcc1=mfcc_song_chunk, mfcc2=mfcc_react_chunk, mse_only=True)

      segment_mfcc_mses[key] = mfcc_score

    return segment_mfcc_mses[key]

def get_segment_mfcc_cosine_similarity_score(reaction, segment, reaction_audio_mfcc=None):


    if len(segment) == 4:
        reaction_start, reaction_end, current_start, current_end = segment
        is_filler = False
    else:
        reaction_start, reaction_end, current_start, current_end, is_filler = segment

    if is_filler:
      return 0

    global segment_mfcc_cosine_scores

    key = get_segment_id(segment)
    if key not in segment_mfcc_cosine_scores:

      if reaction_audio_mfcc is None:
        reaction_audio_mfcc = reaction.get('reaction_audio_mfcc')
      
      song_audio_mfcc = conf.get('song_audio_mfcc')
      hop_length = conf.get('hop_length')

      mfcc_react_chunk = reaction_audio_mfcc[:, round(reaction_start / hop_length):round(reaction_end / hop_length)]
      mfcc_song_chunk =      song_audio_mfcc[:, round(current_start / hop_length):round(current_end / hop_length)]
      mfcc_score = mfcc_cosine_similarity(mfcc1=mfcc_song_chunk, mfcc2=mfcc_react_chunk)
      if mfcc_score < 0 or math.isnan(mfcc_score):
        mfcc_score = 0
      segment_mfcc_cosine_scores[key] = mfcc_score

    return segment_mfcc_cosine_scores[key]

def get_segment_raw_cosine_similarity_score(reaction, segment):
    reaction_start, reaction_end, current_start, current_end, is_filler = segment

    if is_filler:
      return 0

    global segment_raw_cosine_scores

    key = get_segment_id(segment)
    if key not in segment_raw_cosine_scores:
      
      base_audio = conf.get('song_audio_data')
      reaction_audio = reaction.get('reaction_audio_data')

      react_chunk = reaction_audio[reaction_start:reaction_end]
      song_chunk =      base_audio[current_start:current_end]
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
    path_duration = sum(end - start for start, end, cstart, cend, filler in path)
    gt_duration = sum(end - start for start, end in gt)
    
    # If the sum of both durations is zero, there's no meaningful percentage overlap to return
    if path_duration + gt_duration == 0:
        return 0

    # Calculate the percentage overlap
    return (total_overlap * 2) / (path_duration + gt_duration) * 100  # Percentage overlap
