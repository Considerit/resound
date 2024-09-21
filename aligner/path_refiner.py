import os, copy
import numpy as np

from utilities import conversion_audio_sample_rate as sr
from utilities import conf

from aligner.scoring_and_similarity import (
    get_segment_mfcc_cosine_similarity_score,
)


def sharpen_path_boundaries(reaction, original_path):
    sharpener_width = 1 * sr
    step_width = 0.01 * sr

    edge = 3 * sr

    path = copy.deepcopy(original_path)

    path_reaction_start = path[0][0]
    path_base_start = path[0][2]
    path_reaction_end = path[-1][1]
    path_base_end = path[-1][3]

    for i, segment1 in enumerate(path):
        if i < len(path) - 1:
            # Finding the best separator between the coarse segment1 and segment2 definition
            segment2 = path[i + 1]
            (reaction_start1, reaction_end1, base_start1, base_end1, fill1) = segment1
            (reaction_start2, reaction_end2, base_start2, base_end2, fill2) = segment2

            b1 = reaction_start1 - base_start1
            b2 = reaction_start2 - base_start2

            start = max(base_start1, base_end1 - edge)
            score1 = []
            score2 = []
            time_x = []

            # Finding the scores of each segment in the border region
            while start < min(base_start2 + edge, base_end2 - sharpener_width):
                end = start + sharpener_width
                section1 = (start + b1, end + b1, start, end)
                section2 = (start + b2, end + b2, start, end)

                s1 = get_segment_mfcc_cosine_similarity_score(reaction, section1)
                s2 = get_segment_mfcc_cosine_similarity_score(reaction, section2)
                score1.append(s1)
                score2.append(s2)
                time_x.append(start)

                start += step_width

            if len(score1) == 0 or len(score2) == 0:
                continue

            # Identify a segmentation point (a value in time_x) such that the scores at times less
            # than the segmentation point are generally higher for segment1, and the scores at
            # times greater than the segmentation point are generally higher for segment 2.

            # Create cumulative sums
            cumulative_score1 = np.cumsum(score1)
            cumulative_score2 = np.cumsum(score2)

            # Total area under the curves
            total_area_score1 = cumulative_score1[-1]
            total_area_score2 = cumulative_score2[-1]

            potential_transition_points = []
            for i, t in enumerate(time_x):
                # Calculate remaining areas under the curves
                remaining_area_score1 = total_area_score1 - cumulative_score1[i]
                remaining_area_score2 = total_area_score2 - cumulative_score2[i]

                # Check if both conditions are met
                if score2[i] > score1[i] and remaining_area_score2 > remaining_area_score1:
                    potential_transition_points.append(i)

            # If no transition points were found, use the last point
            if not potential_transition_points:
                segmentation_point = time_x[-1]
            else:
                # Find the transition point that maximizes the combined difference in cumulative scores
                max_combined_diff = float("-inf")
                for idx in potential_transition_points:
                    diff_before_transition = cumulative_score1[idx] - cumulative_score2[idx]
                    diff_after_transition = (total_area_score2 - cumulative_score2[idx]) - (
                        total_area_score1 - cumulative_score1[idx]
                    )

                    combined_diff = diff_before_transition + diff_after_transition
                    if combined_diff > max_combined_diff:
                        max_combined_diff = combined_diff
                        segmentation_point = time_x[idx]

            if False:
                # Plot score1 and score2 as y vals of two lines, with time_x being the x axis.
                # Draw a vertical dashed line at the segmentation point.
                plt.plot([t / sr for t in time_x], score1, label="Segment 1 Scores")
                plt.plot([t / sr for t in time_x], score2, label="Segment 2 Scores")

                plt.axvline(
                    x=segmentation_point / sr,
                    color="r",
                    linestyle="--",
                    label="Segmentation Point",
                )
                plt.xlabel("Time (s)")
                plt.ylabel("MFCC Cosine Similarity Score")
                plt.legend()
                plt.show()

            # Now we'll sharpen up the path definition
            segmentation_point = int(segmentation_point)

            assert (
                segmentation_point >= path_base_start,
                f"Segmentation point at {segmentation_point / sr} is less than than {path_base_start/sr}",
                i,
                segment1,
                segment2,
                path,
            )
            assert (
                segmentation_point <= path_base_end,
                f"Segmentation point at {segmentation_point / sr} is greater than {path_base_end/sr}",
                i,
                segment1,
                segment2,
                path,
            )

            segment1[1] = min(path_reaction_end, b1 + segmentation_point)  # new reaction_end
            segment2[0] = max(path_reaction_start, b2 + segmentation_point)  # new reaction_start
            segment1[3] = min(path_base_end, segmentation_point)  # new base_end
            segment2[2] = max(path_base_start, segmentation_point)  # new base_start

    removed_segment = False
    completed_path = []
    for s in path:
        if s[3] > s[2]:
            completed_path.append(s)
        else:
            print(f"************\nREMOVED SEGMENT!!!!! {s[0]/sr} {s[1]/sr} {s[2]/sr} {s[3]/sr}")
            removed_segment = True

    if removed_segment:
        return sharpen_path_boundaries(reaction, completed_path)

    else:
        return path
