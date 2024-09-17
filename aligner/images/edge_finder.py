import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

from aligner.images.frame_operations import crop_with_noise


def find_edges_in_region_of_video(frames, region, is_vertical, visualize=True):
    """
    Finds prominent vertical or horizontal lines in a video using cv2.Canny,
    scoped to a particular region of each frame of that video.

    :param frames: A list of video frames extracted by cv2 in which lines are to be found.
    :param region: Bounding box of the region of each frame in which to find lines.
    :param is_vertical: Whether we're looking for vertical or horizontal lines.
    :param visualize: Whether to show a visualization of the head map of lines found.
    :return: List of line segments detected across the frames.
    """

    # Initialize the heatmap with zeros (same size as the frame, single channel)
    x, y, x2, y2 = region

    if x2 <= x or y2 <= y:
        return None, None

    region_width = int(x2 - x)
    region_height = int(y2 - y)

    line_heatmap = np.zeros((region_height, region_width), dtype=np.float32)

    # Process each frame to find edges and lines

    for i, frame in enumerate(frames):
        # print(f"Processing frame {i + 1}/{len(frames)}...")

        frame_snip = crop_with_noise(frame, region)

        edges, contours = find_edges_and_contours(frame_snip)

        lines = detect_lines(edges, frame_snip, is_vertical=is_vertical)

        # Update the heatmap with the detected lines
        line_heatmap = update_line_heatmap(line_heatmap, lines)

        if False and visualize:
            # Visualize the heatmap incrementally on top of the current frame's contours
            # and lines
            synthesized_lines = synthesize_line_segments(
                line_heatmap, is_vertical=is_vertical
            )

            visualize_line_heatmap(
                line_heatmap,
                line_heatmap.shape[:2],
                len(frames),
                frame_snip,
                edges,
                lines=lines,
                synthesized_lines=synthesized_lines,
                original_frame=frame,
                region=region,
            )

    synthesized_lines = synthesize_line_segments(line_heatmap, is_vertical=is_vertical)

    if visualize:
        # Final heatmap visualization after all frames are processed
        visualize_line_heatmap(
            line_heatmap,
            line_heatmap.shape[:2],
            len(frames),
            frame_snip,
            synthesized_lines=synthesized_lines,
            original_frame=frame,
            region=region,
        )

    # Translate the "end" and "start" points of each synthesized line back
    # into the frame coordinates based on the region.
    for line in synthesized_lines:
        line["start"][0] += x
        line["start"][1] += y
        line["end"][0] += x
        line["end"][1] += y

    return synthesized_lines, line_heatmap


def find_edges_and_contours(frame):
    """
    Detect edges and contours in a single video frame.

    :param frame: The frame in which to detect edges.
    :return: The edges and contours found in the frame.
    """
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return edges, contours


def detect_lines(edges, frame, is_vertical, wiggle_room=1):
    """
    Detect horizontal and vertical lines in the edge image using Hough Line Transform.
    Also, return the edges of the reaction video itself (the borders).

    :param edges: The edges detected in the frame (binary image).
    :param frame_shape: Shape of the video frame (height, width).
    :param wiggle_room: Allowed angle deviation from 0 degrees (horizontal) and 90 degrees (vertical).
    :return: Lists of detected horizontal and vertical lines including the video frame edges.
    """
    frame_height, frame_width = frame.shape[:2]

    # Hough Line Transform to detect horizontal and vertical lines
    rough_lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )
    lines = []

    if rough_lines is not None:
        for line in rough_lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Detect horizontal lines (angle close to 0 or 180 degrees)
            if not is_vertical:
                if (
                    -wiggle_room <= angle <= wiggle_room
                    or 180 - wiggle_room <= angle <= 180 + wiggle_room
                ):
                    lines.append((x1, y1, x2, y2))
            else:
                # Detect vertical lines (angle close to 90 degrees)
                if 90 - wiggle_room <= abs(angle) <= 90 + wiggle_room:
                    lines.append((x1, y1, x2, y2))

    # TODO: this is no longer correct when dealing with regions of a frame

    # # Add the edges of the video frame itself
    # if is_vertical:
    #     # Vertical edges (left and right)
    #     lines.append((0, 0, 0, frame_height))  # Left edge
    #     lines.append((frame_width, 0, frame_width, frame_height))  # Right edge

    # else:
    #     # Horizontal edges (top and bottom)
    #     lines.append((0, 0, frame_width, 0))  # Top edge
    #     lines.append((0, frame_height, frame_width, frame_height))  # Bottom edge

    return lines


def update_line_heatmap(heatmap, lines):
    """
    Update the heatmap by incrementing the values along the detected lines.

    :param heatmap: The heatmap grid to update.
    :param lines: List of lines to draw (either horizontal or vertical).
    """
    temp_heatmap = np.zeros_like(heatmap)

    for x1, y1, x2, y2 in lines:
        # Draw the lines on a temporary heatmap
        cv2.line(temp_heatmap, (x1, y1), (x2, y2), color=1, thickness=3)

    # Increment the main heatmap by the temporary one
    heatmap += temp_heatmap

    return heatmap


def filter_line_segments(segments, tolerance=20):
    """
    Filter line segments such that a segment A is filtered out if there exists another segment B
    whose start is <= A's start, whose end is >= A's end, and whose position is within `tolerance` pixels of A.

    :param segments: List of synthesized line segments (either horizontal or vertical).
    :param tolerance: The allowed pixel difference to consider line segments "near" each other.
    :return: Filtered list of line segments.
    """
    filtered_segments = []

    # Iterate over each segment and compare it with all other segments
    for i, segment_a in enumerate(segments):
        filter_out = False
        for j, segment_b in enumerate(segments):
            if (
                i != j and segment_a["type"] == segment_b["type"]
            ):  # Ensure we're comparing same types (horizontal/vertical)
                if segment_a["type"] == "horizontal":
                    # For horizontal lines, compare x-values and check y-distance
                    if (
                        segment_b["start"][0] <= segment_a["start"][0]
                        and segment_b["end"][0] >= segment_a["end"][0]
                        and abs(segment_b["start"][1] - segment_a["start"][1])
                        <= tolerance
                        and segment_b["total_heat"] > segment_a["total_heat"]
                    ):
                        filter_out = True
                        # print(f"Filtered out:")
                        # print("\t", segment_a)
                        # print("\t", segment_b)
                        break

                elif segment_a["type"] == "vertical":
                    # For vertical lines, compare y-values and check x-distance
                    if (
                        segment_b["start"][1] <= segment_a["start"][1]
                        and segment_b["end"][1] >= segment_a["end"][1]
                        and abs(segment_b["start"][0] - segment_a["start"][0])
                        <= tolerance
                        and segment_b["total_heat"] > segment_a["total_heat"]
                    ):
                        filter_out = True
                        break

        if not filter_out:
            filtered_segments.append(segment_a)

    return filtered_segments


def merge_line_segments(segments, is_vertical, tolerance=1000):
    """
    Merge line segments that share the same position (horizontal: same y-value, vertical: same x-value),
    absorbing smaller segments into larger ones based on total heat. The absorbing segment's properties
    are updated, and the smaller segment is removed.

    :param horizontal_segments: List of horizontal line segments.
    :param vertical_segments: List of vertical line segments.
    :param tolerance: Tolerance to consider lines adjacent (default is 1, meaning directly adjacent).
    :return: Merged list of horizontal and vertical line segments.
    """

    def consume_segment(predator, prey, idx):
        # Absorb predator segment into prey segment

        predator["start"] = list(predator["start"])
        predator["start"][idx] = min(
            predator["start"][idx],
            prey["start"][idx],
        )
        predator["end"] = list(predator["end"])
        predator["end"][idx] = max(
            predator["end"][idx],
            prey["end"][idx],
        )

        predator["length"] = abs(predator["end"][idx] - predator["start"][idx]) + 1

        predator["total_heat"] += prey["total_heat"]

        predator["pixels"] += prey["pixels"]

        predator["avg_quality"] = (
            np.mean([p[2] for p in predator["pixels"]]) / predator["length"]
        )

        # burp
        return predator

    if is_vertical:
        value_idx = 0
        distance_idx = 1
    else:
        value_idx = 1
        distance_idx = 0

    # Group segments by their position (y for horizontal, x for vertical)
    segment_groups = defaultdict(list)
    for segment in segments:
        # Group segments by their value (e.g. horizontal = y-value)
        segment_groups[segment["start"][value_idx]].append(segment)

    merged_segments = []

    # Iterate over each group and merge segments
    for position, group in segment_groups.items():
        group = sorted(group, key=lambda s: s["start"][distance_idx])

        i = 0
        while i < len(group):
            current_segment = group[i]
            j = i + 1

            while j < len(group):
                next_segment = group[j]

                # Check if segments overlap or are adjacent
                if (
                    next_segment["start"][distance_idx]
                    <= current_segment["end"][distance_idx] + tolerance
                ):
                    # Absorb the smaller segment into the larger one based on total heat
                    if next_segment["total_heat"] > current_segment["total_heat"]:
                        current_segment, next_segment = (
                            next_segment,
                            current_segment,
                        )

                    consume_segment(current_segment, next_segment, distance_idx)

                    # Remove the next_segment after absorption
                    group.pop(j)
                else:
                    j += 1

            merged_segments.append(current_segment)
            i += 1

    return merged_segments


def synthesize_line_segments(heatmap, is_vertical, min_length=20, line_threshold=1):
    """
    Synthesize horizontal and vertical line segments from the heatmap.

    :param heatmap: The accumulated heatmap of detected lines.
    :param min_length: Minimum length of line segments (in pixels).
    :param line_threshold: Minimum heat value for a pixel to be considered part of a line.
    :return: List of horizontal and vertical line segments.
    """
    height, width = heatmap.shape
    lines = []

    if not is_vertical:
        # Synthesize horizontal lines by grouping consecutive pixels in each row

        def create_segment(line_pixels):
            start_x = np.min([p[0] for p in line_pixels])
            end_x = np.max([p[0] for p in line_pixels])
            total_heat = np.sum(
                [p[2] for p in line_pixels]
            )  # Sum of the heat of the segment
            median_y = np.median([p[1] for p in line_pixels])
            length = end_x - start_x + 1  # Calculate length of the line segment
            avg_quality = np.mean([p[2] for p in line_pixels]) / length

            return {
                "type": "horizontal",
                "start": [start_x, median_y],
                "end": [end_x, median_y],
                "avg_quality": avg_quality,
                "total_heat": total_heat,
                "length": length,
                "pixels": line_pixels,
            }

        for y in range(height):
            line_pixels = []
            for x in range(width):
                if heatmap[y, x] >= line_threshold:
                    line_pixels.append((x, y, heatmap[y, x]))
                else:
                    if (
                        len(line_pixels) >= min_length
                    ):  # Only consider segments longer than min_length
                        new_segment = create_segment(line_pixels)
                        lines.append(new_segment)

                    line_pixels = []  # Reset the list for the next segment
            if (
                len(line_pixels) >= min_length
            ):  # Handle the case where a segment reaches the end of the row
                new_segment = create_segment(line_pixels)
                lines.append(new_segment)

    if is_vertical:
        # Synthesize vertical lines by grouping consecutive pixels in each column

        def create_segment(line_pixels):
            start_y = np.min([p[1] for p in line_pixels])
            end_y = np.max([p[1] for p in line_pixels])
            avg_quality = np.mean([p[2] for p in line_pixels])
            total_heat = np.sum([p[2] for p in line_pixels])
            median_x = np.median([p[0] for p in line_pixels])
            length = end_y - start_y + 1  # Calculate length of the line segment

            return {
                "type": "vertical",
                "start": [median_x, start_y],
                "end": [median_x, end_y],
                "avg_quality": avg_quality,
                "total_heat": total_heat,
                "length": length,
                "pixels": line_pixels,
            }

        for x in range(width):
            line_pixels = []
            for y in range(height):
                if heatmap[y, x] >= line_threshold:
                    line_pixels.append((x, y, heatmap[y, x]))
                else:
                    if (
                        len(line_pixels) >= min_length
                    ):  # Only consider segments longer than min_length
                        new_segment = create_segment(line_pixels)
                        lines.append(new_segment)

                    line_pixels = []  # Reset the list for the next segment
            if (
                len(line_pixels) >= min_length
            ):  # Handle the case where a segment reaches the end of the column
                new_segment = create_segment(line_pixels)
                lines.append(new_segment)

    # print("Filtering line segments")
    # Filter the line segments based on overlap criteria
    filtered_lines = filter_line_segments(lines, tolerance=20)

    # print("Merging line segments")
    merged_lines = merge_line_segments(filtered_lines, is_vertical, tolerance=500)

    return merged_lines


def visualize_line_heatmap(
    heatmap,
    frame_shape,
    num_frames,
    cropped_frame=None,
    edges=None,
    lines=None,
    synthesized_lines=None,
    original_frame=None,  # The full frame from which cropped_frame was extracted (based on region)
    region=None,  # a bounding box (x, y, x2, y2) that was used to crop cropped_frame
):
    """
    Visualize the heatmap showing the most frequently detected lines, with the option to overlay
    on the current frame's edges (contours) and detected lines. Only the underlying frame will have 0.25 opacity.
    Additionally, visualize synthesized line segments in a subplot.
    """
    # Normalize the heatmap to reflect accumulated intensity

    if cropped_frame is not None:
        assert (
            original_frame is not None and region is not None
        )  # Make sure these are provided if cropped_frame is used

    mask = np.zeros_like(heatmap, dtype=bool)
    edge_margin = 10
    mask[edge_margin:-edge_margin, edge_margin:-edge_margin] = True
    max_value = max(np.max(heatmap[mask]), num_frames / 4)
    if max_value > 0:
        normalized_heatmap = (heatmap / max_value * 255).astype(np.uint8)
    else:
        normalized_heatmap = heatmap.astype(np.uint8)

    # Create a custom green-to-red color map (brighter colors)
    color_map = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        color_map[i, 0] = [128 - i, i, 0]  # Green to Red transition

    heatmap_colored = cv2.LUT(
        cv2.cvtColor(normalized_heatmap, cv2.COLOR_GRAY2BGR), color_map
    )

    # Mask areas of the heatmap where no lines were detected (i.e., where the heatmap is 0)
    mask = heatmap > 0

    # print("Synthesizing line segments")

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Heatmap with original horizontal and vertical lines
    if cropped_frame is not None:
        overlay_frame = original_frame.copy()  # Use the full original frame

        # Resize heatmap to match the cropped frame shape
        heatmap_resized = cv2.resize(
            heatmap_colored, (cropped_frame.shape[1], cropped_frame.shape[0])
        )

        # Expand the mask to 3 channels to match overlay_frame and heatmap_resized
        mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Overlay the heatmap on the cropped area in the full frame
        x1, y1, x2, y2 = region
        cropped_region = overlay_frame[y1:y2, x1:x2]
        if cropped_region[mask_expanded].size == 0:
            print("Warning: cropped_region[mask_expanded] is empty!")

        # Apply the heatmap on the masked area with full opacity using np.where for element-wise operations
        cropped_region = np.where(mask_expanded, heatmap_resized, cropped_region)

        # Place the modified cropped region back into the full frame
        overlay_frame[y1:y2, x1:x2] = cropped_region

        # Draw lines in blue (on top of the heatmap and edges) with full opacity
        if lines:
            for x1, y1, x2, y2 in lines:
                cv2.line(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw a border around the cropped region
        cv2.rectangle(
            overlay_frame,
            (region[0], region[1]),
            (region[2], region[3]),
            (255, 0, 0),
            2,
        )

        axs[0].imshow(cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Frame with Heatmap and Cropped Area")
    else:
        axs[0].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Heatmap")

    axs[0].axis("off")

    # Plot 2: Synthesized line segments
    if cropped_frame is not None:
        overlay_frame = original_frame.copy()  # Use the full original frame

        # Draw synthesized horizontal or vertical lines with offset to account for the region
        for line in synthesized_lines:
            start_x = int(line["start"][0]) + region[0]
            start_y = int(line["start"][1]) + region[1]
            end_x = int(line["end"][0]) + region[0]
            end_y = int(line["end"][1]) + region[1]

            cv2.line(
                overlay_frame,
                (start_x, start_y),
                (end_x, end_y),
                (255, 0, 0),
                2,
            )

        # Draw a border around the cropped region
        cv2.rectangle(
            overlay_frame,
            (region[0], region[1]),
            (region[2], region[3]),
            (255, 0, 0),
            2,
        )

        axs[1].imshow(cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Synthesized Line Segments on Full Frame")
    axs[1].axis("off")

    plt.show()
