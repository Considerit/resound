from utilities import conversion_audio_sample_rate as sr
from utilities import conf

import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time
import math
import shutil

#########################
# Drawing

# This was originally created for the how-I-make-these-concerts video
GENERATE_FULL_ALIGNMENT_VIDEO = False


def get_paint_portfolio_path(reaction, chunk_size):
    paint_portfolio = os.path.join(
        conf.get("temp_directory"),
        f"{reaction.get('channel')}-painting-{int(chunk_size/sr)}",
    )
    if not os.path.exists(paint_portfolio):
        os.makedirs(paint_portfolio)

    return paint_portfolio


def splay_paint(
    reaction,
    strokes,
    stroke_alpha,
    chunk_size,
    draw_intercept=False,
    show_live=True,
    paths=None,
    best_path=None,
    id="",
    copy_to_main=True,
    stroke_color="blue",
    stroke_linewidth=2,
):
    paint_portfolio = get_paint_portfolio_path(reaction, chunk_size)
    plot_fname = os.path.join(paint_portfolio, f"{reaction.get('channel')}-painting-{id}.png")

    # if os.path.exists(plot_fname):
    #     return

    fig = plt.figure(figsize=(20, 10))
    plt.style.use("dark_background")

    ax = fig.gca()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Creating the first plot
    if draw_intercept:
        plt.subplot(1, 2, 1)

    plt.title(
        f"Alignment painting for {reaction.get('channel')} / {conf.get('song_key')}",
        fontsize="xx-large",
        pad=12,
        fontdict={"fontweight": "bold"},
    )

    draw_strokes(
        reaction,
        chunk_size,
        strokes,
        stroke_alpha,
        stroke_color=stroke_color,
        stroke_linewidth=stroke_linewidth,
        paths=paths,
        best_path=best_path,
        intercept_based_figure=False,
    )

    if draw_intercept:
        plt.subplot(1, 2, 2)

        draw_strokes(
            reaction,
            chunk_size,
            strokes,
            stroke_alpha,
            stroke_color=stroke_color,
            stroke_linewidth=stroke_linewidth,
            paths=paths,
            best_path=best_path,
            intercept_based_figure=True,
        )

        plt.tight_layout()

    # Save the plot instead of displaying it

    plt.savefig(plot_fname, dpi=300)

    if copy_to_main:
        # Easy access to last one
        filename = os.path.join(
            conf.get("temp_directory"),
            f"{reaction.get('channel')}-painting-{int(chunk_size/sr)}.png",
        )

        shutil.copyfile(plot_fname, filename)

    if show_live:
        plt.show()
    else:
        plt.close()


def draw_strokes(
    reaction,
    chunk_size,
    strokes,
    stroke_alpha,
    paths,
    best_path=None,
    stroke_linewidth=2,
    stroke_color="blue",
    intercept_based_figure=False,
):
    from aligner.align_by_audio.align_by_audio import get_lower_bounds

    alignment_bounds = reaction["alignment_bounds"]
    if len(alignment_bounds) > 0:
        base_ts, intercepts = zip(*alignment_bounds)
        base_ts = [bs / sr for bs in base_ts]
    else:
        base_ts = [0, conf.get("song_length")]

    if reaction.get("ground_truth"):
        for rs, re, cs, ce, f in reaction.get("ground_truth_path"):
            make_stroke(
                (rs, re, cs, ce),
                alpha=1,
                color="#aaa",
                linewidth=4,
                intercept_based_figure=intercept_based_figure,
            )

    # for segment in strokes:
    #     for stroke in segment['strokes']:
    #         make_stroke(stroke, linewidth=2, alpha = stroke_alpha, intercept_based_figure=intercept_based_figure)

    strokes.sort(key=lambda x: x.get("source", "audio-alignment"))
    for segment in strokes:
        # if not segment.get('pruned', False) and 'old_end_points' in segment:
        if segment.get("pruned", False):
            alpha = 0.25
            color = "black"
            linewidth = stroke_linewidth / 2
        else:
            alpha = stroke_alpha
            color = stroke_color
            linewidth = stroke_linewidth

            if segment.get("source", None) == "image-alignment":
                color = "green"
                linewidth = 1

            make_stroke(
                segment.get("old_end_points", segment["end_points"]),
                linewidth=linewidth,
                color="orange",
                alpha=alpha,
                intercept_based_figure=intercept_based_figure,
            )

        make_stroke(
            segment.get("end_points"),
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            intercept_based_figure=intercept_based_figure,
        )

    # for segment in strokes:
    #     if not segment.get('pruned', False):

    #         for stroke in segment['strokes']:
    #             make_stroke(stroke, linewidth=4, alpha = stroke_alpha, intercept_based_figure=intercept_based_figure)

    #         # make_stroke(segment['end_points'], linewidth=1, color='blue', alpha = 1, intercept_based_figure=intercept_based_figure)

    if best_path is not None:
        visualize_candidate_path(
            best_path,
            intercept_based_figure=intercept_based_figure,
            color="green",
            linewidth=8,
        )

    if paths is not None:
        for path in paths:
            visualize_candidate_path(
                path,
                intercept_based_figure=intercept_based_figure,
                linewidth=2,
                color="orange",
            )

    # Draw the alignment bounds
    if not GENERATE_FULL_ALIGNMENT_VIDEO:
        # Find the min and max values of base_ts for the width of the chart
        x_max = max(base_ts)
        alignment_bound_linewidth = 0.5

        # Upper bounds are red
        for xx, c in alignment_bounds:
            c /= sr  # Calculate the y-intercept
            if intercept_based_figure:
                plt.plot(
                    [0, xx / sr],
                    [c, c],
                    linewidth=alignment_bound_linewidth,
                    alpha=1,
                    color="red",
                    linestyle="dotted",
                )  # Plot the line using the y = mx + c equation
            else:
                plt.plot(
                    [0, xx / sr],
                    [0 + c, xx / sr + c],
                    linewidth=alignment_bound_linewidth,
                    alpha=1,
                    color="red",
                    linestyle="dotted",
                )  # Plot the line using the y = mx + c equation

        # Lower bounds are green
        lower_bounds = get_lower_bounds(reaction, chunk_size)
        for xx, c in lower_bounds:
            c /= sr
            if intercept_based_figure:
                plt.plot(
                    [xx / sr, x_max],
                    [c - xx / sr, c - xx / sr],
                    linewidth=alignment_bound_linewidth,
                    alpha=1,
                    color="green",
                    linestyle="dotted",
                )  # Plot the line using the y = mx + c equation
            else:
                plt.plot(
                    [xx / sr, x_max],
                    [c, c + (x_max - xx / sr)],
                    linewidth=alignment_bound_linewidth,
                    alpha=1,
                    color="green",
                    linestyle="dotted",
                )  # Plot the line using the y = mx + c equation

    x = conf.get("song_audio_data")
    y = reaction.get("reaction_audio_data")

    if intercept_based_figure:
        ylabel = "React Audio intercept"
    else:
        ylabel = "Reaction Audio"

    plt.ylabel(ylabel, fontsize="x-large", labelpad=24)
    plt.xlabel("Song Audio", fontsize="x-large", labelpad=24)

    if GENERATE_FULL_ALIGNMENT_VIDEO:
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        major_ticks = np.arange(0, len(x) / sr + 30, 30)
        x_labels = [seconds_to_timestamp(x) for x in major_ticks]
        plt.xticks(ticks=major_ticks, labels=x_labels)

        major_ticks = np.arange(0, len(y) / sr + 30, 30)
        y_labels = [seconds_to_timestamp(y) for y in major_ticks]
        plt.yticks(ticks=major_ticks, labels=y_labels)

    else:
        plt.xticks(np.arange(0, len(x) / sr + 30, 30))
        plt.yticks(np.arange(0, len(y) / sr + 30, 30))

    plt.grid(True, color=(0.35, 0.35, 0.35))


def make_stroke(stroke, color=None, linewidth=1, alpha=1, intercept_based_figure=False):
    reaction_start, reaction_end, current_start, current_end = stroke

    if color is None:
        color = "red"

    if intercept_based_figure:
        plt.plot(
            [current_start / sr, current_end / sr],
            [
                (reaction_start - current_start) / sr,
                (reaction_start - current_start) / sr,
            ],
            color=color,
            linestyle="solid",
            linewidth=linewidth,
            alpha=alpha,
        )
    else:
        plt.plot(
            [current_start / sr, current_end / sr],
            [reaction_start / sr, reaction_end / sr],
            color=color,
            linestyle="solid",
            linewidth=linewidth,
            alpha=alpha,
        )


def visualize_candidate_path(path, color=None, linewidth=4, alpha=1, intercept_based_figure=False):
    segment_color = color
    for i, segment in enumerate(path):
        reaction_start, reaction_end, current_start, current_end, filler = segment[:5]

        if color is None:
            segment_color = "blue"
            if filler:
                segment_color = "turquoise"

        # draw vertical dashed line connecting segments
        if i > 0:
            (
                last_reaction_start,
                last_reaction_end,
                last_current_start,
                last_current_end,
                last_filler,
            ) = previous_segment[:5]

            if intercept_based_figure:
                plt.plot(
                    [last_current_end / sr, current_start / sr],
                    [
                        (reaction_start - current_start) / sr,
                        (reaction_start - current_start) / sr,
                    ],
                    color=segment_color,
                    linestyle="dashed",
                    linewidth=1,
                    alpha=alpha,
                )
            else:
                plt.plot(
                    [last_current_end / sr, current_start / sr],
                    [last_reaction_end / sr, reaction_start / sr],
                    color=segment_color,
                    linestyle="dashed",
                    linewidth=1,
                    alpha=alpha,
                )

        # draw horizontal segment
        if intercept_based_figure:
            plt.plot(
                [current_start / sr, current_end / sr],
                [
                    (reaction_start - current_start) / sr,
                    (reaction_start - current_start) / sr,
                ],
                color=segment_color,
                linestyle="solid",
                linewidth=linewidth,
                alpha=alpha,
            )
        else:
            plt.plot(
                [current_start / sr, current_end / sr],
                [reaction_start / sr, reaction_end / sr],
                color=segment_color,
                linestyle="solid",
                linewidth=linewidth,
                alpha=alpha,
            )
        previous_segment = segment


def compile_images_to_video(img_dir, video_filename, FPS=10):
    import cv2

    images = sorted(
        [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(".png")],
        key=lambda t: os.stat(t).st_mtime,  # sort by date modified (ascending)
    )
    if not images:
        raise ValueError("No images found in the specified directory!")

    # Find out the frame width and height from the first image
    frame = cv2.imread(images[0])
    h, w, layers = frame.shape
    size = (w, h)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(
        os.path.join(img_dir, video_filename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        size,
    )  # 1 FPS

    for img_path in images:
        img = cv2.imread(img_path)

        if "stroke" in img_path:
            times = 1
        elif "path-test" in img_path:
            times = FPS
        else:
            times = 2 * FPS

        for i in range(times):
            out.write(img)

    out.release()
