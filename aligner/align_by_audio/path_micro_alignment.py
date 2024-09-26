from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm


def micro_align_path(reaction, path, strokes):
    # print("Microaligning:")
    # print_path(path, reaction)

    reaction_len = len(reaction.get("reaction_audio_data"))
    new_path = []
    for idx, segment in enumerate(path):
        (
            reaction_start,
            reaction_end,
            base_start,
            base_end,
            is_filler,
            strokes_key,
        ) = segment
        if is_filler:
            new_path.append(segment)
            continue

        if True:
            my_strokes = strokes[strokes_key]

            if len(my_strokes["strokes"]) == 0:
                new_path.append(segment)
                continue

            scatter = []
            min_intercept = None
            max_intercept = None
            for stroke in my_strokes["strokes"]:
                if stroke[2] >= base_start - 3 * sr and stroke[3] <= base_end + 3 * sr:
                    x = stroke[2]
                    b = stroke[0] - stroke[2]

                    scatter.append((x, b, stroke))

                    if min_intercept is None or min_intercept > b:
                        min_intercept = b

                    if max_intercept is None or max_intercept < b:
                        max_intercept = b

            scatter.sort(key=lambda x: x[0])

            if min_intercept is None or max_intercept is None:
                new_path.append(segment)
                continue

            # my_range = max_intercept - min_intercept
            # if my_range < .05 * sr:
            #     new_path.append(segment)
            #     continue

        else:
            scatter = []
            my_strokes = {"strokes": []}
            chunk_size = get_chunk_size(reaction)
            step = int(0.5 * sr)
            padding = 2 * step

            starting_points = range(base_start, base_end - chunk_size, step)
            for i, start in enumerate(starting_points):
                d = start - base_start
                r_start = reaction_start + d - padding
                r_end = r_start + 2 * padding

                signals, evaluate_with = get_audio_signals(reaction, start, r_start, chunk_size)

                candidates = get_candidate_starts(
                    reaction,
                    signals,
                    peak_tolerance=0.7,
                    open_start=r_start,
                    closed_start=start,
                    chunk_size=chunk_size,
                    distance=1 * sr,
                    upper_bound=r_end,
                    evaluate_with=evaluate_with,
                )

                candidates = [
                    c + r_start + padding
                    for c in candidates
                    if c + r_start + padding >= reaction_start + d
                    and c + r_start + padding + chunk_size <= reaction_end + d
                ]

                y1 = candidates[0] + r_start + padding

                y2 = y1 + chunk_size
                x1 = start
                x2 = start + chunk_size

                new_stroke = (y1, y2, x1, x2)

                b = y1 - x1
                scatter.append((x1, b, new_stroke))

                scatter.sort(key=lambda x: x[0])

                my_strokes["strokes"].append(new_stroke)

        # Extract the x and y values for DBSCAN
        X = [(x, b) for x, b, _ in scatter]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply DBSCAN. You might need to adjust the 'eps' and 'min_samples' parameters based on your data's nature.
        db = DBSCAN(eps=0.25, min_samples=8).fit(X_scaled)

        # Labels assigned by DBSCAN to each point in scatter. -1 means it's an outlier.
        labels = db.labels_

        misc_subsegment = []

        unique_labels = np.unique(labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        subsegments = [[] for _ in unique_labels]

        for i, label in enumerate(labels):
            if label == -1:
                misc_subsegment.append(scatter[i])
            else:
                idx = label_to_idx[label]
                subsegments[idx].append(scatter[i])

        subsegments = [s for s in subsegments if s and len(s) > 0]

        # Break up a subsegment into new subsegments when there is a gap of at
        # least 3 seconds on the x-axis amongst the strokes in the subsegment.
        new_subsegments = []

        for subsegment in subsegments:
            # Sort the strokes in the subsegment by their start time
            subsegment = sorted(subsegment, key=lambda x: x[0])

            # This list will temporarily store strokes of the current fragment of the subsegment
            current_fragment = [subsegment[0]]

            for i in range(1, len(subsegment)):
                # If there's a gap between the current and the previous stroke...
                if subsegment[i][0] - subsegment[i - 1][0] >= 4 * sr:
                    # ... then finalize the current fragment as a new subsegment
                    new_subsegments.append(current_fragment)
                    # ... and start a new fragment for the next strokes
                    current_fragment = []

                # Add the current stroke to the current fragment
                current_fragment.append(subsegment[i])

            # Finalize the last fragment after exiting the loop
            if current_fragment:
                new_subsegments.append(current_fragment)

        # Replace the original subsegments with the new ones
        subsegments = new_subsegments

        if len(subsegments) == 0:
            new_path.append(segment)
            continue

        # if len(subsegments) == 1:
        #     subsegment = subsegments[0]
        #     start_x, intercept, _ = subsegment[0]
        #     end_x, _, _ = subsegment[-1]
        #     new_subsegments = [
        #         [],
        #         [],
        #         [],
        #         [],
        #     ]

        #     for stroke in subsegment:
        #         idx = min(3, math.floor(  4 * (stroke[0] - start_x) / (end_x - start_x)   ))

        #         # stroke = list(stroke)
        #         # import random
        #         # variation = random.randint(-int(sr / 5), int(sr / 5) )
        #         # stroke[0] += variation
        #         # stroke[1] += variation

        #         new_subsegments[idx].append(stroke)

        #     subsegments = [s for s in new_subsegments if len(s) > 0]

        # Filter out clusters that don't span at least 3 seconds
        filtered_subsegments = []
        for subsegment in subsegments:
            start_x, _, _ = subsegment[0]
            end_x, _, _ = subsegment[-1]

            if end_x - start_x < 3 * sr:
                misc_subsegment.extend(subsegment)
            else:
                filtered_subsegments.append(subsegment)

        subsegments = filtered_subsegments
        # lines = find_representative_lines(subsegments)

        # Fit lines to the clusters
        lines = []
        intercept = reaction_start - base_start

        for i, subsegment in enumerate(subsegments):
            subsegment.sort(key=lambda x: x[0])

            intercepts = [point[1] for point in subsegment]

            subsegment_start = subsegment[0][2][2]
            subsegment_end = subsegment[-1][2][3]

            subsegment_reaction_start = max(
                0, reaction_start + (subsegment_start - base_start) - int(sr / 2)
            )
            subsegment_reaction_end = min(
                reaction_len, reaction_end + (subsegment_end - base_end) + int(sr / 2)
            )

            intercepts.append(intercept)
            # print(f"{base_start / sr}-{base_end / sr}  {subsegment_start / sr}-{subsegment_end / sr}  {subsegment_reaction_start / sr}-{subsegment_reaction_end / sr} {(subsegment_end - subsegment_start) / sr} {(max_intercept - min_intercept) / sr}")
            best_intercept = find_best_intercept(
                reaction,
                intercepts,
                subsegment_start,
                subsegment_end,
                include_cross_correlation=False,
                reaction_start=subsegment_reaction_start,
                reaction_end=subsegment_reaction_end,
            )

            lines.append((best_intercept, subsegment, subsegment_start, subsegment_end))

        lines.sort(key=lambda x: x[1][0])

        if len(new_path) == 0:
            previous_end = 0
        else:
            previous_end = new_path[-1][3]

        default_intercept = reaction_start - base_start
        sub_path = []
        if len(lines) == 0:
            new_path.append(segment)
            continue

        # create the new path
        for i, (intercept, subsegment, min_x, max_x) in enumerate(lines):
            min_x = max(min_x, previous_end, base_start)

            if min_x > previous_end + 1:
                my_reaction_start = reaction_start + (previous_end - base_start)
                my_reaction_end = my_reaction_start + (min_x - previous_end)
                fill_intercept = find_best_intercept(
                    reaction,
                    [default_intercept],
                    previous_end,
                    min_x,
                    include_cross_correlation=True,
                    reaction_start=my_reaction_start,
                    reaction_end=my_reaction_end,
                    print_intercepts=False,
                )
                sub_path.append(
                    [
                        previous_end + fill_intercept,
                        min_x + fill_intercept,
                        previous_end,
                        min_x,
                        False,
                    ]
                )

            if i < len(lines) - 1:
                max_x = min(max_x, lines[i + 1][2], base_end)
            else:
                max_x = min(max_x, base_end)

            sub_path.append([min_x + intercept, max_x + intercept, min_x, max_x, False])

            if i == len(lines) - 1 and max_x < base_end:
                fill_intercept = find_best_intercept(
                    reaction,
                    [default_intercept],
                    max_x,
                    base_end,
                    include_cross_correlation=True,
                    reaction_start=reaction_start + (max_x - base_start),
                    reaction_end=reaction_end,
                    print_intercepts=False,
                )

                sub_path.append(
                    [
                        max_x + fill_intercept,
                        base_end + fill_intercept,
                        max_x,
                        base_end,
                        False,
                    ]
                )

            previous_end = max_x

        # Sharpen up the edges of the new path
        sharpened_sub_path = sharpen_path_boundaries(reaction, sub_path)

        # commit the new path to the overall path
        new_path += sharpened_sub_path

        if False:
            x_vals = []  # to store starts for plotting
            y_vals = []  # to store intercepts for plotting

            for stroke in my_strokes["strokes"]:
                if stroke[2] >= base_start - 3 * sr and stroke[3] <= base_end + 3 * sr:
                    b = stroke[0] - stroke[2]
                    # Collecting data for plotting
                    x_vals.append(stroke[2] / sr)
                    y_vals.append(b)

            # Define a list of colors for each subsegment. Make sure there are enough colors for all subsegments.
            colors = cm.rainbow(np.linspace(0, 1, len(subsegments)))

            # Plot each subsegment with its unique color
            for idx, subsegment in enumerate(subsegments):
                subsegment_x_vals = [s[0] / sr for s in subsegment]
                subsegment_y_vals = [s[1] for s in subsegment]
                plt.scatter(
                    subsegment_x_vals,
                    subsegment_y_vals,
                    marker="o",
                    color=colors[idx],
                    label=f"Subsegment {idx + 1}",
                )

            # Plot the misc_subsegment with a distinguishable color, like black
            misc_x_vals = [s[0] / sr for s in misc_subsegment]
            misc_y_vals = [s[1] for s in misc_subsegment]
            if misc_x_vals:  # Only plot if there are any miscellaneous points
                plt.scatter(
                    misc_x_vals,
                    misc_y_vals,
                    marker="o",
                    color="black",
                    label="Miscellaneous",
                )

            # Plotting best_line_def as a green line
            plt.plot(
                [x_vals[0], x_vals[-1]],
                [reaction_start - base_start, reaction_start - base_start],
                color="green",
                label="Best Line",
            )

            for intercept, subsegment, min_x, max_x in lines:
                stroke_length = subsegment[0][2][1] - subsegment[0][2][0]
                min_x = min([p[0] for p in subsegment])
                max_x = max([p[0] for p in subsegment]) + stroke_length
                # plt.plot([min_x / sr, max_x / sr], [intercept, intercept], color='red')

            for rs, re, bs, be, filler in sub_path:
                plt.plot(
                    [bs / sr, be / sr],
                    [rs - bs, rs - bs],
                    linewidth=3,
                    color="orange",
                    label="Best Line",
                )

            for rs, re, bs, be, filler in sharpened_sub_path:
                plt.plot(
                    [bs / sr, be / sr],
                    [rs - bs, rs - bs],
                    linewidth=3,
                    color="purple",
                    label="Sharpened Best Line",
                )

            plt.title("Intercept vs. Start of Strokes")
            plt.xlabel("Start of Stroke")
            plt.ylabel("Intercept")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)

            plt.show()

    return new_path
