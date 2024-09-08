import numpy as np
import matplotlib.pyplot as plt
import cv2, math
from scipy.signal import correlate, fftconvolve, find_peaks

from utilities import conf, conversion_audio_sample_rate as sr


######################################################
# Sound landmarks help find alignment pathways through
# quiet parts of songs by identifying a short distinctive
# sound during a quiet part of the song, and using that
# to find its presence in the reaction video. These
# landmarks can then be used to reward paths align the
# song with that part of the reaction.
#
# To operationalize this, we:
#   1) extract the landmark audio from the song
#   2) cross-correlate this with the reaction audio
#   3) create a binary mask of the quieter sections of
#      the reaction audio
#   4) mask the cross-correlation with the quiet section
#   5) select the top correlations as landmark matches
#      in the reaction audio


##############################
# For checking an alignment segment for whether it contains a landmark
def contains_landmark(
    sound_landmarks, current_start, current_end, reaction_start, reaction_end
):
    landmarks_spanned = 0  # how many landmarks this segment *should* cover
    landmarks_matched = 0  # how many landmarks this segment *did* cover

    landmark_matches = {}

    for landmark_start_sec, landmark in sound_landmarks.items():
        landmark_start = landmark_start_sec * sr
        landmark_end = landmark["end"] * sr
        matches = landmark["matches"]

        if current_start <= landmark_start and current_end >= landmark_end:
            # This segment claims to cover a sound landmark. Does it?
            accurate_match = False
            hypothesized_match = reaction_start + (landmark_start - current_start)
            # print(
            #     f"Hypothesized match for {landmark_start / sr} is {hypothesized_match / sr}"
            # )
            closest_match = 99999999999999999999999999
            for possible_match in matches:
                # print(
                #     f"\tMatch? {possible_match / sr} {abs(possible_match - hypothesized_match) / sr}"
                # )
                diff = abs(possible_match - hypothesized_match)
                if diff < closest_match:
                    closest_match = diff

                if diff < 1 * sr:
                    accurate_match = True
                    landmark_matches[landmark_start_sec] = True
                    break

            landmarks_spanned += 1
            if accurate_match:
                landmarks_matched += 1

            print(
                f"\t\tCovers landmark {sec_to_time(landmark_start / sr)}. Found? Hypothesized match = {sec_to_time(hypothesized_match / sr)} which is {closest_match / sr}s different."
            )
        # else:
        #     print(
        #         f"\t Not in bounds: {sec_to_time(current_start / sr)} - {sec_to_time(current_end / sr)} vs \n\t           {sec_to_time(landmark_start / sr)} - {sec_to_time(landmark_end / sr)}"
        #     )

    return landmarks_spanned, landmarks_matched, landmark_matches


##############################
# For creating the landmarks
def find_sound_landmarks_in_reaction(
    reaction,
    show_visual=False,
):
    if "landmark_matches" not in reaction:
        landmark_matches = {}
        landmarks = conf.get("sound_landmarks", [])

        landmarks = [landmarks[0]]

        (
            silent_regions,
            window_size_seconds,
            mean_loudness,
            loudness,
            loudness_threshold,
        ) = get_silence_binary_mask(reaction.get("reaction_audio_data"))

        print(
            f"Looking for {len(landmarks)} sound landmarks for {reaction.get('channel')}"
        )

        for landmark_start, landmark_end in landmarks:
            landmark_matches[landmark_start] = {
                "end": landmark_end,
                "matches": get_landmark_matches(
                    reaction,
                    landmark_start,
                    landmark_end,
                    silent_regions,
                    window_size_seconds,
                    mean_loudness,
                    loudness,
                    loudness_threshold,
                    show_visual=show_visual,
                ),
            }

            print(f"\t({landmark_start / sr}-{landmark_end / sr}):")
            for peak in landmark_matches[landmark_start]["matches"]:
                print(f"\t\t{peak / sr}: {sec_to_time(peak / sr)}")

        reaction["landmark_matches"] = landmark_matches

    return reaction["landmark_matches"]


# Find silent regions (longer than 3 seconds and quieter than 33% of mean loudness)
def get_silence_binary_mask(
    reaction_audio,
    min_silent_duration=3,
    loudness_threshold=0.666,
    window_size_seconds=1,
):
    loudness = compute_rms_loudness_chunked(reaction_audio, sr, window_size_seconds)
    mean_loudness = np.mean(loudness)

    silent_regions = find_silent_regions(
        loudness,
        mean_loudness,
        sr,
        min_silent_duration=min_silent_duration,
        window_size_seconds=window_size_seconds,
        threshold=loudness_threshold,
    )

    return (
        silent_regions,
        window_size_seconds,
        mean_loudness,
        loudness,
        loudness_threshold,
    )


def get_landmark_matches(
    reaction,
    landmark_start,
    landmark_end,
    silent_regions,
    window_size_seconds,
    mean_loudness,
    loudness,
    loudness_threshold,
    show_visual=False,
):
    landmark = conf.get("song_audio_data")[
        int(sr * landmark_start) : int(sr * landmark_end)
    ]
    reaction_audio = reaction.get("reaction_audio_data")

    # Compute the cross-correlation between the signal and reaction
    correlation = fftconvolve(reaction_audio, landmark[::-1], mode="full")

    max_value = np.max(correlation)
    if max_value <= 0:
        return None

    correlation /= max_value

    lags = np.arange(-len(landmark) + 1, len(reaction_audio))
    lags_in_seconds = lags / sr  # Convert samples to seconds using the sample rate

    # Apply the binary mask to the correlation
    weighted_correlation = apply_binary_mask(
        correlation, silent_regions, sr, window_size_seconds
    )
    max_value = np.max(weighted_correlation)
    if max_value <= 0:
        return None

    weighted_correlation /= max_value

    # Use find_peaks to get peaks above the threshold
    peaks, _ = find_peaks(weighted_correlation, height=0.5, distance=1.5 * sr)

    peaks = [round(correct_peak_index(p, landmark_end - landmark_start)) for p in peaks]

    if show_visual:
        visualize_landmark_correlation(
            lags_in_seconds,
            correlation,
            reaction_audio,
            loudness,
            loudness_threshold,
            mean_loudness,
            silent_regions,
            weighted_correlation,
            peaks,
            reaction.get("video_path"),
        )

    return peaks


def compute_rms_loudness_chunked(audio, sr, window_size_seconds):
    """
    Compute the RMS loudness of the audio signal in chunks to reduce memory usage.
    audio: The audio data (1D array).
    sr: Sample rate of the audio.
    window_size_seconds: The size of each chunk in seconds.
    """
    window_size = int(sr * window_size_seconds)  # Convert seconds to samples
    loudness = []

    # Process the audio in chunks
    for start in range(0, len(audio), window_size):
        end = start + window_size
        chunk = audio[start:end]
        if len(chunk) > 0:
            rms_value = np.sqrt(np.mean(chunk**2))
            loudness.append(rms_value)

    return np.array(loudness)


def find_silent_regions(
    loudness,
    mean_loudness,
    sr,
    window_size_seconds,
    min_silent_duration,
    threshold,
):
    """Find silent regions where loudness is less than threshold * mean loudness."""
    silence_threshold = threshold * mean_loudness
    silence_mask = (loudness < silence_threshold).astype(int)

    # Convert min_silent_duration from seconds to samples (based on window size)
    min_silent_samples = int(
        min_silent_duration / window_size_seconds
    )  # Adjust for window size

    silent_regions = np.zeros_like(silence_mask)

    start = 0
    while start < len(silence_mask):
        if silence_mask[start] == 1:
            end = start
            while end < len(silence_mask) and silence_mask[end] == 1:
                end += 1
            if (end - start) >= min_silent_samples:
                silent_regions[start:end] = 1
            start = end
        else:
            start += 1

    return silent_regions


def expand_binary_mask(binary_mask, correlation_length, sr, window_size_seconds):
    """
    Expand the binary mask to match the length of the correlation array.
    Each value in the binary mask corresponds to a chunk of the audio, so we repeat
    the value of the mask for all samples that fall into the corresponding chunk.
    """
    # Calculate the number of samples corresponding to one chunk
    samples_per_chunk = int(sr * window_size_seconds)

    # Repeat each value in the binary mask to cover the corresponding number of samples
    expanded_mask = np.repeat(binary_mask, samples_per_chunk)

    # Trim or pad the expanded mask to match the correlation length
    if len(expanded_mask) > correlation_length:
        expanded_mask = expanded_mask[:correlation_length]
    elif len(expanded_mask) < correlation_length:
        expanded_mask = np.pad(
            expanded_mask, (0, correlation_length - len(expanded_mask)), mode="constant"
        )

    return expanded_mask


def apply_binary_mask(correlation, binary_mask, sr, window_size_seconds):
    """Apply the expanded binary mask to the correlation."""
    # Expand the binary mask to match the length of the correlation array
    expanded_mask = expand_binary_mask(
        binary_mask, len(correlation), sr, window_size_seconds
    )

    # Apply the expanded mask to the correlation
    masked_correlation = correlation * expanded_mask

    return masked_correlation


def visualize_landmark_correlation(
    lags_in_seconds,
    correlation,
    reaction_audio,
    loudness,
    loudness_threshold,
    mean_loudness,
    silent_regions,
    weighted_correlation,
    peaks,
    reaction_vid_path,
):
    # Create a plot with 4 subplots: correlation, loudness, binary mask, and weighted correlation
    fig, axs = plt.subplots(4, 1, figsize=(16, 9), sharex=True)  # Adjust figure size

    # Plot the correlation
    axs[0].plot(lags_in_seconds, correlation)
    axs[0].set_title("Correlation between Signal and Reaction")
    axs[0].set_ylabel("Correlation")
    axs[0].grid(True)

    # Plot the relative loudness
    loudness_time = np.linspace(0, len(reaction_audio) / sr, num=len(loudness))
    axs[1].plot(loudness_time, loudness, label="Loudness (RMS)")

    # Compute and plot the loudness threshold
    abs_loudness_threshold = loudness_threshold * mean_loudness
    axs[1].axhline(
        y=abs_loudness_threshold,
        color="r",
        linestyle="--",
        label=f"Threshold ({abs_loudness_threshold:.2f})",
    )

    axs[1].set_title("Relative Loudness of Reaction Audio")
    axs[1].set_ylabel("Loudness (RMS)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    # Plot the binary mask for silent regions
    axs[2].plot(loudness_time, silent_regions)
    axs[2].set_title("Binary Mask for Silent Regions (1 = Silent)")
    axs[2].set_ylabel("Mask (0 or 1)")
    axs[2].grid(True)

    # Plot the weighted correlation (correlation * silent regions mask)
    axs[3].plot(lags_in_seconds, weighted_correlation, label="Weighted Correlation")

    # Mark the peaks on the weighted correlation plot
    axs[3].plot(
        lags_in_seconds[peaks], weighted_correlation[peaks], "rx", label="Peaks"
    )

    axs[3].set_title("Weighted Correlation with Peaks (Emphasis on Silent Regions)")
    axs[3].set_xlabel("Time (seconds)")
    axs[3].set_ylabel("Weighted Correlation")
    axs[3].legend(loc="upper right")
    axs[3].grid(True)

    # Extract video frames at peaks
    cap = cv2.VideoCapture(reaction_vid_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {reaction_vid_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    peaks.sort()
    frames = []
    for peak in peaks:
        frame_number = int(peak / sr * fps)

        # Set the video to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if not ret:
            print(f"Error: Could not read frame {frame_number}")
            continue

        # Convert the frame from BGR (OpenCV format) to RGB (Matplotlib format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    # Calculate the number of frames and how they will fit in the plot
    num_frames = len(frames)
    if num_frames == 0:
        print("No frames to display.")
        return

    # Set the height of the frames (fixed height for each image)
    frame_height = 0.15  # 15% of the figure height for all frames
    fig_width = fig.get_size_inches()[0]
    frame_width = fig_width / num_frames  # Width of each frame to fit in the figure

    # Define a new axes at the bottom for the frames
    for i, frame in enumerate(frames):
        left = i / num_frames
        bottom = 0.01  # Fixed bottom margin for the frames
        width = 1 / num_frames
        height = frame_height  # Fixed height for each frame (15% of figure height)

        # Add each frame as a new axes on the figure
        new_ax = fig.add_axes([left, bottom, width, height], anchor="SW", zorder=1)
        new_ax.imshow(frame)
        new_ax.axis("off")  # Hide the axes for the frame

    # Adjust layout to make space for frames at the bottom
    plt.tight_layout(
        rect=[0, 0.15, 1, 1]
    )  # Leave space for frames (bottom 15% for frames)
    plt.show()


def correct_peak_index(peak_index, chunk_len):
    # return max(0,peak_index)
    return max(0, peak_index - (chunk_len - 1))


def sec_to_time(sec):
    return f"{math.floor(sec / 60)}:{sec % 60}"
