import glob, os
import librosa
import ffmpeg
import numpy as np

from utilities.utilities import conversion_audio_sample_rate as sr
from utilities.utilities import extract_audio
from utilities.audio_processing import (
    normalize_audio_manually,
    pitch_contour,
    spectral_flux,
    root_mean_square_energy,
    continuous_wavelet_transform,
    convert_to_mono,
    normalize_reaction_audio,
)
from utilities import conf, save_object_to_file, read_object_from_file

from utilities import print_profiling

from library import channel_labels

from moviepy.editor import vfx, VideoFileClip

from inventory.channels import get_reactors_inventory

# from guppy import hpy
# h = hpy()


conf = {}  # global conf

constrain_to = [
    "Andrea Silvestro",
    "Braindead Breakdown",
    "The Music Recording Network",
]
constrain_to = []

from library import search_testers, search_defs, test_title


def get_song_directory(song):
    return os.path.join("Media", song)


def make_conf(song_def, options, temp_directory):
    global conf

    conf.clear()

    conf.update(options)

    song = f"{song_def['artist']} - {song_def['song']}"

    song_directory = get_song_directory(song)
    reaction_directory = os.path.join(song_directory, "reactions")

    compilation_path = os.path.join(song_directory, f"{song} (compilation).mp4")
    # if conf.get('draft', False):
    #     compilation_path += "fast.mp4"

    full_output_dir = os.path.join(song_directory, temp_directory)
    if not os.path.exists(full_output_dir):
        # Create a new directory because it does not exist
        os.makedirs(full_output_dir)

    base_audio_path = os.path.join(song_directory, f"{os.path.basename(song_directory)}.wav")

    intro_path = os.path.join(song_directory, "intro.mp4")
    if not os.path.exists(intro_path):
        intro_path = False

    outro_path = os.path.join(song_directory, "outro.mp4")
    if not os.path.exists(outro_path):
        outro_path = False

    background_extensions = ["mp4", "png", "jpg", "jpeg"]
    for ext in background_extensions:
        background_path = os.path.join(song_directory, f"background.{ext}")
        if os.path.exists(background_path):
            break

    if not os.path.exists(background_path):
        background_path = None

    def reaction_title_tester(song):
        defs = search_defs.get(song)
        return lambda title, is_channel_search=False: test_title(
            title, is_channel_search=is_channel_search, defs=defs
        )

    include_base_video = song_def.get("include_base_video", True)

    default_placement = "center / bottom" if song_def.get("include_base_video") else "centered"

    min_segment_length_in_seconds = 3
    first_n_samples = int(min_segment_length_in_seconds * sr)

    conf.update(
        {
            "artist": song_def["artist"],
            "song_name": song_def["song"],
            "song_key": song,
            "base_audio_path": base_audio_path,
            "include_base_video": include_base_video,
            "song_directory": song_directory,
            "reaction_directory": reaction_directory,
            "transcription_directory": os.path.join(reaction_directory, "transcripts"),
            "compilation_path": compilation_path,
            "temp_directory": full_output_dir,
            "has_asides": None if options.get("skip_asides", False) else song_def.get("asides"),
            "hop_length": 256,
            "n_mfcc": 20,
            # 'channel_branding': 'production/resound_channel_reencoded.mp4',
            "introduction": intro_path,
            "outro": outro_path,
            "search_tester": search_testers.get(song, reaction_title_tester(song)),
            "background": background_path,
            "convert_videos": song_def.get("convert_videos", []),
            "base_video_transformations": song_def.get("base_video_transformations", {}),
            "disable_backchannel_backgrounding": song_def.get(
                "disable_backchannel_backgrounding", False
            ),
            "audio_mixing": song_def.get("audio_mixing", {}),
            "base_video_proportion": song_def.get("base_video_proportion", 0.45),
            "base_video_placement": song_def.get("base_video_placement", default_placement),
            "zoom_pans": song_def.get("zoom_pans", []),
            "background.reverse": song_def.get("background.reverse", False),
            "sound_landmarks": song_def.get("sound_landmarks", []),
            # alignment parameters
            "step_size": 1,
            "min_segment_length_in_seconds": min_segment_length_in_seconds,
            "peak_tolerance": 0.5,
            "first_n_samples": first_n_samples,
            "use_image_based_alignment": song_def.get("use_image_based_alignment", True),
        }
    )

    def load_reactions():
        from inventory.inventory import get_reactions_manifest

        global channel_labels

        from inventory import get_manifest_path

        try:
            manifest = open(get_manifest_path(conf["artist"], conf["song_name"]), "r")
        except:
            print(f"Manifest doesn't yet exist for {conf['song_name']}")
        reaction_videos = prepare_reactions(conf.get("convert_videos"))

        temp_directory = conf.get("temp_directory")

        reactions = {}

        multiple_reactors = song_def.get("multiple_reactors", None)
        priority = song_def.get("priority", {})
        swap_grid_positions = song_def.get("swap_grid_positions", None)
        fake_reactor_position = song_def.get("fake_reactor_position", None)
        start_reaction_search_at = song_def.get("start_reaction_search_at", {})

        end_reaction_search_at = song_def.get("end_reaction_search_at", None)
        unreliable_bounds = song_def.get("unreliable_bounds", None)
        chunk_size = song_def.get("chunk_size", {})
        manual_normalization_point = song_def.get("manual_normalization_point", None)
        insert_filler = song_def.get("insert_filler", None)
        manual_bounds = song_def.get("manual_bounds", None)
        foregrounded_backchannel = song_def.get("foregrounded_backchannel", None)
        backgrounded_backchannel = song_def.get("backgrounded_backchannel", None)

        face_orientation = song_def.get("face_orientation", None)

        mute = song_def.get("mute", None)

        reactors_inventory = get_reactors_inventory()
        reactions_manifest = get_reactions_manifest(song_def["artist"], song_def["song"])[
            "reactions"
        ]
        metrics = {}

        for i, reaction_video_path in enumerate(reaction_videos):
            print_profiling()
            channel, __ = os.path.splitext(os.path.basename(reaction_video_path))

            target_score = song_def.get("target_scores", {}).get(channel, None)

            if len(constrain_to) > 0 and channel not in constrain_to:
                continue

            ground_truth = song_def.get("ground_truth", {}).get(channel, None)
            ground_truth_path = None
            if ground_truth:
                ground_truth = [(int(s * sr), int(e * sr)) for (s, e) in ground_truth]
                ground_truth_path = []
                current_start = 0
                for reaction_start, reaction_end in ground_truth:
                    current_end = current_start + reaction_end - reaction_start

                    ground_truth_path.append(
                        (
                            reaction_start,
                            reaction_end,
                            current_start,
                            current_end,
                            False,
                        )
                    )

                    current_start += reaction_end - reaction_start

            if channel == "Resound":
                channelId = "XXXXXX"
                vid = None
                reaction_manifest = {}
            else:
                reaction_manifest = None
                channelId = None
                vid = None
                for vidID, reaction_manifest in reactions_manifest.items():
                    reaction_file_prefix = reaction_manifest.get(
                        "file_prefix", reaction_manifest["reactor"]
                    )
                    if channel == reaction_file_prefix:
                        channelId = reaction_manifest.get("channelId")
                        vid = vidID
                        break

                if channelId is None:
                    continue

            channel_data = reactors_inventory.get(channelId, None)
            assert channel_data is not None, (channelId, reaction_manifest, channel)

            featured = channel in song_def.get("featured", [])

            reactions[channel] = {
                "vid": vid,
                "channel": channel,
                "video_path": reaction_video_path,
                "aligned_path": os.path.join(
                    temp_directory, os.path.basename(channel) + f"-CROSS-EXPANDER.mp4"
                ),
                "alignment_metadata": os.path.join(
                    temp_directory, os.path.basename(channel) + f"-CROSS-EXPANDER.json"
                ),
                "featured": featured,
                "asides": None
                if options.get("skip_asides", False)
                else song_def.get("asides", {}).get(channel, None),
                "ground_truth": ground_truth,
                "ground_truth_path": ground_truth_path,
                "target_score": target_score,
                "manifest": reaction_manifest,
                "channel_data": channel_data,
            }

            if multiple_reactors is not None:
                reactions[channel]["num_reactors"] = multiple_reactors.get(channel, None)

            if foregrounded_backchannel is not None and foregrounded_backchannel.get(
                channel, False
            ):
                reactions[channel]["foregrounded_backchannel"] = foregrounded_backchannel.get(
                    channel
                )

            if backgrounded_backchannel is not None and backgrounded_backchannel.get(
                channel, False
            ):
                reactions[channel]["backgrounded_backchannel"] = backgrounded_backchannel.get(
                    channel
                )

            if fake_reactor_position is not None and fake_reactor_position.get(channel, False):
                reactions[channel]["fake_reactor_position"] = fake_reactor_position.get(channel)

            if start_reaction_search_at is not None and start_reaction_search_at.get(
                channel, False
            ):
                reactions[channel]["start_reaction_search_at"] = int(
                    start_reaction_search_at.get(channel) * sr
                )

            if end_reaction_search_at is not None and end_reaction_search_at.get(channel, False):
                reactions[channel]["end_reaction_search_at"] = (
                    end_reaction_search_at.get(channel) * sr
                )

            if unreliable_bounds is not None and unreliable_bounds.get(channel, False):
                reactions[channel]["unreliable_bounds"] = unreliable_bounds.get(channel)

            reactions[channel]["chunk_size"] = chunk_size.get(channel, 3)

            if manual_normalization_point is not None and manual_normalization_point.get(
                channel, False
            ):
                reactions[channel]["manual_normalization_point"] = manual_normalization_point.get(
                    channel
                )

            if insert_filler is not None and insert_filler.get(channel, False):
                reactions[channel]["insert_filler"] = insert_filler.get(channel)

            if mute is not None and mute.get(channel, False):
                reactions[channel]["mute"] = [
                    (int(s * sr), int(e * sr)) for s, e in mute.get(channel)
                ]

            if manual_bounds is not None and manual_bounds.get(channel, False):
                reactions[channel]["manual_bounds"] = manual_bounds.get(channel)
                reactions[channel]["manual_bounds"].sort(key=lambda x: x[0])

            if face_orientation is not None and face_orientation.get(channel, False):
                reactions[channel]["face_orientation"] = face_orientation.get(channel)

            if channel == "Resound":
                default_priority = 1
            else:
                default_priority = 40

            if featured:
                default_priority += 25

            if reactions[channel]["asides"] is not None:
                default_priority += 30

            # print("Priority", channel, priority.get(channel, default_priority))
            reactions[channel]["priority"] = priority.get(channel, default_priority)

            if swap_grid_positions is not None and swap_grid_positions.get(channel, False):
                reactions[channel]["swap_grid_positions"] = swap_grid_positions.get(channel)

            reactions[channel]["channel_label"] = channel_labels.get(channel, channel)
            if "Black Pegasus" in reactions[channel]["channel_label"]:
                reactions[channel]["channel_label"] = "Black Pegasus"

            metrics[channel] = {
                "subs": channel_data.get("subscriberCount", 1),
                "views": reaction_manifest.get("views", 1),
            }

        views = [1]
        subscribers = [1]
        for channel, reaction in reactions.items():
            views.append(metrics[channel]["views"])
            subscribers.append(metrics[channel]["subs"])

        max_views = max(views)
        max_subs = max(subscribers)
        for channel, reaction in reactions.items():
            my_views = metrics[channel]["views"]
            my_subs = metrics[channel]["subs"]

            metrics_bonus = max(my_subs / max_subs, my_views / max_views)

            metrics_bonus *= reaction["channel_data"].get("metric_bonus_multiplier", 1)

            reaction["priority"] += 20 * metrics_bonus

            # print(f"PRIORITY BONUS: {100*metrics_bonus:.1f} for {channel}")

        conf["reactions"] = reactions

    def load_reaction(channel):
        print_profiling()
        print(f"Loading reaction {channel}")
        reaction_conf = conf.get("reactions")[channel]

        if not conf.get("base_video_path"):
            load_base()

        if not "reaction_audio_path" in reaction_conf:
            reaction_video_path = reaction_conf["video_path"]
            reaction_audio_data, __, reaction_audio_path = extract_audio(reaction_video_path)

            normalization_factor = get_normalization_factor(reaction_conf)

            if normalization_factor != 1:
                normalized_reaction_audio_data = reaction_audio_data * normalization_factor
                reaction_conf["reaction_audio_data"] = normalized_reaction_audio_data
            else:
                reaction_conf["reaction_audio_data"] = reaction_audio_data

            reaction_conf["reaction_audio_path"] = reaction_audio_path

            output_dir = conf.get("reaction_directory")

            load_audio_transformations(
                reaction_conf,
                "reaction",
                output_dir,
                reaction_audio_path,
                reaction_audio_data,
            )

    def remove_reaction(channel):
        if channel in conf.get("reactions").get(channel, False):
            unload_reaction(channel)
            del conf["reactions"][channel]

    def free_conf():
        import gc

        for channel in conf.get("reactions", {}):
            remove_reaction(channel)

        temp_dir = conf.get("temp_directory")
        conf.clear()
        conf["temp_directory"] = temp_dir  # so that compilation lock can be lifted
        gc.collect()

    conf.update(
        {
            "load_base": load_base,
            "unload_base": unload_base,
            "load_reactions": load_reactions,
            "load_reaction": load_reaction,
            "remove_reaction": remove_reaction,
            "free_conf": free_conf,
        }
    )

    return False


def get_normalization_factor(reaction):
    normalization_dir = os.path.join(conf.get("song_directory"), "_normalization")
    if not os.path.exists(normalization_dir):
        os.makedirs(normalization_dir)

    normalization_file = os.path.join(
        normalization_dir, f"{reaction.get('channel')}-normalization.json"
    )

    if os.path.exists(normalization_file):
        normalization_factor = read_object_from_file(normalization_file)
        reaction["normalization_factor"] = normalization_factor
        return normalization_factor

    reaction_video_path = reaction["video_path"]
    reaction_audio_data, __, __ = extract_audio(reaction_video_path)

    if reaction.get("manual_normalization_point", False):
        (
            _,
            normalized_reaction_audio_data,
            normalization_factor,
        ) = normalize_audio_manually(
            conf.get("song_audio_data"),
            reaction_audio_data,
            reaction.get("manual_normalization_point"),
        )
    else:
        song_mfcc = librosa.feature.mfcc(
            y=conf.get("song_audio_data"),
            sr=sr,
            n_mfcc=conf.get("n_mfcc"),
            hop_length=conf.get("hop_length"),
        )
        reaction_mfcc = librosa.feature.mfcc(
            y=reaction_audio_data,
            sr=sr,
            n_mfcc=conf.get("n_mfcc"),
            hop_length=conf.get("hop_length"),
        )

        normalization_factor = normalize_reaction_audio(
            reaction,
            conf.get("song_audio_data"),
            reaction_audio_data,
            song_mfcc,
            reaction_mfcc,
        )

    reaction["normalization_factor"] = normalization_factor
    save_object_to_file(normalization_file, normalization_factor)
    return normalization_factor


to_delete = ["aligned_reaction_data", "mixed_audio"]
for source in ["_", "_vocals", "_accompaniment"]:
    for metric in [
        "data",
        "mfcc",
        "pitch_contour",
        "spectral_flux",
        "root_mean_square_energy",
    ]:  # _continuous_wavelet_transform
        to_delete.append(f"reaction_audio{source}{metric}")


def unload_reaction(channel):
    import gc

    global conf

    # print(f"Unloading reaction {channel}")

    reaction_conf = conf.get("reactions")[channel]

    for field in to_delete:
        if field in reaction_conf:
            del reaction_conf[field]

    gc.collect()


def load_audio_transformations(local_conf, prefix, output_dir, audio_path, audio_data):
    print_profiling()

    vocal_path_filename = "vocals-post-high-passed.wav"

    # vocals_path = os.path.join(separation_path, vocal_path_filename)

    # if not os.path.exists( vocals_path ):
    #     from backchannel_isolator.track_separation import separate_vocals
    #     separate_vocals(separation_path, audio_path, vocal_path_filename, duration=len(audio_data) / sr + 1)

    for source in [""]:  # 'accompaniment', 'vocals' ]:
        if source == "":
            data = audio_data
            path = audio_path
            source = "_"
        else:
            separation_path = os.path.join(
                output_dir, os.path.splitext(audio_path)[0].split("/")[-1]
            )
            path = os.path.join(separation_path, f"{source}.wav")
            data, _, _ = extract_audio(path, keep_file=source != "")
            source = f"_{source}_"

        local_conf[f"{prefix}_audio{source}path"] = path
        local_conf[f"{prefix}_audio{source}data"] = data

        mfcc = librosa.feature.mfcc(
            y=data, sr=sr, n_mfcc=conf.get("n_mfcc"), hop_length=conf.get("hop_length")
        )
        local_conf[f"{prefix}_audio{source}mfcc"] = mfcc

        # pc = pitch_contour(data, sr, hop_length=conf.get('hop_length'))
        # local_conf[f'{prefix}_audio{source}pitch_contour'] = pc

        # sf = spectral_flux(y=data, sr=sr, hop_length=conf.get('hop_length'))
        # local_conf[f'{prefix}_audio{source}spectral_flux'] = sf

        # rmse = root_mean_square_energy(y=data, sr=sr, hop_length=conf.get('hop_length'))
        # local_conf[f'{prefix}_audio{source}root_mean_square_energy'] = rmse


def unload_base():
    import gc

    global conf

    # print(f"Unloading reaction {channel}")

    to_delete_for_base = ["aligned_reaction_data", "mixed_audio"]
    for source in ["_", "_vocals", "_accompaniment"]:
        for metric in [
            "_data",
            "_mfcc",
            "_pitch_contour",
            "_spectral_flux",
            "_root_mean_square_energy",
        ]:  # _continuous_wavelet_transform
            to_delete_for_base.append(f"song_audio{source}{metric}")

    for field in to_delete_for_base:
        if field in conf:
            del conf[field]

    gc.collect()


def load_base():
    if "song_audio_data" in conf:
        return

    from silence import get_edge_silence

    song_directory = conf.get("song_directory")

    base_video_path_webm = os.path.join(song_directory, f"{os.path.basename(song_directory)}.webm")
    base_video_path_mp4 = os.path.join(song_directory, f"{os.path.basename(song_directory)}.mp4")

    #############
    # Make sure to remove silence at the beginning or end of the base video
    if os.path.exists(base_video_path_webm):
        if not os.path.exists(base_video_path_mp4):
            base_video = base_video_path_webm

            # Check if base_video is in webm format, and if corresponding mp4 doesn't exist, convert it
            base_video_name, base_video_ext = os.path.splitext(base_video_path_webm)

            start, end = get_edge_silence(base_video_path_webm)
            if start > sr / 5 or end > sr / 5:
                command = f'ffmpeg -i "{base_video_path_webm}" -y -vf "setpts=PTS" -q:v 40 -c:v h264_videotoolbox -r {conversion_frame_rate} -ar {conversion_audio_sample_rate} -ss {start} -to {end} "{base_video_path_mp4}"'
                subprocess.run(command, shell=True, check=True)
                os.remove(base_video_path_webm)
        else:
            os.remove(base_video_path_webm)

    if os.path.exists(base_video_path_mp4):
        base_video_path = base_video_path_mp4
    elif os.path.exists(base_video_path_webm):
        base_video_path = base_video_path_webm
    else:
        print("ERROR! base audio not found")
        raise Exception()

    if conf.get("base_video_transformations").get("flip", False):
        base_name, extension = os.path.splitext(base_video_path)
        original_video_path = base_video_path
        base_video_path = f"{base_name}-flipped{extension}".replace(".webm", ".mp4")

        if not os.path.exists(base_video_path):
            base_video = VideoFileClip(original_video_path)
            base_video = base_video.fx(vfx.mirror_x)
            base_video.write_videofile(
                base_video_path, codec="h264_videotoolbox", ffmpeg_params=["-q:v", "60"]
            )

    song_audio_data, _, base_audio_path = extract_audio(base_video_path, keep_file=True)
    song_audio_data = song_audio_data  # [0:150*sr]

    base_data = {
        "base_video_path": base_video_path,
        "song_audio_data": song_audio_data,
        "song_length": len(song_audio_data),
    }

    output_dir = conf.get("reaction_directory")

    load_audio_transformations(base_data, "song", output_dir, base_audio_path, song_audio_data)

    conf.update(base_data)


from moviepy.editor import VideoFileClip
from utilities import conversion_frame_rate
import subprocess


def prepare_reactions(convert_videos=[]):
    reaction_dir = conf.get("reaction_directory")

    print("Processing reactions in: ", reaction_dir)

    # Get all reaction video files
    mkv_videos = glob.glob(os.path.join(reaction_dir, "*.mkv"))

    # Convert all mkv videos to mp4 in the same directory and then delete the mkv videos

    for mkv_video in mkv_videos:
        print(f"HANDLING MKV {mkv_video}")
        # Strip existing video extension from filename
        base_name = os.path.basename(mkv_video)
        file_name_without_ext = os.path.splitext(base_name)[0]
        # If the stripped filename still has an extension, remove that too
        if any(ext in file_name_without_ext for ext in [".webm", ".mp4", ".mkv"]):
            file_name_without_ext = os.path.splitext(file_name_without_ext)[0]

        output_file = os.path.join(reaction_dir, file_name_without_ext + ".mp4")

        print(f"OUTPUT FILE {output_file}")
        ffmpeg.input(mkv_video).output(output_file).run()
        os.remove(mkv_video)

    # Process each reaction video

    for channel in convert_videos:
        webm_video = glob.glob(os.path.join(reaction_dir, f"{channel}.webm"))
        # mp4_video = glob.glob(os.path.join(reaction_dir, f"{channel}.mp4"))
        # react_video = (webm_video + mp4_video)[0]
        if len(webm_video) == 0:
            print(f"COULD NOT CONVERT {channel}")
            continue

        react_video = webm_video[0]

        react_video_name, react_video_ext = os.path.splitext(react_video)
        react_video_mp4 = react_video_name + ".mp4"
        if not os.path.exists(react_video_mp4):
            with VideoFileClip(react_video) as clip:
                width, height = clip.size
            resize_command = ""
            # if width > 1920 or height > 1080:
            #     # Calculate aspect ratio
            #     aspect_ratio = width / height
            #     if width > height:
            #         new_width = 1920
            #         new_height = int(new_width / aspect_ratio)
            #     else:
            #         new_height = 1080
            #         new_width = int(new_height * aspect_ratio)

            #     resize_command = f"-vf scale={new_width}:{new_height}"
            # Generate the ffmpeg command
            command = f'ffmpeg -y -i "{react_video}" {resize_command} -c:v libx264 -r {conversion_frame_rate} -ar {sr} -c:a aac "{react_video_mp4}"'

            print(command)
            subprocess.run(command, shell=True, check=True)
            if os.path.exists(react_video):
                os.remove(react_video)

    webm_videos = glob.glob(os.path.join(reaction_dir, "*.webm"))
    mp4_videos = glob.glob(os.path.join(reaction_dir, "*.mp4"))
    react_videos = webm_videos + mp4_videos + mkv_videos

    return react_videos
