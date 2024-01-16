import cv2
import numpy as np
import subprocess
import os
import math
from PIL import Image, ImageDraw, ImageChops
import colorsys


from moviepy.editor import ImageClip, TextClip, VideoFileClip, CompositeVideoClip, ImageSequenceClip
from moviepy.editor import concatenate_videoclips, vfx, clips_array

from moviepy.video.VideoClip import VideoClip, ColorClip
from moviepy.audio.AudioClip import AudioClip

from moviepy.audio.fx import all as audio_fx
from moviepy.video.fx import fadeout
from moviepy.video.fx.all import resize

from moviepy.video.tools.drawing import color_gradient



import soundfile as sf

from utilities import unload_reaction, conf, conversion_frame_rate, conversion_audio_sample_rate as sr

from compositor.layout import create_layout_for_composition
from compositor.mix_audio import mix_audio
from compositor.zoom_and_pan import animateZoomPans
from compositor.asides import incorporate_asides

from aligner.create_trimmed_video import check_compatibility


# I have a base video, and then any number of videos of people reacting to that base video. I would 
# like to use MoviePy to compose a video that has all of these videos together. 

# The most difficult part of this will probably be the layout, given that:
#   1) There can be any number of reaction videos
#   2) There won't always be a base video

# We want to find a visually attractive layout that:
#   1) When there is a base video, it should be shown prominently. Its size shouldn't depend on the 
#      number of reactions. Perhaps its size should be invariate to the total space allocated to 
#      the reactions, say at least 25% of the space allocated to reactions (or some other percentage). 
#      Its position can change though, depending on aesthetic considerations given the number of reactions. 
#   2) The individual reactions should be sized to fit the layout.  
#   3) There can be any number of layout patterns used. There could be a base video in the middle, with 
#      reactions surrounding it. The base video could be at the top, with reactions to the bottom, 
#      or the bottom and the sides. etc. etc. I'm particularly interested in using a honeycomb layout 
#      for the reactions, to create interlocking hexagon pattern. I'm open to other layout styles.  
#   4) Each reaction video will also provide data about the reactor's orientation in the video 
#      (like, looking to the right or left). The algorithm should use this information to try to 
#      put each reaction video in a place where it looks like reactor is looking toward the base 
#      video. This is a soft constraint. 

# The aspect ration of the resulting video should be approximately that of a laptop screen, and can 
# be any size up to the resolution of a modern macbook pro. These constraints are soft. 



def compose_reactor_compilation(extend_by=0, output_size=(1920, 1080), shape="hexagon"):
    conf.get('load_base_video')()

    output_path = conf.get('compilation_path')
    base_video_path = conf.get('base_video_path')

    if os.path.exists(output_path):
      print("Compilation already exists", output_path)
      return

    print(f"Creating compilation for {output_path}")

    draft = conf.get('draft', False)

    base_video = VideoFileClip(base_video_path)
    if conf.get('base_video_transformations').get('flip', False):
        # Flip the clip horizontally
        base_video = base_video.fx(vfx.mirror_x)
        # base_video.write_videofile("flipped_video.mp4")


    width, height = output_size

    base_video, cell_size, base_video_position = create_layout_for_composition(base_video, width, height, shape=shape)
    print("\tLayout created")
    all_clips, audio_clips, clip_length, audio_scaling_factors = create_clips(base_video, cell_size, draft, output_size, shape=shape)
    print("\tClips created")

    if base_video_position is not None:
        all_clips[-1]['video'] = all_clips[-1]['video'].set_position(base_video_position)
        print("Setting position of base video in composition to ", base_video_position)
    clips = [c for c in all_clips if 'video' in c]
    clips.sort(key=lambda x: x['priority'])
    clips = [c['video'] for c in clips]

    active_segments = find_active_segments(audio_scaling_factors)
    text_clips = create_channel_labels_video(active_segments, cell_size, output_size)
    clips += text_clips





    final_clip, audio_output = compose_clips(base_video, clips, audio_clips, clip_length, extend_by, output_size)
    

    
    print("\tClips composed")



    # pre_background = final_clip

    if conf.get('background'):
        reversing_bg = conf.get('background.reverse', False)
        video_background_path = conf.get('background')
        if video_background_path.endswith('mp4'):
            video_background = VideoFileClip(video_background_path).without_audio()

            if video_background.duration < final_clip.duration:

                if not reversing_bg:
                    loops_required = int(np.ceil(final_clip.duration / video_background.duration))
                    video_clips = [video_background] * loops_required
            
                else: 
                    def reverse_clip(clip):
                        frames = [frame for frame in clip.iter_frames()]
                        reversed_frames = frames[::-1]
                        return ImageSequenceClip(reversed_frames, fps=clip.fps)

                    loops_required = int(np.ceil(final_clip.duration / (2 * video_background.duration)))
                    
                    video_clips = []
                    reversed_clip = None
                    for _ in range(loops_required):
                        video_clips.append(video_background)  # Forward clip
                        if reversed_clip is None:
                            reversed_clip = reverse_clip(video_background)
                        video_clips.append(reversed_clip)  # Backward clip

                video_background = concatenate_videoclips(video_clips)
        else: 
            video_background = ImageClip(video_background_path).set_duration(final_clip.duration)


        # Set the final_clip as a layer on top of the background
        final_clip = CompositeVideoClip([
            video_background.resize(output_size).set_duration(final_clip.duration),
            final_clip
        ])



    final_clip = animateZoomPans(final_clip, show_viewport=False) #.set_duration(10)
    # final_clip.preview()
    # return

    # Unload a lot of the conf & reactions here to free memory
    audio_clips = []
    conf['free_conf']()

    # Save the result
    if draft:
      output_path = output_path + "fast.mp4"
      if not os.path.exists(output_path):
        final_clip.resize(.25).set_fps(12).write_videofile(output_path, 
                                         codec="h264_videotoolbox", 
                                         audio_codec="aac", 
                                         ffmpeg_params=['-q:v', '10'], 
                                         preset='ultrafast')

    else:
        final_clip.write_videofile(output_path, 
                                 codec="h264_videotoolbox", 
                                 ffmpeg_params=['-q:v', '60']
                                )


        # try: 
        #     pre_background.write_videofile(output_path+'.mov', 
        #                    codec="prores_ks", 
        #                    ffmpeg_params=['-profile:v', '4444', '-q:v', '60', '-pix_fmt', 'yuv444p10le']
        #                   )
        # except:
        #     print("Could not export transparent")


    merge_audio_and_video(output_path, audio_output)


def merge_audio_and_video(output_path, audio_output):
    command = [
        'ffmpeg',
        '-i', output_path,
        '-i', audio_output,
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-shortest',
        output_path + '.mp4'
    ]

    subprocess.run(command)

    command = [
        'mv',
        output_path + '.mp4',
        output_path
    ]    
    subprocess.run(command)

def compose_clips(base_video, clips, audio_clips, clip_length, extend_by, output_size):
    duration = max(base_video.duration, clip_length)
    # duration = 30 


    final_clip = CompositeVideoClip(clips, size=output_size)
    final_clip = final_clip.set_duration(duration)

    # Determine the maximum length among all audio clips
    max_len = max([track.shape[0] for track in audio_clips])

    # Check if the audio clips are stereo (2 channels)
    if any(track.shape[1] != 2 for track in audio_clips if len(track.shape) > 1):
        raise ValueError("All tracks must be stereo (2 channels)")

    # Initialize the mixed_track with zeros
    mixed_track = np.zeros((max_len, 2), dtype=np.float64)

    for i, track in enumerate(audio_clips):
        # Pad the track if it's shorter than the longest one
        pad_length = max_len - track.shape[0]

        track_padded = np.pad(track, ((0, pad_length), (0, 0)), 'constant')

        # Mix the tracks (simple averaging)
        mixed_track += track_padded

        # sf.write(os.path.join( conf.get('temp_directory'),  f"MAIN-full-mixed-directly-{i}.wav"   ), track_padded, sr)

    audio_output = os.path.join( conf.get('temp_directory'),  f"MAIN-full-mixed-directly.wav"   )
    sf.write(audio_output, mixed_track, sr)

    audio_output = os.path.join(conf.get('temp_directory'), "MAIN-full-mixed-directly.flac")
    sf.write(audio_output, mixed_track, sr, format='FLAC')

    print(f"MIXED {len(audio_clips)} audio clips together!")


    # if extend_by > 0:
    #   clip1 = final_audio.subclip(0, duration - extend_by)
    #   clip2 = final_audio.subclip(duration - extend_by, duration)

    #   # Reduce the volume of the second clip
    #   clip2 = clip2.fx(audio_fx.volumex, 0.5)  # reduce volume to 50%
    #   final_audio = concatenate_audioclips([clip1, clip2])



    # final_clip = final_clip.set_audio(final_audio)
    final_clip = final_clip.without_audio()

    final_clip = final_clip.set_fps(30)

    # final_clip = final_clip.set_duration(10)

    return final_clip, audio_output




def find_active_segments(audio_scaling_factors, duration_threshold=1.5):
    global sr  # Assuming sr is defined globally
    active_segments = {}
    final_segments = []

    # Step 1: Identify Qualifying Segments for Each Channel
    for channel, audio_mask in audio_scaling_factors.items():
        start_sample = None
        active_segments[channel] = []

        for i, value in enumerate(audio_mask):
            if value == 1:
                if start_sample is None:
                    start_sample = i  # Start of a new active segment
            else:
                if start_sample is not None and i - start_sample >= duration_threshold * sr:
                    active_segments[channel].append((start_sample, i - 1))
                start_sample = None  # Reset for the next segment

        # Check if the last segment reaches the end of the array
        if start_sample is not None and len(audio_mask) - start_sample >= duration_threshold * sr:
            active_segments[channel].append((start_sample, len(audio_mask) - 1))

    # Step 2: Modify and Check Segments for Overlaps
    for channel, segments in active_segments.items():
        for segment in segments:
            segment_fragments = [segment]

            for other_channel, other_segments in active_segments.items():
                if other_channel != channel:
                    for other_segment in other_segments:
                        new_fragments = []
                        for fragment in segment_fragments:
                            start, end = fragment
                            other_start, other_end = other_segment

                            # Check for overlap and split the segment if necessary
                            if not (end < other_start or start > other_end):
                                if start < other_start:
                                    new_fragments.append((start, other_start - 1))
                                if end > other_end:
                                    new_fragments.append((other_end + 1, end))
                            else:
                                new_fragments.append(fragment)

                        segment_fragments = new_fragments

            # Add non-overlapping fragments that meet the duration threshold to the final list
            for fragment in segment_fragments:
                if fragment[1] - fragment[0] >= duration_threshold * sr:
                    final_segments.append((channel, fragment[0], fragment[1]))


    summative = {}
    for channel, start, end in final_segments:
        if channel not in summative:
            summative[channel] = 0

        summative[channel] += end - start

    most_featured = [ (c, t) for c,t in summative.items()   ]
    most_featured.sort( key=lambda x: x[1], reverse=True)
    print("Featured Time by Channel")
    for c,t in most_featured:
        print(c, t/sr)
    for c,t in most_featured:
        print(c)


    return final_segments


from moviepy.editor import ImageSequenceClip

def create_progress_bar_clip(duration, width, height, start_time, bar_color=(0, 255, 0)):
    """ Creates a progress bar clip using a lazy frame generation method. """
    def make_frame(t):
        progress = t / duration
        new_width = int(progress * width)
        bar_frame = np.zeros((height, width, 3), dtype=np.uint8)
        bar_frame[:, :new_width, :] = bar_color
        return bar_frame

    bar_clip = VideoClip(make_frame, duration=duration).set_start(start_time)
    return bar_clip


def create_channel_labels_video(active_segments, cell_size, output_size):
    channel_textclips = []
    progress_bar_height = 10  # Height of the progress bar

    print('creating channel label video')
    for channel, start, end in active_segments:
        reaction = conf.get('reactions').get(channel)
        duration = (end - start) / sr

        # Create shadow text clip (black and slightly offset)
        shadow_clip = TextClip(reaction.get('channel_label'), fontsize=24, color='black', 
                               font="Fira-Sans-ExtraBold")

        # Create main text clip (white)
        txt_clip = TextClip(reaction.get('channel_label'), fontsize=24, color='white', 
                            font="Fira-Sans-ExtraBold")

        text_width, text_height = txt_clip.size

        # Determine text position
        reactors = reaction.get('reactors')
        xs = [r['position'][0] for r in reactors]
        x = min(xs) + (max(xs) - min(xs)) // 2  + cell_size // 2
        y = reactors[0]['position'][1]

        x -= text_width // 2

        if y >= output_size[1] + cell_size + text_height + progress_bar_height + 4 + 4:  # Bottom row
            y = y - text_height // 2 - progress_bar_height - 4 - 4  # Place above the channel
        else:
            y += cell_size + text_height // 2 - progress_bar_height - 4 - 4 # Place below the channel

        # Adjust position if at an edge
        x = max(min(x, output_size[0] - text_width  // 2), 10)
        y = max(min(y, output_size[1] - text_height // 2 - progress_bar_height - 4), 10)

        # Composite the shadow and text clips
        comp_clip = CompositeVideoClip([shadow_clip.set_position((x+1, y+1)), txt_clip.set_position((x, y))], 
                                       size=output_size)
        comp_clip = comp_clip.set_duration(duration).set_start(start / sr)

        channel_textclips.append(comp_clip)

        # Calculate position for progress bar
        bar_position = (x, y + text_height // 2 + progress_bar_height + 4)

        # Create progress bar clip
        progress_bar_clip = create_progress_bar_clip(
            duration, txt_clip.size[0], progress_bar_height, start / sr
        )
        progress_bar_clip = progress_bar_clip.set_position(bar_position)

        channel_textclips.append(progress_bar_clip)

    return channel_textclips







def create_clips(base_video, cell_size, draft, output_size, shape="hexagon"):


    reactor_colors = generate_hsv_colors(len(conf.get('reactions').keys()), 1, .6)
    clip_length = 0




    ###################
    # Set position
    print("\tSetting all positions")    
    for i, (name, reaction) in enumerate(conf.get('reactions').items()):
        print(f"\t\tSetting position for {name}")
        reactors = reaction.get('reactors')
        if reactors is None:
            continue

        for idx, reactor in enumerate(reactors): 
            x,y = reactor['grid_assignment']

            size = cell_size
            if reaction['featured']: 
              size *= 1.15
              size = int(size)

            position = (x - size / 2, y - size / 2)

            reactor['position'] = position
            reactor['size'] = size

    #####################
    # Mix audio

    base_audio_clip, audio_scaling_factors = mix_audio(base_video, output_size)


    #####################
    # Create video clips

    base_video, base_audio_clip = incorporate_asides(base_video, base_audio_clip, audio_scaling_factors)
    all_clips = []

    print("\tCreating all clips")
    audio_clips = []
    for i, (name, reaction) in enumerate(conf.get('reactions').items()):
        print(f"\t\tCreating clip for {name}")
        reactors = reaction.get('reactors')
        if reactors is None:
            continue

        # if i > 1:
        #     continue



        reaction_color = reactor_colors.pop()

        audio = reaction.get('mixed_audio')
        audio_clips.append(audio)
        # print(f"MIXED AUDIO LEN {name}={len(audio) / sr}")
        audio_volume = get_audio_volume(audio)

        for idx, reactor in enumerate(reactors): 
            print(f"\t\t\tReactor {idx}")

            clip = reactor['clip']

            size = reactor['size']

            featured = reaction['featured']

            clip = clip.resize((size, size))
            if not draft:
                print('\t\t\t\t...masking')
                clip = create_masked_video(channel=name,
                                           clip=clip, 
                                           audio_volume=audio_volume,
                                           border_color=reaction_color, 
                                           audio_scaling_factor=audio_scaling_factors[name], 
                                           border_thickness=min(30, max(5, size / 15)), 
                                           width=size, 
                                           height=size, 
                                           shape= 'circle' if featured else shape)

            position = reactor['position']

            # Flip the clip horizontally if reactor was placed on the wrong side

            horiz_gaze, vert_gaze = reaction.get('face_orientation')
            if (horiz_gaze ==  'left' and reactor['position'][0] < output_size[0] / 2) or \
               (horiz_gaze == 'right' and reactor['position'][0] > output_size[0] / 2):

                if reaction.get('channel') == 'Second Hand Reactions ' and conf.get('song_name') == 'Hero':
                    print("skipping the flip")
                else:
                    clip = clip.fx(vfx.mirror_x)

            clip = clip.set_position(position)

            if clip_length < clip.duration:
              clip_length = clip.duration

            priority = reaction.get('priority')

            clip_info = {
              'channel': name,
              'reaction': reaction,
              'priority': priority,
              'video': clip,
              'position': position,
              'reactor_idx': idx
            }

            all_clips.append(clip_info)
            print('\t\t\t\t...done!')





    base_opacity = conf.get('base_video_transformations').get('opacity', False)
    if base_opacity:
        base_video = base_video.set_opacity(base_opacity)


    audio_clips.append(base_audio_clip)
    base_clip = {
      'base': True,
      'priority': 0,
      'video': base_video
    }

    all_clips.append(base_clip)


    if draft or not conf['include_base_video']:
        del base_clip['video']



    return all_clips, audio_clips, clip_length, audio_scaling_factors










# I’m using the following function to mask a video to a hexagon or circular shape, with 
# a border that pulses with color when there is audio from the reactor. 

def create_masked_video(channel, clip, audio_volume, width, height, audio_scaling_factor, border_color, border_thickness=10, shape="hexagon"):

    # Create new PIL images with the same size as the clip, fill with black color
    mask_img_large = Image.new('1', (width, height), 0)
    mask_img_small = Image.new('1', (width, height), 0)
    draw_large = ImageDraw.Draw(mask_img_large)
    draw_small = ImageDraw.Draw(mask_img_small)

    if shape == 'circle':
        # Draw larger and smaller circles on the mask images
        draw_large.ellipse([(0, 0), (width, height)], fill=1)
        draw_small.ellipse([(border_thickness, border_thickness), ((width - border_thickness), (height - border_thickness))], fill=1)
    elif shape == 'diamond':
        def calc_diamond_vertices(cx, cy, size):
            return [(cx, cy - size / 2),  # Top
                    (cx + size / 2, cy),  # Right
                    (cx, cy + size / 2),  # Bottom
                    (cx - size / 2, cy)]  # Left

        vertices_large = calc_diamond_vertices(width / 2, height / 2, min(width, height))
        vertices_small = calc_diamond_vertices(width / 2, height / 2, min(width, height) - border_thickness)

        # Draw the larger and smaller diamonds on the mask images
        draw_large.polygon(vertices_large, fill=1)
        draw_small.polygon(vertices_small, fill=1)
    elif shape == 'hexagon':
        assert(height == width)

        def calc_hexagon_vertices(cx, cy, size):
            relative_to = 0
            l = size / 2 #size / math.sqrt(3) # side length



            vertices = [  
              [cx,            cy - size / 2],  # center top
              [cx + size / 2, cy - l / 2],     # right top
              [cx + size / 2, cy + l / 2],     # right bottom
              [cx,            cy + size / 2],  # center bottom                                                    
              [cx - size / 2, cy + l / 2],     # left bottom
              [cx - size / 2, cy - l / 2]]     # left top


            vertices = [ (v[0] + relative_to, v[1] + relative_to) for v in vertices]


            return vertices

        vertices_large = calc_hexagon_vertices(width / 2, height / 2, width)
        vertices_small = calc_hexagon_vertices(width / 2, height / 2, width - border_thickness)


        # Draw the larger and smaller hexagons on the mask images
        draw_large.polygon(vertices_large, fill=1)
        draw_small.polygon(vertices_small, fill=1)

    # Subtract smaller mask from larger mask to create border mask
    border_mask = ImageChops.subtract(mask_img_large, mask_img_small)
    border_mask_np = np.array(border_mask)
    mask_img_small_np = np.array(mask_img_small)


    # I’d like to make the border dynamic. Specifically:
    #   - Each reaction video should have its own bright saturated HSV color assigned to it
    #   - The HSV color for the respective reaction video should be mixed with white, 
    #     inversely proportional to the volume of the track at that timestamp. That is, 
    #     when there is no volume, the border should be white, and when it is at its 
    #     loudest, it should be the HSV color.
    def colorize_when_backchannel_active(hsv_color, clip):
        h, s, v = hsv_color
        

        def color_func(t):
            # Get the volume at current time
            idx = int(t * sr)

            if idx < len(audio_volume):
                volume = audio_volume[idx]
            else: 
                volume = 0

            if volume == 0:
                return 0, 0, 0

            

            if idx < len(audio_scaling_factor) and audio_scaling_factor[idx] > .5:
                v_modulated = 1 - volume * (1 - v)
                r, g, b = colorsys.hsv_to_rgb(h, s, v_modulated)
            else: 
                return 1, 1, 1


            # # Calculate the interpolated V value based on audio volume
            # if len(audio_scaling_factor) > idx: 
            #     v_modulated = .5 * v + .5 * v * audio_scaling_factor[idx]
            # else:
            #     v_modulated = v
            # # v_modulated = 1 - volume * (1 - v)

            # # Convert modulated HSV color back to RGB
            # r, g, b = colorsys.hsv_to_rgb(h, s, v_modulated)


            return r, g, b

        return color_func

    border_func = colorize_when_backchannel_active(border_color, clip)


    def make_frame(t):
        img = np.zeros((height, width, 3))

        # Convert the color from HSV to RGB, then scale from 0-1 to 0-255
        color_rgb = np.array(border_func(t)) * 255
        img[border_mask_np > 0] = color_rgb  # apply color to border

        return img

    def make_mask(t):
        mask = np.zeros((height, width))
        idx = int(t * sr)

        # Get the volume at current time
        if (idx < len(audio_volume)):
            volume = audio_volume[idx]
        else: 
            volume = 0

        if volume > 0:
            mask[border_mask_np > 0] = 1  # Border visible

        return mask

    border_clip = VideoClip(make_frame, duration=clip.duration)

    # Apply mask to the border_clip
    border_mask_clip = VideoClip(make_mask, ismask=True, duration=clip.duration)
    border_clip = border_clip.set_mask(border_mask_clip)

    # Create video clip by applying the smaller mask to the original video clip
    clip = clip.set_mask(ImageClip(mask_img_small_np, ismask=True))

    # Overlay the video clip on the border clip
    final_clip = CompositeVideoClip([border_clip, clip])

    return final_clip




def get_audio_volume(audio_array, fps=None):
    if fps is None:
      fps = sr
      
    """Calculate the volume of the audio clip"""
    audio = audio_array
    audio_volume = np.sqrt(np.mean(np.square(audio), axis=1))  # RMS amplitude
    audio_volume /= np.max(audio_volume)  # normalize to range [0, 1]
    return audio_volume



def generate_hsv_colors(n, s, v):
    """Generates n evenly distributed HSV colors with the same S and V"""
    return [(i/n, s, v) for i in range(n)]





