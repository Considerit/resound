import cv2
import numpy as np
import subprocess
import os
import math
import gc

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

from compositor.layout import create_layout_for_composition, set_reactor_positions
from compositor.mix_audio import mix_audio
from compositor.zoom_and_pan import animateZoomPans
from compositor.asides import incorporate_asides_video, incorporate_asides_audio

from aligner.create_trimmed_video import check_compatibility


##########
# Monkey patch MoviePy VideoClip to provide a position attribute on a clip.
# Original set_position method
original_set_position = VideoClip.set_position

def new_set_position(self, pos, relative=False):
    # Call the original set_position method to ensure the clip is positioned correctly
    result = original_set_position(self, pos, relative)
    # Check if pos is a lambda (function) or not before updating my_position
    if not callable(pos):
        result.my_position = pos
    else:
        result.my_position = pos(0)

    result.my_position_is_dynamic = callable(pos)

    return result

# Monkey patch the set_position method
VideoClip.set_position = new_set_position

# Save the original __init__ method
original_init = VideoClip.__init__

def new_init(self, *args, **kwargs):
    # Call the original __init__ method
    original_init(self, *args, **kwargs)
    # Initialize a default position
    self.my_position = (0, 0)  # Default position could be (0,0) or any other default you prefer

# Monkey patch the __init__ method
VideoClip.__init__ = new_init





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




def create_reaction_concert(extend_by=0, output_size=(3840, 2160), shape="hexagon"): 
    conf.get('load_base')()

    output_path = conf.get('compilation_path')
    base_video_path = conf.get('base_video_path')


    if os.path.exists(output_path):
      print("Concert already exists", output_path)
      return

    print(f"Creating compilation for {output_path}")

    draft = conf.get('draft', False)

    base_video = VideoFileClip(base_video_path)

    border_width = 20  # Width of the border in pixels
    border_color = (255, 255, 255)  # Color of the border
    base_video = base_video.margin(border_width, color=border_color)


    width, height = output_size

    video_background = get_video_background(output_size, clip_length=base_video.duration + extend_by)

    base_video, cell_size, base_video_position = create_layout_for_composition(base_video, width, height, shape=shape)
    
    set_reactor_positions(cell_size)

    print("\tLayout created")

    ###########################
    # Create final audio track
    audio_clips, audible_segments = create_audio_clips(base_video, output_size)
    audio_output = compose_audio_clips(audio_clips)    
    audio_clips = []
    gc.collect()
    ###########################

    active_segments = find_active_segments(audible_segments)
    all_clips, clip_length, video_background = create_video_clips(base_video, video_background, cell_size, draft, output_size, audible_segments, shape=shape)
    

    print("\tClips created")

    if base_video_position is not None:
        base_video = all_clips[-1]['video'] = all_clips[-1]['video'].set_position(base_video_position)
        print(f"\t\tSetting position of base video {id(base_video)} in composition to ", base_video_position)
    my_clips = [c for c in all_clips if 'video' in c]
    my_clips.sort(key=lambda x: x['priority'])
    
    clips = []
    for clip in my_clips:
        clip = clip['video']
        if type(clip) == list:
            for c in clip:
                clips.append(c)
        else: 
            clips.append(clip)

    if video_background is not None:
        clips.insert(0, video_background)


    print("\tCreating channel labels")
    text_clips = create_channel_labels_video(active_segments, cell_size, output_size)
    clips += text_clips

    print("\tComposing clips")
    final_clip = compose_video_clips(base_video, clips, clip_length, extend_by, output_size)

    # final_clip = animateZoomPans(final_clip, show_viewport=False) #.set_duration(10)
    # final_clip.preview()
    # return

    # Unload a lot of the conf & reactions here to free memory
    conf['free_conf']()

    # visualize_clip_structure(final_clip)

    print("\tCreating tiles")
    tile_zoom = 1 # creates 4 tiles
    tiles = create_tiles(final_clip, output_size=output_size, zoom_level=tile_zoom)

    # tiles is a list of CompositeVideoClips 
    for idx, tile in enumerate(tiles):
        # tile = tile.set_duration(3)

        print(f"\tWriting Tile {idx} of {len(tiles)}")
        if idx == 0:
            tile_path = output_path
        else:
            tile_path = os.path.splitext(output_path)[0] + f'-tile-{idx}.mp4'            

        # Save the result
        if draft:
          tile_path = os.path.splitext(tile_path)[0] + "-draft.mp4"
          if not os.path.exists(tile_path):
            tile.set_fps(15).write_videofile(tile_path, 
                                     codec="h264_videotoolbox", 
                                     audio_codec="aac", 
                                     ffmpeg_params=['-q:v', '60'])

        else:
            tile.write_videofile(tile_path, 
                               codec="libx264", 
                               preset="slow",
                               ffmpeg_params=[ '-crf', '18' ]
                              )



    merge_audio_and_video(output_path, audio_output)



########
# Several issues arise when composing hundreds of reaction videos together in a CompositeVideoClip:

# 1) Memory and CPU performance issues when calling write_videofile

# 2) In post-production I’d like to be able to zoom in on particular parts of the concert and not have 
#    them be blurry because each was written out to a such a small area. However, I can't just write a 
#    higher resolution video because there is a C-library level limit we push at > 4k resolution in 
#    MoviePy.

# To address these issues, I'd like to be able to tile the CompositeVideoClip into several different tiles:

#   - There should be an option to figure out how many tiles. It will be constrained to powers of two. A value 
#     of 2 means that four tiles will be created, with the tile dividers vertically at x/2 and the horizontal 
#     division at y/2. Similarily, a value of 4 means 16 tiles will be created, with each dimension divided into 
#     4 equal parts. A value of 1 means no tiling.
#   - Each tile will maintain the aspect ratio of the original video. Furthermore, it will be resized to match 
#     the size of original video. 
#   - Each tile should cull subclips which are not at all visible in the tile
#   - Each write_videofile is called sequentially on each tile
#   - For each tile, each subclip's position will need to be translated from the original clips' coordinates 
#     to the tile's coordinates.

from copy import deepcopy

def create_tiles(composite_clip, output_size, zoom_level=1):
    # if zoom_level == 1:
    #     return [composite_clip]

    # Calculate the number of tiles
    tiles_count = 2 ** zoom_level
    tile_width = output_size[0] / zoom_level
    tile_height = output_size[1] / zoom_level

    # List to store each tile's CompositeVideoClip
    tiles = []
    clip_already_adjusted = {}


    for i in range(zoom_level):
        for j in range(zoom_level):
            # Calculate tile boundaries
            x_start = i * tile_width
            y_start = j * tile_height
            x_end   = x_start + tile_width
            y_end   = y_start + tile_height

            # print( f"Creating TILE {i}x{j} at ({x_start}, {y_start}) x ({x_end}, {y_end})"  )

            # Filter subclips visible within the current tile
            visible_clips = []
            for clip in composite_clip.clips:
                clip_id = id(clip)  # Unique identifier for the clip

                try:
                    clip_x, clip_y = clip.my_position
                except Exception as e:
                    print("ERROR! No my_position for ", id(clip))
                    clip_x, clip_y = (0,0)

                clip_width, clip_height = clip.size

                # Calculate the clip's bounding box
                clip_right = clip_x + clip_width
                clip_bottom = clip_y + clip_height

                # Check if the clip overlaps with the tile

                if clip_right >= x_start and clip_x <= x_end and clip_bottom >= y_start and clip_y <= y_end:
                    # Clone the clip if it has already been adjusted for another tile
                    # if clip_id in clip_already_adjusted:
                    #     clip = deepcopy(clip)
                    # else:
                    #     clip_already_adjusted[clip_id] = True

                    # Calculate the new position relative to the tile
                    new_x = clip_x - x_start
                    new_y = clip_y - y_start

                    # Apply translation
                    adjusted_clip = clip.set_position((new_x, new_y))

                    visible_clips.append(adjusted_clip)

            # Create a tile CompositeVideoClip with visible clips
            if len(visible_clips) > 0:
                tile_clip = CompositeVideoClip(visible_clips, size=(int(tile_width), int(tile_height))).resize(zoom_level).without_audio()
                tiles.append(tile_clip)

    # assert( tiles_count == len(tiles), tiles_count, tiles  )

    return tiles





def get_video_background(output_size, clip_length):
    video_background = None
    if conf.get('background'):
        reversing_bg = conf.get('background.reverse', False)
        video_background_path = conf.get('background')
        if video_background_path.endswith('mp4'):
            video_background = VideoFileClip(video_background_path).without_audio()

            if video_background.duration < clip_length:

                if not reversing_bg:
                    loops_required = int(np.ceil(clip_length / video_background.duration))
                    video_clips = [video_background] * loops_required
            
                else: 
                    def reverse_clip(clip):
                        frames = [frame for frame in clip.iter_frames()]
                        reversed_frames = frames[::-1]
                        return ImageSequenceClip(reversed_frames, fps=clip.fps)

                    loops_required = int(np.ceil(clip_length / (2 * video_background.duration)))
                    
                    video_clips = []
                    reversed_clip = None
                    for _ in range(loops_required):
                        video_clips.append(video_background)  # Forward clip
                        if reversed_clip is None:
                            reversed_clip = reverse_clip(video_background)
                        video_clips.append(reversed_clip)  # Backward clip

                video_background = concatenate_videoclips(video_clips)
        else: 
            video_background = ImageClip(video_background_path)

        video_background = video_background.resize(output_size).set_duration(clip_length)
    return video_background


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


def compose_audio_clips(audio_clips):

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

    print(f"\t\t\tMIXED {len(audio_clips)} audio clips together!")

    return audio_output







def compose_video_clips(base_video, clips, clip_length, extend_by, output_size):
    duration = max(base_video.duration, clip_length)

    final_clip = CompositeVideoClip(clips, size=output_size)
    final_clip = final_clip.set_duration(duration).without_audio().set_fps(30)

    return final_clip




def find_active_segments(audible_segments, duration_threshold=1.5):
    active_segments = {}
    final_segments = []


    for channel, segments in audible_segments.items():
        for (start, end, audio_factor) in segments:
            if end - start > duration_threshold * sr:
                print("\tSegment passes duration threshold", channel, start/sr, end/sr, audio_factor, (end-start)/sr)
                if audio_factor > .2:
                    print("\t\tSegment passes volume threshold", channel, start/sr, end/sr, audio_factor, (end-start)/sr)

                    final_segments.append((channel, start, end))

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
    progress_bar_height = 10 * 2 # Height of the progress bar


    font = "Fira-Sans-ExtraBold" # "BadaBoom-BB"

    for channel, start, end in active_segments:
        reaction = conf.get('reactions').get(channel)
        duration = (end - start) / sr



        # Create shadow text clip (black and slightly offset)
        shadow_clip = TextClip(reaction.get('channel_label'), fontsize=28*2, color='black', 
                               font=font)
        # Create shadow text clip (black and slightly offset)
        shadow_clip2 = TextClip(reaction.get('channel_label'), fontsize=28*2, color='black', 
                               font=font)

        # Create main text clip (white)
        txt_clip = TextClip(reaction.get('channel_label'), fontsize=28*2, color='white', 
                            font=font)

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
        # comp_clip = CompositeVideoClip([shadow_clip.set_position((x+1, y+1)), txt_clip.set_position((x, y))], 
        #                                size=output_size)
        # comp_clip = comp_clip.set_duration(duration).set_start(start / sr)
        # channel_textclips.append(comp_clip)

        shadow_clip  = shadow_clip.set_position((x+2, y+2)).set_duration(duration).set_start(start / sr)
        shadow_clip2 = shadow_clip2.set_position((x-1, y-1)).set_duration(duration).set_start(start / sr)

        txt_clip = txt_clip.set_position((x, y)).set_duration(duration).set_start(start / sr)

        channel_textclips.append(shadow_clip2)        
        channel_textclips.append(shadow_clip)        
        channel_textclips.append(txt_clip)

        # Calculate position for progress bar
        bar_position = (x, y + int(text_height * .75) + progress_bar_height + 4)

        # Create progress bar clip
        progress_bar_clip = create_progress_bar_clip(
            duration, txt_clip.size[0], progress_bar_height, start / sr
        )
        progress_bar_clip = progress_bar_clip.set_position(bar_position)

        channel_textclips.append(progress_bar_clip)


        # print(f"\tTEXT LABEL for {channel}: {start/sr} for {duration}")


    return channel_textclips



def create_video_clips(base_video, video_background, cell_size, draft, output_size, audible_segments, shape="hexagon"):


    reactor_colors = generate_hsv_colors(len(conf.get('reactions').keys()), 1, 1)
    clip_length = 0


    #####################
    # Create video clips

    base_video, video_background = incorporate_asides_video(base_video, video_background)

    all_clips = []

    print("\tCreating all video clips")

    for i, (name, reaction) in enumerate(conf.get('reactions').items()):
        # print(f"\t\tCreating video clip for {name}")
        reactors = reaction.get('reactors')
        if reactors is None:
            continue

        reaction_color = reactor_colors.pop()

        for idx, reactor in enumerate(reactors): 
            # print(f"\t\t\tReactor {idx}")

            clip = reactor['clip']
            size = reactor['size']
            featured = reaction['featured']

            clip = clip.resize((size, size))

            # print('\t\t\t\t...masking')
            clips = create_masked_video(channel=name,
                                       clip=clip, 
                                       audio_volume=reaction['audio_volume'],
                                       border_color=reaction_color, 
                                       audible_segments=audible_segments[name], 
                                       border_thickness=min(30, max(5, size / 15)), 
                                       width=size, 
                                       height=size, 
                                       shape= 'circle' if featured else shape)

            if reactor['layout_adjustments'].get('flip-x', False):
                clips[-1] = clips[-1].fx(vfx.mirror_x)

            position = reactor['position']
            for n, vid in enumerate(clips):
                clips[n] = vid.set_position(position)

            if clip_length < clip.duration:
              clip_length = clip.duration

            priority = reaction.get('priority')

            clip_info = {
              'channel': name,
              'reaction': reaction,
              'priority': priority,
              'video': clips, 
              'position': position,
              'reactor_idx': idx
            }

            all_clips.append(clip_info)
            # print('\t\t\t\t...done!')


    base_opacity = conf.get('base_video_transformations').get('opacity', False)
    if base_opacity:
        base_video = base_video.set_opacity(base_opacity)

    base_clip = {
      'base': True,
      'priority': 0,
      'video': base_video
    }

    all_clips.append(base_clip)

    if not conf['include_base_video']: # draft or 
        del base_clip['video']

    return all_clips, clip_length, video_background

downsample_factor = 100

def create_audio_clips(base_video, output_size):

    base_audio_clip, audible_segments = mix_audio(base_video, output_size)

    base_audio_clip = incorporate_asides_audio(base_video, base_audio_clip, audible_segments)

    all_clips = []

    print("\tCreating all audio clips")

    audio_clips = []
    for i, (name, reaction) in enumerate(conf.get('reactions').items()):
        print(f"\t\tCreating audio clip for {name}")

        audio = reaction['mixed_audio']
        audio_clips.append(audio)
        reaction['audio_volume'] = get_audio_volume(audio, downsample_factor=downsample_factor)   # I can probably downsample this pretty heavily and still have it accurate enough        
        del reaction['mixed_audio']


    audio_clips.append(base_audio_clip)

    return audio_clips, audible_segments










def create_mask_shape(shape, width, height, border_thickness):
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

    return mask_img_large, mask_img_small


precomputed_masks = {}
def create_border_clip_old(audio_volume, audible_segments, width, height, mask_img_large, mask_img_small, border_color, duration):

    def get_volume(t):
        idx = int(t * sr / downsample_factor)
        if idx < len(audio_volume):
            volume = audio_volume[idx]
        else: 
            volume = 0
        return volume

    # Subtract smaller mask from larger mask to create border mask
    border_mask = ImageChops.subtract(mask_img_large, mask_img_small)
    border_mask_np = np.array(border_mask)

    precomputed_mask_key = f"{height} - {width}"
    if precomputed_mask_key not in precomputed_masks:
        full_border_mask = np.zeros((height, width))
        full_border_mask[border_mask_np > 0] = 1  # Border visible

        white_border = np.zeros((height, width, 3))
        white_border[border_mask_np > 0] = np.array( [1,1,1]  ) * 255
        precomputed_masks[precomputed_mask_key] = {
            "no_border": np.zeros((height, width)),
            "visible_border": full_border_mask,
            "no_color_border": np.zeros((height, width, 3)),
            "white_border": white_border
        }

    # Each reaction video has its own bright saturated HSV color assigned to it       
    h, s, v = border_color
    my_r, my_g, my_b = colorsys.hsv_to_rgb(h, s, v)

    no_color_default = precomputed_masks[precomputed_mask_key]['no_color_border']
    white_border = precomputed_masks[precomputed_mask_key]['white_border']
    my_border = np.zeros((height, width, 3))
    my_border[border_mask_np > 0] = np.array( [my_r, my_g, my_b] ) * 255

    def make_border_frame(t):
        idx = int(t * sr)

        while len(audible_segments) > 0 and idx > audible_segments[0][1]:
            audible_segments.pop(0)

        in_foreground = len(audible_segments) > 0 and idx <= audible_segments[0][1] and audible_segments[0][2] > .2

        if in_foreground:
            return my_border
        elif get_volume(t) > 0:
            return white_border
        else: 
            return no_color_default


    empty_border_mask = precomputed_masks[precomputed_mask_key]["no_border"]
    full_border_mask = precomputed_masks[precomputed_mask_key]["visible_border"]
    def make_border_mask(t):

        if get_volume(t) > 0:
            return full_border_mask
        else:
            return empty_border_mask


    border_clip = VideoClip(make_border_frame, duration=duration)

    # Apply mask to the border_clip
    border_mask_clip = VideoClip(make_border_mask, ismask=True, duration=duration)
    border_clip = border_clip.set_mask(border_mask_clip)

    return border_clip




def create_border_clip(audio_volume, audible_segments, width, height, mask_img_large, mask_img_small, border_color, duration):


    # Subtract smaller mask from larger mask to create border mask
    border_mask = ImageChops.subtract(mask_img_large, mask_img_small)
    border_mask_np = np.array(border_mask)

    precomputed_mask_key = f"{height} - {width}"
    if precomputed_mask_key not in precomputed_masks:
        full_border_mask = np.zeros((height, width))
        full_border_mask[border_mask_np > 0] = 1  # Border visible

        white_border = np.zeros((height, width, 3))
        white_border[border_mask_np > 0] = np.array( [1,1,1]  ) * 255
        precomputed_masks[precomputed_mask_key] = {
            "no_border": np.zeros((height, width)),
            "visible_border": full_border_mask,
            "no_color_border": np.zeros((height, width, 3)),
            "white_border": white_border
        }

    def get_volume(t):
        idx = int(t * sr / downsample_factor)
        if idx < len(audio_volume):
            volume = audio_volume[idx]
        else: 
            volume = 0
        return volume

    no_color_default = precomputed_masks[precomputed_mask_key]['no_color_border']
    white_border = precomputed_masks[precomputed_mask_key]['white_border']

    def make_white_border(t):
        if get_volume(t) > 0:
            return white_border
        else: 
            return no_color_default

    empty_border_mask = precomputed_masks[precomputed_mask_key]["no_border"]
    full_border_mask = precomputed_masks[precomputed_mask_key]["visible_border"]
    def make_border_mask(t):
        if get_volume(t) > 0:
            return full_border_mask
        else:
            return empty_border_mask

    # Each reaction video has its own bright saturated HSV color assigned to it       
    h, s, v = border_color
    my_r, my_g, my_b = colorsys.hsv_to_rgb(h, s, v)
    my_border = np.zeros((height, width, 3))
    my_border[border_mask_np > 0] = np.array( [my_r, my_g, my_b] ) * 255

    clips = []
    for segment in audible_segments:
        start, end, volume = segment
        duration = (end-start) / sr

        if segment[2] > .2 and duration > 1.5:
            clip = ImageClip(my_border, duration=duration)
            border_mask_clip = ImageClip(full_border_mask, ismask=True)

        else:
            clip = VideoClip(make_white_border, duration=duration)
            border_mask_clip = VideoClip(make_border_mask, ismask=True)

        # Apply mask to the border_clip
        clip = clip.set_mask(border_mask_clip)
        clip = clip.set_start(start / sr)


        # print(f"\tBORDER CLIP: {start/sr} for {duration} ()")
        clips.append(clip)

    if clips:
        border_clip = clips
    else:
        border_clip = []  # Or an empty clip with duration set to the original video duration


    # print("BORDER CLIP COUNT", len(clips))
    return border_clip




# I’m using the following function to mask a video to a hexagon or circular shape, with 
# a border that pulses with color when there is audio from the reactor. 
def create_masked_video(channel, clip, audio_volume, width, height, audible_segments, border_color, border_thickness=10, shape="hexagon"):

    audible_segments = audible_segments.copy()

    mask_img_large, mask_img_small = create_mask_shape(shape, width, height, border_thickness)

    # print(f"MAKING MASKED VIDEO FOR {channel}")
    clips = create_border_clip(audio_volume, audible_segments, width, height, mask_img_large, mask_img_small, border_color, clip.duration)

    # Create video clip by applying the smaller mask to the original video clip
    mask_img_small_np = np.array(mask_img_small)    
    clip = clip.set_mask(ImageClip(mask_img_small_np, ismask=True))

    clips.append(clip)
    return clips


def get_audio_volume(audio_array, downsample_factor=None):
    """Calculate the volume of the audio clip and optionally downsample to a smaller sample rate by averaging.
    """


    if downsample_factor is not None and downsample_factor != 1:
        # Calculate the downsampling factor
        factor = downsample_factor
        
        # Determine the length of the resulting downsampled audio
        downsampled_length = audio_array.shape[0] // factor
        
        # Initialize the downsampled audio array
        downsampled_audio = np.zeros((downsampled_length, audio_array.shape[1]))
        
        for i in range(downsampled_length):
            start_index = i * factor
            end_index = start_index + factor
            # Average the chunks of the original audio array
            downsampled_audio[i, :] = np.mean(audio_array[start_index:end_index, :])
        
        audio = downsampled_audio

    else:
        audio = audio_array

    # Calculate volume    
    audio_volume = np.sqrt(np.mean(np.square(audio), axis=1))  # RMS amplitude
    audio_volume /= np.max(audio_volume)  # normalize to range [0, 1]

    return audio_volume



def generate_hsv_colors(n, s, v):
    """Generates n evenly distributed HSV colors with the same S and V"""
    return [(i/n, s, v) for i in range(n)]








def visualize_clip_structure(clip, depth=0):
    indent = "  " * depth  # Indentation to represent depth

    if isinstance(clip, CompositeVideoClip):
        print(f"{indent}CompositeVideoClip [{id(clip)}]:")
        for subclip in clip.clips:
            visualize_clip_structure(subclip, depth + 1)
    else:
        # Handle other clip types here
        print(f"{indent}{type(clip).__name__} [{id(clip)}]")


