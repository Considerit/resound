import cv2
import numpy as np
import subprocess
import os
import math
from PIL import Image, ImageDraw, ImageChops
import colorsys


from moviepy.editor import ImageClip, CompositeVideoClip, CompositeAudioClip, concatenate_audioclips, concatenate_videoclips
from moviepy.audio.AudioClip import AudioArrayClip, AudioClip
from moviepy.video.VideoClip import VideoClip, ColorClip
from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import crop
from moviepy.audio.fx import all as audio_fx

from utilities import conf, conversion_frame_rate, conversion_audio_sample_rate

from compositor.layout import create_layout_for_composition

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


def compose_reactor_compilation(extend_by=0, output_size=(1792, 1120)):
    conf.get('load_base_video')()

    output_path = conf.get('compilation_path')
    base_video_path = conf.get('base_video_path')

    if os.path.exists(output_path):
      print("Compilation already exists", output_path)
      return

    print(f"Creating compilation for {output_path}")

    draft = conf.get('draft', False)

    base_video = VideoFileClip(base_video_path)

    width, height = output_size

    base_video, cell_size = create_layout_for_composition(base_video, width, height)
    print("\tLayout created")
    all_clips, clip_length = create_clips(base_video, cell_size, draft)
    print("\tClips created")

    clips = [c for c in all_clips if 'video' in c]
    clips.sort(key=lambda x: x['priority'])
    clips = [c['video'] for c in clips]
    
    audio_clips = [c['audio'] for c in all_clips ]

    final_clip = compose_clips(base_video, clips, audio_clips, clip_length, extend_by, output_size)
    print("\tClips composed")

    # Save the result
    if draft:
      fast_path = output_path + "fast.mp4"
      if not os.path.exists(fast_path):
        final_clip.resize(.25).set_fps(12).write_videofile(output_path + "fast.mp4", 
                                         codec="h264_videotoolbox", 
                                         audio_codec="aac", 
                                         ffmpeg_params=['-q:v', '10'], 
                                         preset='ultrafast')

    else:
      final_clip.write_videofile(output_path, codec="h264_videotoolbox", audio_codec="aac", ffmpeg_params=['-q:v', '40'])



def compose_clips(base_video, clips, audio_clips, clip_length, extend_by, output_size):
    duration = max(base_video.duration, clip_length)
    # duration = 30 

    final_clip = CompositeVideoClip(clips, size=output_size)
    final_clip = final_clip.set_duration(duration)

    final_audio = CompositeAudioClip(audio_clips)
    final_audio = final_audio.set_duration(duration)        

    if extend_by > 0:
      clip1 = final_audio.subclip(0, duration - extend_by)
      clip2 = final_audio.subclip(duration - extend_by, duration)

      # Reduce the volume of the second clip
      clip2 = clip2.fx(audio_fx.volumex, 0.5)  # reduce volume to 50%
      final_audio = concatenate_audioclips([clip1, clip2])

    final_clip = final_clip.set_audio(final_audio)
    final_clip.set_fps(30)

    return final_clip


def create_clips(base_video, cell_size, draft):

    total_reactors = 0 
    for name, reaction in conf.get('reactions').items():
      total_reactors += len(reaction.get('reactors'))

    reactor_colors = generate_hsv_colors(total_reactors, 1, .6)
    clip_length = 0

    all_clips = []

      

    base_video = incorporate_asides(base_video)

    base_video.audio.fps = conversion_audio_sample_rate
    base_audio_as_array = base_video.audio.to_soundarray()


    for name, reaction in conf.get('reactions').items():
        print(f"\t\tCreating clip for {name}")
        reactors = reaction.get('reactors')


        positions = []
        for idx, reactor in enumerate(reactors): 
            print(f"\t\t\tReactor {idx}")

            reactor_color = reactor_colors.pop()
            x,y = reactor['grid_assignment']

            featured = reaction['featured']

            clip = reactor['clip']

            clip.audio.fps = conversion_audio_sample_rate  
            volume_adjusted_audio = match_audio_peak(base_audio_as_array, clip.audio.to_soundarray(), factor=1)
            volume_adjusted_clip = AudioArrayClip(volume_adjusted_audio, fps=clip.audio.fps)

            size = cell_size
            if featured: 
              size *= 1.15
              size = int(size)

            clip = clip.resize((size, size))
            if not draft:
              clip = create_masked_video(clip, border_color=reactor_color, border_thickness=10, width=size, height=size, as_circle=featured)

            position = (x - size / 2, y - size / 2)
            clip = clip.set_position(position)

            if clip_length < clip.duration:
              clip_length = clip.duration

            if featured:
              priority = 10
            else: 
              priority = 1

            clip_info = {
              'channel': name,
              'reaction': reaction,
              'priority': priority,
              'video': clip,
              'audio': volume_adjusted_clip,
              'position': position,
              'reactor_idx': idx
            }

            all_clips.append(clip_info)
            positions.append(position)




    base_clip = {
      'audio': base_video.audio,
      'base': True,
      'priority': 0,
      'video': base_video
    }


    all_clips.append(base_clip)


    if draft or not conf['include_base_video']:
        del base_clip['video']

    return all_clips, clip_length


# Any of the reaction clips can have any number of "asides". An aside is a bonus 
# video clip spliced into a specific point in the respective reaction video clip. 
# When an aside is active, only the respective video clip is playing, and all the 
# other clips are paused. When a clip is paused because an aside is playing, the 
# previous frame is replicated until the aside is finished, and no audio is played. 



def incorporate_asides(base_video):

    all_asides = []
    for name, reaction in conf.get('reactions').items():
        reactors = reaction.get('reactors')

        if reaction.get('aside_clips', None):
            for insertion_point, aside_clips in reaction.get('aside_clips').items():
                all_asides.append([insertion_point, aside_clips, reaction.get('channel') ])

    if len(all_asides) == 0:
        return

    print('INCORPORATING ASIDES')
    all_asides.sort(key=lambda x: x[0], reverse=True)


    def extend_for_aside(clip, insertion_point, duration, aside=None):
        if clip.duration < insertion_point:
            return clip

        if aside is None: 
            # print('insertion_point', insertion_point)
            frame = clip.get_frame(insertion_point)  
            extended_clip = ImageClip(frame, duration=duration).set_audio(AudioClip(lambda t: 0, duration=duration))
        else:
            extended_clip = aside

        # Splice the aside into the current reaction clip
        before = clip.subclip(0, insertion_point)
        after = clip.subclip(insertion_point)  # errors out giving the old duration

        new_clip = concatenate_videoclips([before, extended_clip, after])
        return new_clip



    for insertion_point, aside_clips, channel in all_asides:
        duration = aside_clips[0]['clip'].duration
        print(f"\tAside at {insertion_point} of {duration} seconds for {channel}")

        base_video = extend_for_aside(base_video, insertion_point, duration)

        for name, reaction in conf.get('reactions').items():
            print(f"\t\tCreating clip for {name}")
            reactors = reaction.get('reactors')

            for idx, reactor in enumerate(reactors): 
                # print(reactor, reactor['clip'])

                if channel == reaction.get('channel'):
                    extended_clip = aside_clips[idx].get('clip')
                    reactor['clip'] = extend_for_aside(reactor['clip'], insertion_point, duration, aside=extended_clip)
                else:                     
                    reactor['clip'] = extend_for_aside(reactor['clip'], insertion_point, duration)

    return base_video





# I’m using the following function to mask a video to a hexagon or circular shape, with 
# a border that pulses with color when there is audio from the reactor. 

def create_masked_video(clip, width, height, border_color, border_thickness=10, as_circle=False):


    # Create new PIL images with the same size as the clip, fill with black color
    mask_img_large = Image.new('1', (width, height), 0)
    mask_img_small = Image.new('1', (width, height), 0)
    draw_large = ImageDraw.Draw(mask_img_large)
    draw_small = ImageDraw.Draw(mask_img_small)

    if as_circle:
        # Draw larger and smaller circles on the mask images
        draw_large.ellipse([(0, 0), (width, height)], fill=1)
        draw_small.ellipse([(border_thickness, border_thickness), ((width - border_thickness), (height - border_thickness))], fill=1)
    else:
        # Define hexagon vertices for larger hexagon
        vertices_large = [(0, height*0.25), (width*0.5, 0), (width, height*0.25),
                          (width, height*0.75), (width*0.5, height), (0, height*0.75)]

        # Calculate adjustments for smaller hexagon vertices
        x_adjust = border_thickness * np.sqrt(3) / 2  # trigonometric calculation
        y_adjust = border_thickness / 2

        # Define hexagon vertices for smaller hexagon
        vertices_small = [(x_adjust, height*0.25 + y_adjust), 
                          (width*0.5, y_adjust), 
                          (width - x_adjust, height*0.25 + y_adjust),
                          (width - x_adjust, height*0.75 - y_adjust), 
                          (width*0.5, height - y_adjust), 
                          (x_adjust, height*0.75 - y_adjust)]

        # Draw the larger and smaller hexagons on the mask images
        draw_large.polygon(vertices_large, fill=1)
        draw_small.polygon(vertices_small, fill=1)

    # Subtract smaller mask from larger mask to create border mask
    border_mask = ImageChops.subtract(mask_img_large, mask_img_small)
    border_mask_np = np.array(border_mask)
    mask_img_small_np = np.array(mask_img_small)

    border_func = colorize_when_backchannel_active(border_color, clip)

    def make_frame(t):
        img = np.ones((height, width, 3))

        # Convert the color from HSV to RGB, then scale from 0-1 to 0-255
        color_rgb = np.array(border_func(t)) * 255

        img[border_mask_np > 0] = color_rgb  # apply color to border

        return img

    # Define make_mask function to create mask for the border
    def make_mask(t):
        mask = np.zeros((height, width))
        mask[border_mask_np > 0] = 1
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





# I’d like to make the border dynamic. Specifically:
#   - Each reaction video should have its own bright saturated HSV color assigned to it
#   - The HSV color for the respective reaction video should be mixed with white, 
#     inversely proportional to the volume of the track at that timestamp. That is, 
#     when there is no volume, the border should be white, and when it is at its 
#     loudest, it should be the HSV color.
def colorize_when_backchannel_active(hsv_color, clip):
    h, s, v = hsv_color
    audio_volume = get_audio_volume(clip)

    def color_func(t):
        # Get the volume at current time
        volume = audio_volume[int(t * conversion_audio_sample_rate)]

        # If volume is zero, return white
        if volume == 0:
            return 1, 1, 1  # White in RGB

        # Calculate the interpolated V value based on audio volume
        v_modulated = 1 - volume * (1 - v)

        # Convert modulated HSV color back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v_modulated)
        return r, g, b

    return color_func


def get_audio_volume(clip, fps=None):
    if fps is None:
      fps = conversion_audio_sample_rate
      
    """Calculate the volume of the audio clip"""
    audio = clip.audio.to_soundarray(fps=fps)
    audio_volume = np.sqrt(np.mean(np.square(audio), axis=1))  # RMS amplitude
    audio_volume /= np.max(audio_volume)  # normalize to range [0, 1]
    return audio_volume



def generate_hsv_colors(n, s, v):
    """Generates n evenly distributed HSV colors with the same S and V"""
    return [(i/n, s, v) for i in range(n)]


def match_audio_peak(base_audio_as_array, audio_as_array, factor=1):
    # Compute scale factor based on peak amplitude
    scale_factor = np.max(base_audio_as_array) / np.max(audio_as_array)

    scale_factor *= factor

    # Scale the target audio
    adjusted_audio_data = audio_as_array * scale_factor

    return adjusted_audio_data

