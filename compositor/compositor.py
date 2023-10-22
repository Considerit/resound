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
from moviepy.video.fx import fadeout


from utilities import conf, conversion_frame_rate, conversion_audio_sample_rate as sr

from compositor.layout import create_layout_for_composition

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


def compose_reactor_compilation(extend_by=0, output_size=(1920, 1080)):
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
    all_clips, clip_length = create_clips(base_video, cell_size, draft, output_size)
    print("\tClips created")

    clips = [c for c in all_clips if 'video' in c]
    clips.sort(key=lambda x: x['priority'])
    clips = [c['video'] for c in clips]
    
    audio_clips = [c['audio'] for c in all_clips ]

    final_clip = compose_clips(base_video, clips, audio_clips, clip_length, extend_by, output_size)
    print("\tClips composed")



    if conf.get('background'):
        video_background_path = conf.get('background')
        video_background = VideoFileClip(video_background_path)

        # Loop the background video if it's shorter than the final_clip's duration
        if video_background.duration < final_clip.duration:
            loops_required = int(np.ceil(final_clip.duration / video_background.duration))
            video_background = concatenate_videoclips([video_background] * loops_required)

        # Set the final_clip as a layer on top of the background
        final_clip = CompositeVideoClip([
            video_background.subclip(0, final_clip.duration).resize(output_size),  # Ensure the background video is the same duration as final_clip
            final_clip
        ])

    outerclips = [final_clip]

    if conf.get('introduction', False):
        outerclips.insert(0, VideoFileClip(conf.get('introduction')))

    if conf.get('channel_branding', False):
        outerclips.insert(0, VideoFileClip(conf.get('channel_branding')))

    if conf.get('outro', False):
        outerclips.append(VideoFileClip(conf.get('outro')))


    clip_to_duration = None
    if clip_to_duration is not None:
        outerclips = [o.set_duration(min(clip_to_duration, o.duration)) for o in outerclips] # for testing

    if len(outerclips) > 1:
        for vid in outerclips:
            vid.set_fps(final_clip.fps)
            vid.resize(newsize=final_clip.size)
            vid.audio.fps = sr
            vid = fadeout.fadeout(vid, 0.1)

            # if vid != final_clip:
            #     check_compatibility(vid, final_clip)

        final_clip = concatenate_videoclips(outerclips)



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







def rms_level_excluding_silence(audio_array, threshold=0.01):
    '''Compute the RMS level, excluding silence.'''
    non_silent_samples = audio_array[np.abs(audio_array) > threshold]
    return np.sqrt(np.mean(non_silent_samples**2))

def adjust_gain_for_rms_match(audio_array, target_rms):
    '''Adjust the gain of audio_array to match the target RMS.'''
    current_rms = rms_level_excluding_silence(audio_array)
    return audio_array * (target_rms / current_rms)

def peak_normalize_with_headroom(audio_array, headroom=0.025):
    '''Peak normalize the audio with a given headroom.'''
    peak = np.max(np.abs(audio_array))
    scale_factor = (1 - headroom) / peak
    return audio_array * scale_factor

def dynamic_limit_without_combining(base_audio, all_clips, threshold=0.95):
    '''Dynamically limit the reactor audios in chunks to avoid exceeding the threshold.'''
    chunk_size = int(sr * 0.5)  # Using half-second chunks
    
    # Find out the max length amongst all audios
    max_len = max(len(base_audio), max([len(clip['audio']) for clip in all_clips]))

    # For padding purposes
    def pad_audio_chunk(chunk, size):
        if len(chunk) < size:
            padding = np.zeros((size - len(chunk), chunk.shape[1]))
            return np.vstack((chunk, padding))
        return chunk

    for i in range(0, max_len, chunk_size):
        chunk_end = min(i + chunk_size, max_len)

        # Fetching base audio chunk and padding if necessary
        base_chunk = base_audio[i:chunk_end]
        base_chunk = pad_audio_chunk(base_chunk, chunk_size)

        combined_chunk = base_chunk.copy()

        # Fetching reactor audios chunk-by-chunk and summing them up
        for clip in all_clips:
            reactor_audio = clip['audio']
            reactor_chunk = reactor_audio[i:chunk_end] if i < len(reactor_audio) else np.zeros_like(base_chunk)
            reactor_chunk = pad_audio_chunk(reactor_chunk, chunk_size)
            combined_chunk += reactor_chunk

        scaling_factor = min(1, threshold / np.max(np.abs(combined_chunk)))

        # Apply scaling factor back to the original chunks of reactor audios
        for clip in all_clips:
            reactor_audio = clip['audio']
            if i < len(reactor_audio):
                reactor_chunk = reactor_audio[i:chunk_end]
                reactor_audio[i:chunk_end] = reactor_chunk * scaling_factor





def pan_audio_stereo(audio_array, pan_position):
    """
    Pan the audio in stereo field.
    -1.0 is fully left
     1.0 is fully right
     0.0 is centered
    """
    # Ensure stereo audio
    if len(audio_array.shape) == 1:
        audio_array = np.array([audio_array, audio_array]).T

    # Calculate gain for each channel
    left_gain = np.clip(1 - pan_position, 0, 1)
    right_gain = np.clip(1 + pan_position, 0, 1)

    # Apply the gain
    audio_array[:, 0] *= left_gain
    audio_array[:, 1] *= right_gain

    return audio_array


def create_clips(base_video, cell_size, draft, output_size):

    total_reactors = 0 
    for name, reaction in conf.get('reactions').items():
      total_reactors += len(reaction.get('reactors', []))

    reactor_colors = generate_hsv_colors(total_reactors, 1, .6)
    clip_length = 0

    all_clips = []

      

    base_video = incorporate_asides(base_video)

    base_video.audio.fps = sr
    base_audio_as_array = base_video.audio.to_soundarray()

    base_audio_as_array = peak_normalize_with_headroom(base_audio_as_array, headroom=0.2)
    base_audio_rms = rms_level_excluding_silence(base_audio_as_array)


    # Collect all reactor audios for dynamic limiting
    reactor_audios = []

    print("\tCreating all clips")
    for name, reaction in conf.get('reactions').items():
        print(f"\t\tCreating clip for {name}")
        reactors = reaction.get('reactors')
        if reactors is None:
            continue


        positions = []
        for idx, reactor in enumerate(reactors): 
            print(f"\t\t\tReactor {idx}")

            reactor_color = reactor_colors.pop()
            x,y = reactor['grid_assignment']

            featured = reaction['featured']

            clip = reactor['clip']

            clip.audio.fps = sr 

            def inspect_clip(clip):
                try:
                    # Try to convert the clip's audio to a sound array
                    _ = clip.audio.to_soundarray()
                except Exception as e:
                    # If an error occurs, print or log relevant details about the clip
                    print(f"Error with clip: {clip}; {clip.duration}")  # Adapt this based on how you want to identify the clip
                        
                    path_to_save = os.path.join(conf.get('temp_directory'), 'problem_clip.wav')
                    # Check if it's a VideoClip, then get its audio to save
                    if isinstance(clip, VideoClip) and clip.audio:
                        clip.audio.write_audiofile(path_to_save)
                        path_to_save = os.path.join(conf.get('temp_directory'), 'problem_clip.mp4')
                        clip.write_videofile(path_to_save)
                    elif isinstance(clip, (AudioClip, CompositeAudioClip)):
                        clip.write_audiofile(path_to_save)


                    print(e)

                # Check if the current clip is a CompositeAudioClip
                if isinstance(clip, CompositeAudioClip):
                    # If it is, recursively inspect its subclips
                    for subclip in clip.clips:
                        inspect_clip(subclip)

            # inspect_clip(clip)


            print("\t\t\t\t...to sound array")
            reactor_audio_array = clip.audio.to_soundarray()
            print('\t\t\t\t...adjust gain')
            adjusted_reactor_audio = adjust_gain_for_rms_match(reactor_audio_array, base_audio_rms)


            # volume_adjusted_audio = match_audio_peak(base_audio_as_array, clip.audio.to_soundarray(), factor=1)
            # volume_adjusted_clip = AudioArrayClip(volume_adjusted_audio, fps=clip.audio.fps)

            size = cell_size
            if featured: 
              size *= 1.15
              size = int(size)

            clip = clip.resize((size, size))
            if not draft:
              print('\t\t\t\t...masking')
              clip = create_masked_video(clip, border_color=reactor_color, border_thickness=min(30, max(5, size / 15)), width=size, height=size, as_circle=featured)

            position = (x - size / 2, y - size / 2)

            # Calculate pan position based on grid assignment
            pan_position = (2 * position[0] / output_size[0]) - 1

            # Pan the reactor's audio
            adjusted_reactor_audio = pan_audio_stereo(adjusted_reactor_audio, pan_position)

            clip = clip.set_position(position)

            if clip_length < clip.duration:
              clip_length = clip.duration

            if featured:
              priority = 100
            else: 
              priority = reaction.get('priority')

            clip_info = {
              'channel': name,
              'reaction': reaction,
              'priority': priority,
              'video': clip,
              'audio': adjusted_reactor_audio,
              'position': position,
              'reactor_idx': idx
            }

            all_clips.append(clip_info)
            positions.append(position)
            print('\t\t\t\t...done!')


    # Apply dynamic limiting on the collected reactor audios
    dynamic_limit_without_combining(base_audio_as_array, all_clips)

    # Set the modified audios back to their respective clips
    for idx, clip_info in enumerate(all_clips):
        clip_info['audio'] = AudioArrayClip(clip_info['audio'], fps=sr)


    base_clip = {
      'audio': AudioArrayClip(base_audio_as_array, fps=sr), #base_video.audio,
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
        if reactors is None:
            continue

        if reaction.get('aside_clips', None):
            for insertion_point, aside_clips in reaction.get('aside_clips').items():
                all_asides.append([insertion_point, aside_clips, reaction.get('channel') ])

    if len(all_asides) == 0:
        return base_video

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
            if reactors is None:
                continue

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


    audio_volume = get_audio_volume(clip)

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

            # If volume is zero, return white
            if volume == 0:
                return 0, 0, 0  # White in RGB

            # Calculate the interpolated V value based on audio volume
            v_modulated = 1 - volume * (1 - v)

            # Convert modulated HSV color back to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v_modulated)
            return r, g, b

        return color_func

    border_func = colorize_when_backchannel_active(border_color, clip)


    def make_frame(t):
        img = np.zeros((height, width, 3))

        # Convert the color from HSV to RGB, then scale from 0-1 to 0-255
        color_rgb = np.array(border_func(t)) * 255
        img[border_mask_np > 0] = color_rgb  # apply color to border

        return img

    # Define make_mask function to create mask for the border
    # def make_mask(t):
    #     mask = np.zeros((height, width))
    #     mask[border_mask_np > 0] = 1
    #     return mask

    def make_mask(t):
        mask = np.zeros((height, width))

        # Get the volume at current time
        volume = audio_volume[int(t * sr)]

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







def get_audio_volume(clip, fps=None):
    if fps is None:
      fps = sr
      
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

