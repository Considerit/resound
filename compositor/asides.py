import numpy as np
import os
import math

from moviepy.editor import ImageClip, VideoFileClip, CompositeVideoClip
from moviepy.editor import concatenate_videoclips

from moviepy.video.VideoClip import VideoClip
from moviepy.audio.AudioClip import AudioClip

from moviepy.video.fx import fadeout

from utilities import extract_audio, conf, conversion_frame_rate, conversion_audio_sample_rate as sr



def create_asides(reaction):
    from face_finder import create_reactor_view

    reaction["aside_clips"] = {}
    # print(f"Asides: {reaction.get('channel')} has {len(reaction['asides'])}")

    all_asides = reaction['asides']

    def consolidate_adjacent_asides(all_asides):
        groups = []
        all_asides.sort(key=lambda x: x[0])  # sort by start value of the aside

        group_when_closer_than = 0.1  # put asides into the same group if one's start value 
                                      # is less than group_when_closer_than greater than the 
                                      # end value of the previous aside 

        # Initialize the first group
        current_group = []

        # Group asides
        for aside in all_asides:
            aside = start, end, insertion_point, rewind = get_aside_config(aside)

            # If current_group is empty or the start of the current aside is close enough to the end of the last aside in the group
            if not current_group or insertion_point - current_group[-1][2] < group_when_closer_than:
                current_group.append(aside)
            else:
                # Current aside starts a new group
                groups.append(current_group)
                current_group = [aside]

        # Add the last group if it's not empty
        if current_group:
            groups.append(current_group)

        consolidated_asides = []
        for group in groups:
            start = group[0][0]
            end = group[-1][1]

            aside_conf = {
                'range': (start, end),
                'insertion_point': group[-1][2],
                'rewind': group[-1][3],
                'keep_segments': [ (g[0] - start, g[1] - start) for g in group ]
            }
            consolidated_asides.append(aside_conf)

        return consolidated_asides

        
    def get_aside_config(aside):
        if len(aside) == 4:
            start, end, insertion_point, rewind = aside
        else: 
            start, end, insertion_point = aside
            rewind = 3

        return start, end, insertion_point, rewind

    def remove_skipped_segments(aside_reactor_views, keep_segments):
        keep_segments.sort( key=lambda x: x[0] )

        for reactor_view in aside_reactor_views:
            # Initialize a list to hold the subclips
            subclips = [ reactor_view['clip'].subclip(aside[0], aside[1]) for aside in keep_segments ]

            # Concatenate all the subclips together
            reactor_view['clip'] = concatenate_videoclips(subclips)

        return aside_reactor_views


    consolidated_asides = consolidate_adjacent_asides(all_asides)

    for i, aside in enumerate(consolidated_asides):

        start, end = aside['range']
        insertion_point = aside['insertion_point']
        rewind = aside['rewind']
        keep_segments = aside['keep_segments']

        # first, isolate the correct part of the reaction video
        aside_video_clip = os.path.join(conf.get('temp_directory'), f"{reaction.get('channel')}-aside-{i}.mp4")
        
        if not os.path.exists(aside_video_clip):            
            react_video = VideoFileClip(reaction.get('video_path'), has_mask=True)
            aside_clip = react_video.subclip(float(start), float(end))
            aside_clip.set_fps(30)
            aside_clip.write_videofile(aside_video_clip, codec="h264_videotoolbox", audio_codec="aac", ffmpeg_params=['-q:v', '40'])
            react_video.close()

        # do face detection on it
        aside_reactor_views, __ = create_reactor_view(reaction, aside_video = aside_video_clip, show_facial_recognition=False)
        aside_reactor_views = remove_skipped_segments(aside_reactor_views, keep_segments)

        audio_data = aside_reactor_views[0]['clip'].audio.set_fps(sr).to_soundarray()
        for reactor_view in aside_reactor_views: 
            reactor_view['audio'] = audio_data
            reactor_view['clip']  = reactor_view['clip'].without_audio()


        reaction["aside_clips"][insertion_point] = (aside_reactor_views, rewind)





# Any of the reaction clips can have any number of "asides". An aside is a bonus 
# video clip spliced into a specific point in the respective reaction video clip. 
# When an aside is active, only the respective video clip is playing, and all the 
# other clips are paused. When a clip is paused because an aside is playing, the 
# previous frame is replicated until the aside is finished, and no audio is played. 


def incorporate_asides(base_video, base_audio_clip, audio_scaling_factors):

    all_asides = []
    for name, reaction in conf.get('reactions').items():
        reactors = reaction.get('reactors')
        if reactors is None:
            continue

        if reaction.get('aside_clips', None):
            for insertion_point, (aside_clips, rewind) in reaction.get('aside_clips').items():
                all_asides.append([insertion_point, aside_clips, reaction.get('channel'), rewind ])

    if len(all_asides) == 0:
        return base_video, base_audio_clip

    print('INCORPORATING ASIDES')
    all_asides.sort(key=lambda x: x[0], reverse=True)


    def extend_for_aside(clip, insertion_point, duration, aside=None, use_countdown_timer=False):
        if clip.duration < insertion_point:
            return clip

        if aside is None: 
            # print('insertion_point', insertion_point)
            frame = clip.get_frame(max(0,insertion_point - 1))
            extended_clip = ImageClip(frame, duration=duration) #.set_audio(AudioClip(lambda t: 0, duration=duration))
        else:
            extended_clip = aside

        # Splice the aside into the current reaction clip
        before = clip.subclip(0, insertion_point)
        after = clip.subclip(insertion_point)

        new_clip = concatenate_videoclips([before, extended_clip, after])
        return new_clip

    def extend_for_audio_aside(audio, insertion_point, duration, aside=None):
        insertion_point = int(insertion_point * sr)

        if audio.shape[0] < insertion_point:
            return audio

        if aside is None:
            # Create a silent audio clip with the given duration
            aside = np.zeros((int(sr * duration), audio.shape[1]))

        new_audio = np.concatenate([audio[0:insertion_point, :], aside, audio[insertion_point:, :]])

        return new_audio


    original_video = base_video

    reaction_segments = {}
    for channel, reaction in conf.get('reactions').items():
        reaction_segments[channel] = []
        reactors = reaction.get('reactors')
        if reactors is None:
            continue
        for idx, reactor in enumerate(reactors): 
            reaction_segments[channel].append([reactor['clip']])




    for i, (insertion_point, aside_clips, channel, rewind) in enumerate(all_asides):
        duration = aside_clips[0]['clip'].duration
        print(f"\tAside at {insertion_point} of {duration} seconds for {channel}")

        if not rewind or original_video.duration < insertion_point:
            rewind = 0

        if rewind > 0:             
            rewind = min(insertion_point, rewind)
            # extend base video for aside with the rewind clip
            print(insertion_point, rewind, insertion_point - rewind)
            rewind_clip = original_video.subclip(insertion_point - rewind, insertion_point) # not sure why I have to use original_video rather than base_video here...

            rewind_icon = ImageClip(os.path.join('compositor', 'rewind.png')).set_duration(1).resize( (100,100)  )
            rewind_icon = rewind_icon.fadeout(.5)
            rewind_icon = rewind_icon.set_position(("center", "center"))

            composite_rewind_clip = CompositeVideoClip([rewind_clip, rewind_icon]) # results in video with just black background
            f = os.path.join(conf.get('temp_directory'), f'COMPOSITE_REWIND_CLIP-{insertion_point-rewind}-{insertion_point}.mp4')
            composite_rewind_clip.write_videofile(f, 
                                 codec="h264_videotoolbox", 
                                 ffmpeg_params=['-q:v', '60']
                                )
            composite_rewind_clip = VideoFileClip(f).resize(rewind_clip.size)

            base_video = extend_for_aside(base_video, insertion_point, duration=rewind, aside=composite_rewind_clip)

            rewind_audio_clip = base_audio_clip[int((insertion_point - rewind) * sr):int(insertion_point * sr), :] 
            base_audio_clip = extend_for_audio_aside(base_audio_clip, insertion_point, duration=rewind, aside=rewind_audio_clip)

        # base_video.write_videofile(os.path.join(conf.get('temp_directory'), f'{i}-before.mp4'), 
        #                                  codec="h264_videotoolbox", 
        #                                  ffmpeg_params=['-q:v', '40']
        #                                 )       

        base_video      = extend_for_aside(base_video, insertion_point, duration)



        base_audio_clip = extend_for_audio_aside(base_audio_clip, insertion_point, duration)


        # base_video.write_videofile(os.path.join(conf.get('temp_directory'), f'{i}-after.mp4'), 
        #                                  codec="h264_videotoolbox", 
        #                                  ffmpeg_params=['-q:v', '40']
        #                                 )       

        for name, reaction in conf.get('reactions').items():
            print(f"\t\tCreating clip for {name}")
            reactors = reaction.get('reactors')
            if reactors is None:
                continue


            for idx, reactor in enumerate(reactors): 

                if rewind > 0:
                    rewind_clip = reactor['clip'].subclip(insertion_point - rewind, insertion_point)
                    reactor['clip'] = extend_for_aside(reactor['clip'], insertion_point, duration=rewind, aside=rewind_clip, use_countdown_timer=False)

                if channel == name:
                    assert( len(aside_clips), len(reactors), f"Number of aside reactors does not match the number of reactors for {channel}, check if the asides face recog found the right number."  )
                    extended_clip = aside_clips[idx].get('clip')

                    reactor['clip'] = extend_for_aside(reactor['clip'], insertion_point, duration, aside=extended_clip, use_countdown_timer=True)

                else:
                    # reactor['clip'] = extend_for_aside(reactor['clip'], insertion_point, duration + rewind)
                    reactor['clip'] = extend_for_aside(reactor['clip'], insertion_point, duration)


            len_audio_before = reaction['mixed_audio'].shape[0]

            if rewind > 0:
                reaction['mixed_audio'] = extend_for_audio_aside(reaction['mixed_audio'], insertion_point, duration=rewind)

                rewind_scale = audio_scaling_factors[name][round(sr * (insertion_point - rewind)):round((sr * insertion_point))]
                # rewind_scale = np.ones( int(sr * rewind))
            else: 
                rewind_scale = np.zeros(0)


            if channel == name:
                extended_clip = aside_clips[idx].get('audio')
                reaction['mixed_audio'] = extend_for_audio_aside(reaction['mixed_audio'], insertion_point, duration, aside=extended_clip)                
                middle = np.ones( int(sr * duration))
                # middle = np.ones( int(sr * (duration + rewind)))

            else: 
                # reaction['mixed_audio'] = extend_for_audio_aside(reaction['mixed_audio'], insertion_point, duration + rewind)                
                reaction['mixed_audio'] = extend_for_audio_aside(reaction['mixed_audio'], insertion_point, duration)
                middle = np.zeros(int(sr * duration))
                # middle = np.zeros(int(sr * (duration + rewind)))


            split = int(sr * insertion_point)
            audio_scaling_factors[name] = np.concatenate([audio_scaling_factors[name][0:split], middle, rewind_scale, audio_scaling_factors[name][split:]])
    


    from compositor.mix_audio import plot_scaling_factors
    plot_scaling_factors(audio_scaling_factors)
        

    return (base_video, base_audio_clip)
