import os, subprocess

import json
import tempfile


from typing import List, Tuple
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip
from moviepy.editor import ImageClip
from moviepy.video.VideoClip import VideoClip, ColorClip
from moviepy.audio.AudioClip import AudioArrayClip, AudioClip

from utilities import conversion_frame_rate, conversion_audio_sample_rate as sr


def trim_and_concat_video(reaction, video_file: str, video_segments: List[Tuple[float, float]],  filler_video: str, output_file: str, extend_by=0, use_fill=True):

    print(f"Frame rate: {conversion_frame_rate}, Audio rate: {sr}")
    temp_dir, _ = os.path.splitext(output_file)

    if not os.path.exists(temp_dir):
       os.makedirs(temp_dir)

    check_compatibility(filler_video, video_file)

    video_segments = [list(s) for s in video_segments if ((not s[4]) and s[0] < s[1]) or (s[4] and s[2] < s[3] )] # integrity check

    # process video  target_resolution=(1080, 1920)
    reaction_video = VideoFileClip(video_file)
    # if reaction_video.w > 1920:
    #     reaction_video = reaction_video.resize( 1920 / reaction_video.w )

    if reaction_video.audio.fps != sr:
        reaction_video = reaction_video.set_audio(reaction_video.audio.set_fps(sr))

    width = reaction_video.w
    height = reaction_video.h

    base_video = VideoFileClip(filler_video)

    
    ##################

    clips = []
    for segment in video_segments:

        start_frame, end_frame, filler_start, filler_end, filler = segment

        if (filler and use_fill) or not filler:
            if filler: 
                start_frame = filler_start
                end_frame = filler_end
                video = base_video
            else:
                video = reaction_video

            end_frame = min(end_frame, video.duration)

            subclip = video.subclip(float(start_frame), float(end_frame))
            if filler:
                subclip = subclip.resize(height=height, width=width)
                subclip = subclip.without_audio()
        else: 
            clip_duration = float(filler_end - filler_start)
            subclip = ColorClip(size=(width, height), color=(0,0,0)).set_duration(clip_duration).without_audio()


        print(f'Adding frames from {start_frame}s to {end_frame}s filler? {filler}')
        clips.append(subclip)


    #########################################################
    # Append 15 seconds (or extend_by) of each reaction video
    if extend_by > 0:
        fill_length = 0
        for (start, end, filler_start, filler_end, filler) in reversed(video_segments):
            if not filler: 
                break
            fill_length += filler_end - filler_start

        if end + fill_length + extend_by <= reaction_video.duration:  # Make sure not to exceed the reaction video duration
            
            start = end + fill_length
            end = end + fill_length + extend_by
            filler_start = filler_end 
            filler_end = filler_end + fill_length
            filler = False 

            # print(reaction.get('reaction_audio_vocals_data'), start, end, filler_end)
            end_frame = min(end_frame, reaction_video.duration)
            subclip = reaction_video.subclip(float(start), float(end))
            # replace audio with the source-separated vocal track
            # vocals_audio = AudioArrayClip(reaction.get('reaction_audio_vocals_data')[int(start*sr):int(end*sr)])
            # subclip = subclip.set_audio(vocals_audio)
            clips.append(subclip)





    # Concatenate the clips together
    final_clip = concatenate_videoclips(clips)

    # Get the duration of each clip
    final_clip_duration = final_clip.duration
    base_video_duration = base_video.duration


    # If final_clip is longer than base_video, trim it
    if final_clip_duration > base_video_duration + extend_by:
        print("...chopping down final clip")
        final_clip = final_clip.subclip(0, base_video_duration + extend_by)
        
    # If final_clip is shorter than base_video, pad it with black frames
    elif final_clip_duration < base_video_duration + extend_by:
        # Create a black clip with the remaining duration
        print("...adding black clip to end")
        black_clip = (ColorClip((base_video.size), col=(0,0,0))
                      .set_duration(base_video_duration + extend_by - final_clip_duration)
                      .set_fps(final_clip.fps)
                      .set_start(final_clip_duration)
                      .without_audio())

        # Combine the final and black clip using CompositeVideoClip
        final_clip = CompositeVideoClip([final_clip, black_clip])


    final_clip.resize(height=height, width=width).set_fps(conversion_frame_rate)

    # Generate the final output video
    # final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")
    # final_clip.write_videofile(output_file, codec="h264_videotoolbox", audio_codec="aac", ffmpeg_params=['-q:v', '40'])
    final_clip.write_videofile(output_file, 
                                codec="libx264",
                                # preset="slow", 
                                ffmpeg_params=[
                                     '-crf', '18'
                                   ], 
                                audio_codec="aac")


    # Close the video files
    reaction_video.close()
    base_video.close()


    output_video = VideoFileClip(output_file)
    print(f'DONE! Duration={output_video.duration} (compare to {base_video_duration})')

    return output_file


import json

def extract_video_info(video):
    if isinstance(video, (VideoClip, VideoFileClip, CompositeVideoClip)):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as tempf:
            video.write_videofile(tempf.name, codec='libx264', audio_codec='aac')
            return _extract_video_info_from_file(tempf.name)
    else:  # Assuming it's a file path
        return _extract_video_info_from_file(video)


def _extract_video_info_from_file(video_file):
    command = [
        'ffprobe', '-i', video_file, 
        '-hide_banner', 
        '-print_format', 'json', 
        '-show_format', 
        '-show_streams'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    json_start_idx = result.stdout.find('{')
    json_output = result.stdout[json_start_idx:]
    
    try:
        video_info = json.loads(json_output)
    except json.JSONDecodeError:
        print("Error while extracting information from video. FFprobe says:")
        print(result.stdout)
        return None
    
    return video_info

def check_compatibility(video1, video2):
    video1_info = extract_video_info(video1)
    video2_info = extract_video_info(video2)

    if not video1_info or not video2_info:
        print("no video info to check compatibility for", video1_info, video2_info)
        return

    video1_video_stream = next(s for s in video1_info['streams'] if s['codec_type'] == 'video')
    video2_video_stream = next(s for s in video2_info['streams'] if s['codec_type'] == 'video')

    video1_audio_stream = next(s for s in video1_info['streams'] if s['codec_type'] == 'audio')
    video2_audio_stream = next(s for s in video2_info['streams'] if s['codec_type'] == 'audio')

    video_attributes_to_check = [
        'codec_name', 'width', 'height', 'pix_fmt', 
        'avg_frame_rate', 'color_range', 'color_space', 
        'color_transfer', 'color_primaries', 'level', 
        'profile', 'sample_aspect_ratio', 'display_aspect_ratio',
        'field_order'
    ]

    audio_attributes_to_check = [
        'codec_name', 'sample_rate', 'channels', 
        'channel_layout', 'sample_fmt'
    ]

    for attribute in video_attributes_to_check:
        v1_attr = video1_video_stream.get(attribute, "Unavailable")
        v2_attr = video2_video_stream.get(attribute, "Unavailable")
        if v1_attr != v2_attr:
            print(f"Different {attribute}: {video1} has {v1_attr}, while {video2} has {v2_attr}")

    if video1_audio_stream and video2_audio_stream:
        for attribute in audio_attributes_to_check:
            v1_attr = video1_audio_stream.get(attribute, "Unavailable")
            v2_attr = video2_audio_stream.get(attribute, "Unavailable")
            if v1_attr != v2_attr:
                print(f"Different {attribute}: {video1} has {v1_attr}, while {video2} has {v2_attr}")
    elif not video1_audio_stream:
        print(f"{video1} does not contain an audio stream.")
    elif not video2_audio_stream:
        print(f"{video2} does not contain an audio stream.")

    print("done checking compatibility")
