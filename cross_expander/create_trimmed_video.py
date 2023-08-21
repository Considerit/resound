import os
from typing import List, Tuple
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip



def trim_and_concat_video(video_file: str, video_segments: List[Tuple[float, float]],  filler_video: str, output_file: str, ext: str, extend_by=0, use_fill=True):

    print(f"Frame rate: {conversion_frame_rate}, Audio rate: {conversion_audio_sample_rate}")
    temp_dir, _ = os.path.splitext(output_file)

    if not os.path.exists(temp_dir):
       os.makedirs(temp_dir)

    check_compatibility(filler_video, video_file)

    video_segments = [list(s) for s in video_segments if ((not s[4]) and s[0] < s[1]) or (s[4] and s[2] < s[3] )] # integrity check

    # process video  target_resolution=(1080, 1920)
    reaction_video = VideoFileClip(video_file)
    width = reaction_video.w
    height = reaction_video.h

    base_video = VideoFileClip(filler_video)

    #########################################################
    # Append 15 seconds (or extend_by) of each reaction video
    if extend_by > 0:
        fill_length = 0
        for (start, end, filler_start, filler_end, filler) in reversed(video_segments):
            if not filler: 
                break
            fill_length += filler_end - filler_start

        if end + fill_length + extend_by <= reaction_video.duration:  # Make sure not to exceed the reaction video duration
            video_segments.append((end + fill_length, end + fill_length + extend_by, filler_end, filler_end + fill_length + extend_by, False))
    
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
                subclip = subclip.set_fps(reaction_video.fps)
        else: 
            clip_duration = float(filler_end - filler_start)
            subclip = ColorClip(size=(width, height), color=(0,0,0)).set_duration(clip_duration).set_fps(reaction_video.fps).without_audio()


        print(f'Adding frames from {start_frame}s to {end_frame}s filler? {filler}')
        clips.append(subclip)


    # Concatenate the clips together
    final_clip = concatenate_videoclips(clips)
    final_clip.set_fps(30)

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


    final_clip.resize(height=height, width=width)

    # Generate the final output video
    final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")


    # Close the video files
    reaction_video.close()
    base_video.close()


    output_video = VideoFileClip(output_file)
    print(f'DONE! Duration={output_video.duration} (compare to {base_video_duration})')

    return output_file