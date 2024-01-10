from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, AudioFileClip
from utilities import conversion_audio_sample_rate as sr, conversion_frame_rate


def convert_video(react_video, conversion_frame_rate, audio_sample_rate, react_video_mp4):
    # Get video dimensions

    # Generate the ffmpeg command
    command = f'ffmpeg -y -i "{react_video}" {resize_command} -c:v libx264 -preset slow -crf 22 -r {conversion_frame_rate} -ar {audio_sample_rate} -c:a aac "{react_video_mp4}"'
    subprocess.run(command, shell=True, check=True)



def insert_filler(video_path, insertion_point, filler_length, output_path):
    # Load the video clip
    clip = VideoFileClip(video_path)

    width, height = clip.size

    resize_command = ""
    if width > 1920 or height > 1080:
        # Calculate aspect ratio
        aspect_ratio = width / height
        if width > height:
            new_width = 1920
            new_height = round(new_width / aspect_ratio)
        else:
            new_height = 1080
            new_width = round(new_height * aspect_ratio)

        clip.resize(width=new_width, height=new_height)

    clip = clip.set_fps(conversion_frame_rate)

    
    # Split the clip at the insertion point
    clip1 = clip.subclip(0, insertion_point)
    clip2 = clip.subclip(insertion_point)
    
    # Create a color clip (black by default) and set its duration
    filler_video = ColorClip(size=clip.size, color=(0, 0, 0), duration=filler_length)
    
    # Set the audio of the filler to be silent (i.e., zero volume)
    filler_audio = AudioFileClip(video_path).subclip(0, filler_length)
    filler_audio = filler_audio.volumex(0)
    filler_video = filler_video.set_audio(filler_audio)
    
    # Concatenate the clips with the filler in between
    final_clip = concatenate_videoclips([clip1, filler_video, clip2])
    
    # Write the result to the output file
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')



if __name__=="__main__":
    from utilities import conf, make_conf, get_song_directory
    from library import songs, drafts, refresh_manifest, finished
    from reactor_core import results_output_dir
    import os, shutil

    song = 'Ren x Chinchilla - Chalk Outlines'
    song_directory = get_song_directory(song)
    reaction = "HarriBest Reactions"
    base = os.path.join(song_directory, 'reactions', reaction)

    insertion = 12
    length = 80
    input_path = base + ".webm"

    output_path = f"{base}-original.mp4"

    insert_filler(input_path, insertion, length, output_path)

    os.remove(input_path)
    os.rename(output_path, input_path.replace('.webm', '.mp4'))

    if os.path.exists( base + ".wav"   ):
        os.remove( base+".wav"   )

    if os.path.exists( base ):
        shutil.rmtree( base )



