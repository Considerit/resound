import glob, os
import librosa

from utilities.utilities import conversion_audio_sample_rate as sr
from utilities.utilities import extract_audio
from utilities.audio_processing import audio_percentile_loudness

conf = {} # global conf


def make_conf(song_def, options, temp_directory): 
  global conf

  conf.clear()

  conf.update(options)


  song = f"{song_def['artist']} - {song_def['song']}"

  song_directory = os.path.join('Media', song)
  reaction_directory = os.path.join(song_directory, 'reactions')

  compilation_path = os.path.join(song_directory, f"{song} (compilation).mp4")
  if conf.get('draft', False):
      compilation_path += "fast.mp4" 

  full_output_dir = os.path.join(song_directory, temp_directory)
  if not os.path.exists(full_output_dir):
     # Create a new directory because it does not exist
     os.makedirs(full_output_dir)


  lock_file = os.path.join(full_output_dir, 'locked')
  if os.path.exists( lock_file  ):
      return True


  conf.update({
    "artist": song_def['artist'],
    "song_name": song_def['song']
  })

  conf.update({
    "song_key": song, 
    "include_base_video": song_def.get('include_base_video', True),
    "song_directory": song_directory,
    "reaction_directory": reaction_directory,
    "compilation_path": compilation_path,
    "temp_directory": full_output_dir,

    'hop_length': 256,
    'n_mfcc': 20
  })


  def load_reactions():
    from inventory import get_manifest_path

    manifest = open(get_manifest_path(conf['artist'], conf['song_name']), "r")
    reaction_videos = prepare_reactions()

    temp_directory = conf.get('temp_directory')

    reactions = {}


    for reaction_video_path in reaction_videos:
      channel, __ = os.path.splitext(os.path.basename(reaction_video_path))

      ground_truth = song_def.get('ground_truth', {}).get(channel, None)
      ground_truth_path = None
      if ground_truth: 
          ground_truth = [ (int(s * sr), int(e * sr)) for (s,e) in ground_truth]
          ground_truth_path = []
          current_start = 0
          for reaction_start, reaction_end in ground_truth:
            current_end = current_start + reaction_end - reaction_start

            ground_truth_path.append( (reaction_start, reaction_end, current_start, current_end, False)   )

            current_start += reaction_end - reaction_start

      target_score = song_def.get('target_scores', {}).get(channel, None)

      if not conf['alignment_test'] or target_score is not None:
        reactions[channel] = {
          'channel': channel,
          'video_path': reaction_video_path, 
          'aligned_path': os.path.join(temp_directory, os.path.basename(channel) + f"-CROSS-EXPANDER.mp4"),
          'featured': channel in song_def.get('featured', []),
          'asides': song_def.get('asides', {}).get(channel, None),
          'ground_truth': ground_truth,
          'ground_truth_path': ground_truth_path,
          'target_score': target_score
        }

    conf['reactions'] = reactions

  def load_reaction(channel):
    reaction_conf = conf.get('reactions')[channel]

    if not conf.get('base_video_path'):
      load_base_video()

    if not 'reaction_audio_path' in reaction_conf: 
      reaction_video_path = reaction_conf['video_path']
      reaction_audio_data, __, reaction_audio_path = extract_audio(reaction_video_path)
      reaction_conf["reaction_audio_data"] = reaction_audio_data
      reaction_conf['reaction_audio_path'] = reaction_audio_path

      reaction_conf['reaction_audio_mfcc'] = librosa.feature.mfcc(y=reaction_audio_data, sr=sr, n_mfcc=conf.get('n_mfcc'), hop_length=conf.get('hop_length'))
      reaction_conf['reaction_percentile_loudness'] = audio_percentile_loudness(reaction_audio_data, loudness_window_size=100, percentile_window_size=1000, std_dev_percentile=None, hop_length=conf.get('hop_length'))



  conf.update({
    'load_base_video': load_base_video,
    'load_reaction': load_reaction
  })

  load_reactions()


  return False

def unload_reaction(channel):
  import gc
  global conf

  reaction_conf = conf.get('reactions')[channel]

  if 'reaction_audio_data' in reaction_conf: 
    del reaction_conf["reaction_audio_data"]
    del reaction_conf['reaction_audio_mfcc']
    del reaction_conf['reaction_percentile_loudness']

  gc.collect()


def load_base_video():
    from silence import get_edge_silence

    song_directory = conf.get('song_directory')

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
                command = f"ffmpeg -i \"{base_video_path_webm}\" -y -vf \"setpts=PTS\" -q:v 40 -c:v h264_videotoolbox -r {conversion_frame_rate} -ar {conversion_audio_sample_rate} -ss {start} -to {end} \"{base_video_path_mp4}\""
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


    base_audio_data, _, base_audio_path = extract_audio(base_video_path)

    base_audio_mfcc = librosa.feature.mfcc(y=base_audio_data, sr=sr, n_mfcc=conf.get('n_mfcc'), hop_length=conf.get('hop_length'))
    song_percentile_loudness = audio_percentile_loudness(base_audio_data, loudness_window_size=100, percentile_window_size=1000, std_dev_percentile=None, hop_length=conf.get('hop_length'))

    base_data = {
      'base_video_path': base_video_path,
      'base_audio_data': base_audio_data,
      'base_audio_path': base_audio_path,
      'song_percentile_loudness': song_percentile_loudness,
      'base_audio_mfcc': base_audio_mfcc
    }

    conf.update(base_data)



def prepare_reactions():

    reaction_dir = conf.get('reaction_directory')

    print("Processing reactions in: ", reaction_dir)

    # Get all reaction video files
    webm_videos = glob.glob(os.path.join(reaction_dir, "*.webm"))
    mp4_videos = glob.glob(os.path.join(reaction_dir, "*.mp4"))
    mkv_videos = glob.glob(os.path.join(reaction_dir, "*.mkv"))    
    react_videos = webm_videos + mp4_videos + mkv_videos    


    # # Process each reaction video
    # for react_video in react_videos:
    #     react_video_name, react_video_ext = os.path.splitext(react_video)
    #     react_video_mp4 = react_video_name + '.mp4'
    #     if not os.path.exists(react_video_mp4):
    #         with VideoFileClip(react_video) as clip:
    #             width, height = clip.size
    #         resize_command = ""
    #         if width > 1920 or height > 1080:
    #             # Calculate aspect ratio
    #             aspect_ratio = width / height
    #             if width > height:
    #                 new_width = 1920
    #                 new_height = int(new_width / aspect_ratio)
    #             else:
    #                 new_height = 1080
    #                 new_width = int(new_height * aspect_ratio)
                
    #             resize_command = f"-vf scale={new_width}:{new_height}"
    #         # Generate the ffmpeg command
    #         command = f'ffmpeg -y -i "{react_video}" {resize_command} -c:v libx264 -r {conversion_frame_rate} -ar {conversion_audio_sample_rate} -c:a aac "{react_video_mp4}"'
            
    #         print(command)
    #         subprocess.run(command, shell=True, check=True)
    #         if os.path.exists(react_video):
    #             os.remove(react_video)
    # react_videos = glob.glob(os.path.join(reaction_dir, "*.mp4"))



    return react_videos

