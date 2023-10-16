import glob, os
import librosa

from utilities.utilities import conversion_audio_sample_rate as sr
from utilities.utilities import extract_audio
from utilities.audio_processing import pitch_contour, spectral_flux, root_mean_square_energy, continuous_wavelet_transform







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


  base_audio_path = os.path.join( song_directory, f"{os.path.basename(song_directory)}.wav")


  intro_path = os.path.join(song_directory, 'intro.mp4')
  if not os.path.exists(intro_path):
    intro_path = False

  outro_path = os.path.join(song_directory, 'outro.mp4')
  if not os.path.exists(outro_path):
    outro_path = False

  background_path = os.path.join(song_directory, 'background.mp4')
  if not os.path.exists(background_path):
    background_path = None


  conf.update({
    "artist": song_def['artist'],
    "song_name": song_def['song'],
    "song_key": song, 
    "base_audio_path": base_audio_path,
    "include_base_video": song_def.get('include_base_video', True),
    "song_directory": song_directory,
    "reaction_directory": reaction_directory,
    "compilation_path": compilation_path,
    "temp_directory": full_output_dir,
    "has_asides": song_def.get('asides'),

    'hop_length': 256,
    'n_mfcc': 20,

    'channel_branding': 'production/resound_channel_reencoded.mp4',
    'introduction': intro_path,
    'outro': outro_path,

    'search_tester': song_def.get('search_tester', None),
    'background': background_path
  })


  def load_reactions():
    from inventory import get_manifest_path

    try: 
      manifest = open(get_manifest_path(conf['artist'], conf['song_name']), "r")
    except:
      print(f"Manifest doesn't yet exist for {conf['song_name']}")
    reaction_videos = prepare_reactions()

    temp_directory = conf.get('temp_directory')

    reactions = {}

    print(reaction_videos)

    multiple_reactors = song_def.get('multiple_reactors', None) 
    priority = song_def.get('priority', {}) 
    swap_grid_positions = song_def.get('swap_grid_positions', None)
    fake_reactor_position = song_def.get('fake_reactor_position', None)
    start_reaction_search_at = song_def.get('start_reaction_search_at', None)
    unreliable_bounds = song_def.get('unreliable_bounds', None)
    chunk_size = song_def.get('chunk_size', None)

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

      if not conf.get('alignment_test', False) or target_score is not None:
        featured = channel in song_def.get('featured', [])

        reactions[channel] = {
          'channel': channel,
          'video_path': reaction_video_path, 
          'aligned_path': os.path.join(temp_directory, os.path.basename(channel) + f"-CROSS-EXPANDER.mp4"),
          'featured': featured,
          'asides': song_def.get('asides', {}).get(channel, None),
          'ground_truth': ground_truth,
          'ground_truth_path': ground_truth_path,
          'target_score': target_score
        }

        if multiple_reactors is not None:
          reactions[channel]['num_reactors'] = multiple_reactors.get(channel, 1)
        
        if fake_reactor_position is not None and fake_reactor_position.get(channel, False): 
          reactions[channel]['fake_reactor_position'] = fake_reactor_position.get(channel)

        if start_reaction_search_at is not None and start_reaction_search_at.get(channel, False):
          reactions[channel]['start_reaction_search_at'] = start_reaction_search_at.get(channel)

        if unreliable_bounds is not None and unreliable_bounds.get(channel, False):
          reactions[channel]['unreliable_bounds'] = unreliable_bounds.get(channel)

        if chunk_size is not None and chunk_size.get(channel, False):
          reactions[channel]['chunk_size'] = chunk_size.get(channel)

        if featured: 
          default_priority = 75
        else:
          default_priority = 50

        reactions[channel]['priority'] = priority.get(channel, default_priority)


        if swap_grid_positions is not None and swap_grid_positions.get(channel, None):
          reactions[channel]['swap_grid_positions'] = swap_grid_positions.get(channel)


    conf['reactions'] = reactions

  def load_reaction(channel):
    print(f"Loading reaction {channel}")
    reaction_conf = conf.get('reactions')[channel]

    if not conf.get('base_video_path'):
      load_base_video()

    if not 'reaction_audio_path' in reaction_conf: 
      reaction_video_path = reaction_conf['video_path']
      reaction_audio_data, __, reaction_audio_path = extract_audio(reaction_video_path)
      reaction_conf["reaction_audio_data"] = reaction_audio_data
      reaction_conf['reaction_audio_path'] = reaction_audio_path


      output_dir = conf.get('reaction_directory')

      load_audio_transformations(reaction_conf, 'reaction', output_dir, reaction_audio_path, reaction_audio_data)




  def load_aligned_reaction_data(channel):
    reaction_conf = conf.get('reactions')[channel]

    if not conf.get('base_video_path'):
      load_base_video()

    if not 'reaction_audio_path' in reaction_conf: 
      load_reaction(channel)

    if not 'aligned_reaction_data' in reaction_conf:
      path = reaction_conf.get('aligned_audio_path')

      aligned_reaction_data, __, __ = extract_audio(path)
      reaction_conf['aligned_reaction_data'] = aligned_reaction_data

  def remove_reaction(channel):
      if channel in conf.get('reactions').get(channel, False):
        unload_reaction(channel)
        del conf['reactions'][channel]

  conf.update({
    'load_base_video': load_base_video,
    'load_reaction': load_reaction,
    'load_aligned_reaction_data': load_aligned_reaction_data,
    'remove_reaction': remove_reaction
  })

  load_reactions()


  return False




to_delete = ['aligned_reaction_data']
for source in ['_', '_vocals', '_accompaniment']:
  for metric in ['_data', '_mfcc', '_pitch_contour', '_spectral_flux', '_root_mean_square_energy']:     # _continuous_wavelet_transform
    to_delete.append(f"reaction_audio{source}{metric}")

def unload_reaction(channel):
  import gc
  global conf

  print(f"Unloading reaction {channel}")

  reaction_conf = conf.get('reactions')[channel]

  for field in to_delete:
    if field in reaction_conf:
      del reaction_conf[field]

  gc.collect()



def load_audio_transformations(local_conf, prefix, output_dir, audio_path, audio_data):

    from backchannel_isolator.track_separation import separate_vocals

    vocal_path_filename = 'vocals-post-high-passed.wav'

    separation_path = os.path.join(output_dir, os.path.splitext(audio_path)[0].split('/')[-1] )

    vocals_path = os.path.join(separation_path, vocal_path_filename)

    if not os.path.exists( vocals_path ):
        separate_vocals(separation_path, audio_path, vocal_path_filename)


    # local_conf[f'{prefix}_audio_mfcc'] = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=conf.get('n_mfcc'), hop_length=conf.get('hop_length'))
    # local_conf[f'{prefix}_audio_pitch_contour'] = pitch_contour(audio_data, sr, hop_length=conf.get('hop_length'))


    for source in ( '', 'accompaniment', 'vocals' ):

      if source == '':
        data = audio_data
        path = audio_path
        source = '_'
      else: 
        path = os.path.join(separation_path, f'{source}.wav')
        data, _, _ = extract_audio(path)
        source = f"_{source}_"

      local_conf[f'{prefix}_audio{source}path'] = path
      local_conf[f"{prefix}_audio{source}data"] = data

      mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=conf.get('n_mfcc'), hop_length=conf.get('hop_length'))
      local_conf[f'{prefix}_audio{source}mfcc'] = mfcc

      pc = pitch_contour(data, sr, hop_length=conf.get('hop_length'))
      local_conf[f'{prefix}_audio{source}pitch_contour'] = pc

      # sf = spectral_flux(y=data, sr=sr, hop_length=conf.get('hop_length'))
      # local_conf[f'{prefix}_audio{source}spectral_flux'] = sf

      # rmse = root_mean_square_energy(y=data, sr=sr, hop_length=conf.get('hop_length'))
      # local_conf[f'{prefix}_audio{source}root_mean_square_energy'] = rmse






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


    song_audio_data, _, base_audio_path = extract_audio(base_video_path)
    song_audio_data = song_audio_data  #[0:150*sr]

    base_data = {
      'base_video_path': base_video_path,
      'song_audio_data': song_audio_data,
    }

    output_dir = conf.get('reaction_directory')

    load_audio_transformations(base_data, 'song', output_dir, base_audio_path, song_audio_data)


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

