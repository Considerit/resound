import noisereduce as nr
import soundfile as sf
import os
import librosa


from utilities.audio_processing import highpass_filter
from utilities import conversion_audio_sample_rate as sr


####################################
# Track separation and high pass filtering


spleeter_separator = None
def get_spleeter(): 
    global spleeter_separator
    if spleeter_separator is None:
        from spleeter.separator import Separator
        spleeter_separator = Separator('spleeter:2stems')
    return spleeter_separator

def separate_vocals(output_dir, song_path, reaction_path, post_process=False):
    # Create a separator with 2 stems (vocals and accompaniment)

    song_sep = output_dir
    song_separation_path = os.path.join(song_sep, os.path.splitext(song_path)[0].split('/')[-1] )

    reaction_sep = song_sep 
    react_separation_path = os.path.join(reaction_sep, os.path.splitext(reaction_path)[0].split('/')[-1] )

    song_vocals_high_passed_path = os.path.join(song_separation_path, 'vocals-post-high-passed.wav')
    reaction_vocals_high_passed_path = os.path.join(react_separation_path, 'vocals-post-high-passed.wav')

    # Perform source separation on song and reaction audio
    if not os.path.exists(song_vocals_high_passed_path):
        song_sources = get_spleeter().separate_to_file(song_path, song_sep)
    if not os.path.exists(reaction_vocals_high_passed_path):
        reaction_sources = get_spleeter().separate_to_file(reaction_path, reaction_sep)

    # Load the separated tracks
    song_vocals_path = os.path.join(song_separation_path, 'vocals.wav')
    reaction_vocals_path = os.path.join(react_separation_path, 'vocals.wav')


    if post_process: 

        sr_reaction = sr_song = None

        if not os.path.exists(song_vocals_high_passed_path):
            song_vocals, sr_song = librosa.load( song_vocals_path, sr=None, mono=True )
            song_vocals = post_process_audio(song_vocals)
            sf.write(song_vocals_high_passed_path, song_vocals.T, sr)
        song_vocals_path = song_vocals_high_passed_path

        if not os.path.exists(reaction_vocals_high_passed_path):
            reaction_vocals, sr_reaction = librosa.load( reaction_vocals_path, sr=None, mono=True )
            reaction_vocals = post_process_audio(reaction_vocals)
            sf.write(reaction_vocals_high_passed_path, reaction_vocals.T, sr)
        reaction_vocals_path = reaction_vocals_high_passed_path

        if sr_reaction and sr_song:
            assert sr_reaction == sr_song, f"Sample rates must be equal {sr_reaction} {sr_song}"

    return (reaction_vocals_path, song_vocals_path)



def post_process_audio(commentary):

    # mono = commentary.ndim == 1

    # # If the audio is mono (1D array), add an extra dimension to make it look like single-channel multi-channel
    # if mono:
    #     commentary = np.expand_dims(commentary, axis=0)

    # processed_commentary = np.zeros_like(commentary)
    
    # for channel in range(commentary.shape[0]):
    #     # Reduce noise
    #     reduced_noise = nr.reduce_noise(commentary[channel], sr=sr)
        
    #     # Apply a highpass filter to remove low-frequency non-speech components
    #     sos = scipy.signal.butter(10, 100, 'hp', fs=sr, output='sos')
    #     filtered = scipy.signal.sosfilt(sos, reduced_noise)
        
    #     processed_commentary[channel] = filtered

    # if mono:
    #     processed_commentary = processed_commentary[0]

    # return processed_commentary

    commentary = highpass_filter(commentary, cutoff=100, fs=sr)

    # Reduce noise
    commentary = nr.reduce_noise(commentary, sr=sr)
    
    # # Apply a highpass filter to remove low-frequency non-speech components
    # sos = scipy.signal.butter(10, 100, 'hp', fs=sr, output='sos')
    # commentary = scipy.signal.sosfilt(sos, commentary)
    
    return commentary
