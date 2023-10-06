import noisereduce as nr
import soundfile as sf
import os
import soundfile as sf


from utilities.audio_processing import highpass_filter, convert_to_mono
from utilities import conf, conversion_audio_sample_rate as sr


####################################
# Track separation and high pass filtering


spleeter_separator = None
def get_spleeter(): 
    global spleeter_separator
    if spleeter_separator is None:
        from spleeter.separator import Separator
        spleeter_separator = Separator('spleeter:2stems')
    return spleeter_separator


def separate_vocals(output_dir, audio_path, output_filename):
    # Create a separator with 2 stems (vocals and accompaniment)

    vocals_high_passed_path = os.path.join(output_dir, output_filename)

    # Perform source separation
    if not os.path.exists(vocals_high_passed_path):
        song_sources = get_spleeter().separate_to_file(audio_path, output_dir)

    # Load the separated tracks
    vocals_path = os.path.join(output_dir, 'vocals.wav')

    # Post process
    if not os.path.exists(vocals_high_passed_path):
        #vocals, sr_song = librosa.load( vocals_path, sr=sr, mono=True )
        # vocals = post_process_audio(vocals)
        # sf.write(song_vocals_high_passed_path, vocals.T, sr)

        audio_data, __ = sf.read(vocals_path)
        vocals = convert_to_mono(audio_data)
        vocals = post_process_audio(vocals)
        sf.write(vocals_high_passed_path, vocals, sr)


    return vocals_high_passed_path




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
