# Various approaches to separate out reactor comments from the base 
# audio file. None of them work.
#
# Prompt: 
# You're given a "base video" and a "reaction video".  The reaction video has one or two people who have 
# recorded themselves watching the base video, which is a music video 
# (often containing vocals). The reaction video and base video are the 
# same length and the audio of the base video should be playing throughout 
# the reaction video, perfectly aligned. However, there may be noise and 
# volume differences. Can you write a python function that does its best 
# to isolate the reactor’s comments in the reaction video, given the base 
# video and the reaction video? Feel free to use Spleeter or any other 
# python libraries.






import numpy as np
import scipy.io.wavfile as wav
from sklearn.decomposition import FastICA
from pydub import AudioSegment


import librosa
from scipy.io.wavfile import write


import torch
import soundfile as sf
from asteroid.models import ConvTasNet

def separate_reactions_from_base_audio(base_video_path, reaction_video_path, output_path):
    # Load the audio files
    base_video = AudioSegment.from_file(base_video_path)
    reaction_video = AudioSegment.from_file(reaction_video_path)
    
    # # Get the raw audio data and the sample rate
    # base_audio_data = np.array(base_video.get_array_of_samples())
    # reaction_audio_data = np.array(reaction_video.get_array_of_samples())
    
    # # Make sure both audio data have the same size
    # min_size = min(base_audio_data.shape[0], reaction_audio_data.shape[0])
    # base_audio_data = base_audio_data[:min_size]
    # reaction_audio_data = reaction_audio_data[:min_size]


    # # Stack the two audio data sets together
    # audio_data = np.vstack((base_audio_data, reaction_audio_data))
    
    # # Initialize FastICA
    # ica = FastICA(n_components=2)
    
    # # Perform the ICA
    # ica.fit(audio_data.T)
    
    # # Get the independent components
    # independent_components = ica.transform(audio_data.T)
    
    # # Convert the independent components back into audio files
    # base_audio_ica = AudioSegment(
    #     independent_components[:, 0],
    #     frame_rate=base_video.frame_rate,
    #     sample_width=base_video.sample_width,
    #     channels=base_video.channels
    # )
    
    # reaction_audio_ica = AudioSegment(
    #     independent_components[:, 1],
    #     frame_rate=reaction_video.frame_rate,
    #     sample_width=reaction_video.sample_width,
    #     channels=reaction_video.channels
    # )
    
    # # Save the audio files
    # base_audio_ica.export(f"{output_path}-base.wav", format="wav")
    # reaction_audio_ica.export(f"{output_path}-reaction.wav", format="wav")







    # # Get the audio data and convert it to mono
    # base_audio_data = np.array(base_video.get_array_of_samples())
    # if base_video.channels == 2:
    #     base_audio_data = base_audio_data.reshape((-1, 2))
    # base_audio_mono = np.mean(base_audio_data, axis=1)

    # reaction_audio_data = np.array(reaction_video.get_array_of_samples())
    # if reaction_video.channels == 2:
    #     reaction_audio_data = reaction_audio_data.reshape((-1, 2))
    # reaction_audio_mono = np.mean(reaction_audio_data, axis=1)

    # # Make sure both audio data have the same size
    # min_size = min(base_audio_mono.shape[0], reaction_audio_mono.shape[0])
    # base_audio_mono = base_audio_mono[:min_size]
    # reaction_audio_mono = reaction_audio_mono[:min_size]

    # # Perform noise reduction (assuming base video's audio is the 'noise')
    # # This may need to be tweaked based on the actual audio data
    # noise_reduced_audio = reaction_audio_mono - base_audio_mono

    # # Convert to int16 format as required by scipy.io.wavfile.write
    # noise_reduced_audio = np.int16(noise_reduced_audio / np.max(np.abs(noise_reduced_audio)) * 32767)

    # # Save the output
    # write(output_path + '-np.wav', base_video.frame_rate, noise_reduced_audio)


    # Load pre-trained model
    model = ConvTasNet.from_pretrained('mpariente/ConvTasNet_WHAM!_sepclean')

    # Load the base and reaction videos audio as waveform

    path, ext = os.path.splitext(base_video_path)

    base_audio, sr = sf.read(path + '.wav')

    path, ext = os.path.splitext(reaction_video_path)    
    reaction_audio, _ = sf.read(os.path.join(path, os.path.basename(path) + '.wav'))

    # If audio is not mono, convert it to mono
    if base_audio.ndim > 1:
        base_audio = base_audio.mean(axis=-1)
    if reaction_audio.ndim > 1:
        reaction_audio = reaction_audio.mean(axis=-1)

    
    # Ensure audio clips have the same length
    min_length = min(base_audio.shape[0], reaction_audio.shape[0])
    base_audio = base_audio[:min_length]
    reaction_audio = reaction_audio[:min_length]

    # Reshape audio to (num_channels, num_samples)
    base_audio = base_audio.reshape(1, -1)
    reaction_audio = reaction_audio.reshape(1, -1)

    # Combine the audios into a single batch
    mixture = torch.stack([torch.Tensor(base_audio), torch.Tensor(reaction_audio)])

    # Perform the separation
    estimates = model.separate(mixture)

    # Save the separated audio tracks to the output path
    for idx, estimate in enumerate(estimates):
        sf.write(f'{output_path}-asteroid-{idx}.wav', estimate.detach().numpy(), sr)


