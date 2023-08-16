
##########################
# We have been trying to solve the following problem:

# I'm trying to isolate the sections of a reaction audio in which one or more commentators are making noise. 
# We also have the original song, and they are *mostly* aligned in time.  

# In my preprocessing, I’m separating audio by source into a vocal track, and ignoring the accompaniment. 
# This gets rid of most of the music from consideration, leaving us with vocals. 

# The first step in my algorithm for identifying parts of a reaction audio where commentators are speaking 
# is to identify the parts of the reaction audio that is most different from the base song (using MFCCs 
#     to get at perceptual differences). 

# To do this, I'm creating a mask that is true where the absolute difference between the two tracks is 
# greater than some threshold.

# Setting this threshold in a principled way is tricky. Our first attempt looks like: 

# def create_mask(diff, percentile):
#     # Calculate the nth percentile of the absolute differences
#     threshold = np.percentile(np.abs(diff), percentile)
#     # Create a mask where the absolute difference is greater than the threshold
#     mask = np.abs(diff) > threshold
#     return mask

# However, because I have separated out the vocals, much of both of the tracks are nearly silent. 
# This distorts the percentile approach to setting a threshold. 

# So what we've been working on has been trying to ignore the near-silent sections of the audio when 
# setting the threshold. We've been first calculating frame energy to identify near-silent parts of 
# the audio, and then trying to use that (unsucessfully) filter the MFCC diff used to compute a 
# threshold for masking the reaction audio.  

# We've been running into lots of errors though in the dimensions of the various arrays involved, 
# and in interpreting the silence threshold as a decibal=>energy mapping. I've included our latest 
# code below. Please fix it, or identify a different method for achieving the goal outlined above. Thanks!

# def calculate_mfcc(audio, sr):
#     # Calculate the MFCCs for the audio
#     mfcc = librosa.feature.mfcc(audio, sr)
#     return mfcc


# def calculate_difference(mfcc1, mfcc2):
#     # Calculate the difference between the two MFCCs
#     diff = np.abs(mfcc1 - mfcc2)
#     return diff


# def calculate_frame_energy(audio, frame_length, hop_length):
#     # Frame the audio signal
#     frames = librosa.util.frame(audio, frame_length, hop_length)
#     # Calculate the energy of each frame
#     energy = np.sum(frames**2, axis=0)
#     return energy

# def create_mask_by_mfcc_difference(song, sr_song, reaction, sr_reaction, percentile, silence_threshold=.001):


#     mfcc_song = calculate_mfcc(song, sr_song)
#     mfcc_reaction = calculate_mfcc(reaction, sr_reaction)

#     if mfcc_song.shape != mfcc_reaction.shape:
#         print("The shapes of the MFCCs do not match!")
#         return
        

#     print("mfcc_song shape: ", mfcc_song.shape)
#     print("mfcc_reaction shape: ", mfcc_reaction.shape)

#     diff = calculate_difference(mfcc_song, mfcc_reaction)


#     # Calculate the frame energy
#     frame_length = 2048  # You may need to adjust this
#     hop_length = 512  # and this
#     frame_energy = calculate_frame_energy(reaction, frame_length, hop_length)

#     # Create a mask where the frame energy is greater than the silence threshold
#     not_silent = frame_energy > silence_threshold

#     # Reduce the dimensionality of diff by calculating the Euclidean norm along the MFCC axis
#     diff_norm = np.linalg.norm(diff, axis=0)

#     # Check if not_silent contains any True values
#     if not np.any(not_silent):
#         print("All frames are silent. Please check your silence threshold.")
#         return None

#     # Reshape not_silent to match diff_norm
#     not_silent_frames = np.reshape(not_silent, (1, len(not_silent)))

#     # Repeat the not_silent_frames to match the rows in diff
#     not_silent = np.repeat(not_silent_frames, diff.shape[0], axis=0)

#     # Adjust the shape of not_silent to match the size of diff
#     if diff.shape[1] > not_silent.shape[1]:
#         padding = diff.shape[1] - not_silent.shape[1]
#         not_silent = np.pad(not_silent, ((0, 0), (0, padding)), constant_values=False)
        
#     # Mask diff with not_silent
#     diff_masked = diff.copy()
#     diff_masked[~not_silent] = 0

#     # Calculate the percentile of the absolute differences where not silent
#     threshold = np.percentile(np.abs(diff_masked), percentile)

#     # Create a mask where the absolute difference is greater than the threshold
#     mask = np.abs(diff) > threshold

#     return mask[0]




def subtract_song(song_path, reaction_path):

    mono = True
    # Load the song and reaction audio
    song, original_sr_song = librosa.load(song_path, sr=None, mono=mono)

    # Load reaction audio
    reaction, original_sr_reaction = librosa.load(reaction_path, sr=None, mono=mono)

    assert original_sr_reaction == original_sr_song, "Sample rates must be equal"

    target_sr = original_sr_reaction # * 4

    # Load the song and reaction audio
    song, sr_song = librosa.load(song_path, sr=target_sr, mono=mono)

    # Load reaction audio
    reaction, sr_reaction = librosa.load(reaction_path, sr=target_sr, mono=mono) 


    if mono:
        # Convert them to pseudo multi-channel (still mono but with an extra dimension to look like multi-channel)
        song = np.expand_dims(song, axis=0)
        reaction = np.expand_dims(reaction, axis=0)

    # Check the standard deviation of the audio data
    assert song.shape[0] == reaction.shape[0], f"Song and reaction must have the same number of channels (song = {song.shape[0]}, reaction = {reaction.shape[0]})"

    delay = 0
    if len(song[0]) > len(reaction[0]):
        song_padded = song[:,:len(reaction[0])]
    else:
        song_padded = np.pad(song, ((0, 0), (delay, len(reaction[0]) - len(song[0]) - delay)))


    commentary = np.zeros_like(reaction)


    def find_shift_energy(shift, song_audio, reaction_audio):
        # Shift the song
        if shift < 0:
            song_shifted = np.concatenate([song_audio[abs(shift):], np.zeros(abs(shift))])
        else:
            song_shifted = np.concatenate([np.zeros((shift,)), song_audio[:-shift]])

        # Ensure song_shifted and reaction_audio are the same length
        if len(song_shifted) < len(reaction_audio):
            song_shifted = np.concatenate([song_shifted, np.zeros(len(reaction_audio) - len(song_shifted))])
        elif len(song_shifted) > len(reaction_audio):
            song_shifted = song_shifted[:len(reaction_audio)]

        # Subtract the song from the reaction audio
        commentary_test = reaction_audio - song_shifted

        # Compute the energy excluding the first and last 3 seconds
        energy = np.sum(commentary_test[int(3*sr_reaction):-int(3*sr_reaction)]**2)

        # Update the minimum energy and best shift
        print(f"Considering shift {shift} [{energy}  {energy / min_energy}]", end='\r')

        return energy


    for channel in range(song.shape[0]):


        # Preprocessing steps
        reaction[channel] = match_audio(reaction[channel], song_padded[channel], sr_reaction)

        # The range of shifts to test
        no_shift = reaction[channel] - song_padded[channel]
        min_energy = np.sum(no_shift[int(3*sr_reaction):-int(3*sr_reaction)]**2)
        best_shift = 0

        

        for shift_range in [range(0,-100,-1), range(0,100,1)]:
            last_shift = float('inf')
            for shift in shift_range:
                energy = find_shift_energy(shift, song_padded[channel], reaction[channel])
                if energy < min_energy:
                    min_energy = energy
                    best_shift = shift
                # if last_shift < energy and (abs(shift) > 120 or min_energy * 4 < energy):
                #     break

                last_shift = energy


        print(f"\nBest shift is {best_shift} [{min_energy}] \n")

        if best_shift < 0:
            song_padded[channel] = np.concatenate([song_padded[channel, abs(best_shift):], np.zeros(abs(best_shift))])
        else:
            song_padded[channel] = np.concatenate([song_padded[channel, best_shift:], np.zeros(best_shift)])

        # Ensure same length before equalization
        if len(song_padded[channel]) > len(reaction[channel]):
            reaction[channel] = np.pad(reaction[channel], (0, len(song_padded[channel]) - len(reaction[channel])))
        else:
            song_padded[channel] = np.pad(song_padded[channel], (0, len(reaction[channel]) - len(song_padded[channel])))

        
        
        # Subtract the song from the reaction audio
        commentary[channel] = reaction[channel] - song_padded[channel]

        check_negative_version(reaction[channel], commentary[channel], song_padded[channel])



    if mono:
        # Squeeze the 2D arrays to 1D
        commentary = np.squeeze(commentary)

    commentary = librosa.resample(commentary, sr_reaction, original_sr_reaction)

    output_path = os.path.splitext(reaction_path)[0] + "_after_subtraction.wav"
    sf.write(output_path, commentary.T, original_sr_reaction)

    # commentary = post_process_audio(commentary, original_sr_reaction)
    # # Now you can save commentary into a file
    # output_path = os.path.splitext(reaction_path)[0] + "_commentary.wav"
    # sf.write(output_path, commentary.T, original_sr_reaction)  # Transpose the output because soundfile expects shape (n_samples, n_channels)

    return output_path


def check_negative_version(reaction_audio, resulting_audio, song_audio):
    # calculate correlations
    corr_resulting = np.corrcoef(reaction_audio, resulting_audio)[0,1]
    corr_inverted_song = np.corrcoef(reaction_audio, -song_audio)[0,1]
    
    # check which correlation is higher
    if corr_resulting > corr_inverted_song:
        print('The resulting audio seems not to be a negative version of the original.')
    else:
        print('The resulting audio may be a negative version of the original.')
    
    print('Correlation between reaction and resulting audio:', corr_resulting)
    print('Correlation between reaction and inverted song:', corr_inverted_song)

