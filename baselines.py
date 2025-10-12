
import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import FastICA

def separate_with_ica(mixed_wav_path, output_prefix='ica_separated'):
    """
    Separates a two-channel mixed audio signal using the FastICA algorithm.

    Parameters:
    - mixed_wav_path (str): The path to the input two-channel WAV file.
    - output_prefix (str): The prefix for the output files.
    """
    try:
        #Load the WAV file
        sample_rate, signal = wavfile.read(mixed_wav_path)
        
        # Ensure the signal is stereo (two-channel)
        if signal.ndim == 1 or signal.shape[1] != 2:
            print("Error: Input audio must be two-channel.")
            return
            
        #Convert the signal to floating-point and normalize it
        signal = signal.astype(np.float32) / 32767.0
        
        # Initialize and apply FastICA
        # n_components=2 means we want to separate two independent sources
        ica = FastICA(n_components=2, random_state=42)
        # scikit-learn's ICA expects data in the format (n_samples, n_features), where n_features is the number of channels
        separated_sources = ica.fit_transform(signal)
        
        #Restore the separated signals to their original amplitude range and save them
        for i in range(separated_sources.shape[1]):
            source = separated_sources[:, i]
            # De-normalize and scale back to the 16-bit PCM range
            source_int16 = (source / np.max(np.abs(source)) * 32767).astype(np.int16)
            output_filename = f"{output_prefix}_source_{i+1}.wav"
            wavfile.write(output_filename, sample_rate, source_int16)
            print(f"Saved separated source to: {output_filename}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
