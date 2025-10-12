import torch
import librosa
import numpy as np

def audio_to_stft_features(wav_path, n_fft=1024, hop_length=512):
    """
    Loads an audio file and converts it to an STFT magnitude spectrum tensor.
    
    Parameters:
    - wav_path (str): The path to the audio file.
    - n_fft (int): The FFT window size.
    - hop_length (int): The hop length between frames.
    
    Returns:
    - torch.Tensor: The STFT magnitude spectrum with shape (n_freq_bins, n_frames).
    """
    # Load the audio; sr=None preserves the original sampling rate
    signal, sr = librosa.load(wav_path, sr=None, mono=False)
    
    # We are processing two channels, here we use the first channel as an example for STFT.
    # The actual model would likely use features from both channels.
    stft_result = librosa.stft(signal[0], n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_result)
    
    return torch.from_numpy(magnitude).float()