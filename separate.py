
import torch
import librosa
import soundfile as sf
import argparse
import os

from models import BiLSTMSeparator
from audio_utils import process_audio_for_model

def separate_audio(model_path, input_wav_path, output_dir, sample_rate=16000):
    """
    Loads a trained model to separate a single-channel input audio.
    """
    # Set up parameters and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N_FFT, HOP_LENGTH = 1024, 512
    FREQ_BINS = N_FFT // 2 + 1
    HIDDEN_SIZE, NUM_LAYERS = 256, 2
    NUM_SPEAKERS = 2

    # Load the model 
    model = BiLSTMSeparator(
        input_size=FREQ_BINS, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        num_speakers=NUM_SPEAKERS
    )
    # Ensure the correct model file is loaded
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    #Load audio
    mixed_signal, sr = librosa.load(input_wav_path, sr=sample_rate, mono=True)
    mixed_waveform = torch.from_numpy(mixed_signal).to(device)
    
    # Process audio using the utility function
    window = torch.hann_window(N_FFT).to(device)
    model_input, mix_stft = process_audio_for_model(
        waveform=mixed_waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        device=device
    )

    # Model inference
    with torch.no_grad():
        predicted_masks = model(model_input).transpose(2, 3) # (B, C, F, T)

    #Reconstruct audio
    # unsqueeze(1) is used because the mask has a speaker dimension (B,C,F,T) that mix_stft (B,F,T) lacks.
    predicted_stft = predicted_masks * mix_stft.unsqueeze(1) 
    
    B, C, F, T = predicted_stft.shape
    predicted_stft_reshaped = predicted_stft.reshape(B * C, F, T)
    predicted_sources = torch.istft(
        predicted_stft_reshaped, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        window=window, 
        length=len(mixed_signal)
    )
    predicted_sources = predicted_sources.reshape(B, C, -1).squeeze(0) # Remove the batch dimension
    predicted_sources = predicted_sources.cpu().numpy()
    
    #Save the results 
    os.makedirs(output_dir, exist_ok=True)
    for i in range(NUM_SPEAKERS):
        output_filename = os.path.join(output_dir, f"separated_source_{i+1}.wav")
        sf.write(output_filename, predicted_sources[i], sr)
        print(f"Saved separated source to: {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Separate a single-channel audio file using a trained BiLSTM model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth) file.')
    parser.add_argument('--input_wav', type=str, required=True, help='Path to the input single-channel WAV file.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save the output files.')
    args = parser.parse_args()
    separate_audio(args.model_path, args.input_wav, args.output_dir)