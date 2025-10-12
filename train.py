
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import MiniLibriMixDataset
# train.py (Corrected code, line 10)
from models import BiLSTMSeparator
from audio_utils import process_audio_for_model

def collate_fn_pad(batch):
    mixes, sources = zip(*batch)
    sources_transposed = [s.transpose(0, 1) for s in sources]
    padded_mixes = pad_sequence(mixes, batch_first=True, padding_value=0.0)
    padded_sources_transposed = pad_sequence(sources_transposed, batch_first=True, padding_value=0.0)
    padded_sources = padded_sources_transposed.transpose(1, 2)
    return padded_mixes, padded_sources

def spectral_mse_loss(preds_mag, targets_mag):
    loss = nn.functional.mse_loss(preds_mag, targets_mag)
    return loss

def pit_spectral_loss(preds_mag, targets_mag):
    loss1 = spectral_mse_loss(preds_mag, targets_mag)
    permuted_targets_mag = targets_mag[:, [1, 0], :, :]
    loss2 = spectral_mse_loss(preds_mag, permuted_targets_mag)
    min_loss, _ = torch.min(torch.stack([loss1, loss2], dim=0), dim=0)
    return min_loss

def get_features(waveform, n_fft, hop_length, window):
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    magnitude = torch.abs(stft)
    #Use log-magnitude spectrum
    log_magnitude = torch.log1p(magnitude)
    return log_magnitude, stft

#Main Function
def main():
    # Parameter Settings
    DATASET_ROOT = r"C:\Users\15531\Desktop\elec5305project\MyProject\datasets\MiniLibriMix\MiniLibriMix"
    N_FFT, HOP_LENGTH = 1024, 512
    FREQ_BINS = N_FFT // 2 + 1
    HIDDEN_SIZE, NUM_LAYERS = 256, 2
    BATCH_SIZE, EPOCHS, LEARNING_RATE = 4, 30, 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRADIENT_CLIP_VAL = 5.0
    
    #Data Loading
    train_dataset = MiniLibriMixDataset(dataset_path=DATASET_ROOT, split='train')
    val_dataset = MiniLibriMixDataset(dataset_path=DATASET_ROOT, split='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn_pad)
    print("Dataset and DataLoader are ready.")

    #Model, Optimizer, and Scheduler
    model = BiLSTMSeparator(input_size=FREQ_BINS, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Make the scheduler more sensitive
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    window = torch.hann_window(N_FFT).to(DEVICE)
    print(f"Model created, will use device: {DEVICE}")

    best_val_loss = float('inf')

    #Training and Validation Loop
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for mix_batch, sources_batch in train_loader:
            mix_batch, sources_batch = mix_batch.to(DEVICE), sources_batch.to(DEVICE)

            # Feature Extraction
            mix_log_mag, mix_stft = get_features(mix_batch, N_FFT, HOP_LENGTH, window)
            
            # Normalize the log-magnitude spectrum
            mean = mix_log_mag.mean(dim=(1, 2), keepdim=True)
            std = mix_log_mag.std(dim=(1, 2), keepdim=True)
            model_input = (mix_log_mag - mean) / (std + 1e-8)
            model_input = model_input.transpose(1, 2)
            
            #Model Prediction
            predicted_masks = model(model_input).transpose(2, 3)
            
            #  Calculate Loss in the Frequency Domain
            predicted_mag = predicted_masks * torch.abs(mix_stft).unsqueeze(1)
            
            # Calculate the target spectrum
            target_s1_mag, _ = get_features(sources_batch[:, 0], N_FFT, HOP_LENGTH, window)
            target_s2_mag, _ = get_features(sources_batch[:, 1], N_FFT, HOP_LENGTH, window)
            target_mag = torch.stack([target_s1_mag, target_s2_mag], dim=1)

            loss = pit_spectral_loss(predicted_mag, target_mag)
            
            #Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for mix_batch, sources_batch in val_loader:
                mix_batch, sources_batch = mix_batch.to(DEVICE), sources_batch.to(DEVICE)
                
                mix_log_mag, mix_stft = get_features(mix_batch, N_FFT, HOP_LENGTH, window)
                
                mean = mix_log_mag.mean(dim=(1, 2), keepdim=True)
                std = mix_log_mag.std(dim=(1, 2), keepdim=True)
                model_input = (mix_log_mag - mean) / (std + 1e-8)
                model_input = model_input.transpose(1, 2)
                
                predicted_masks = model(model_input).transpose(2, 3)
                predicted_mag = predicted_masks * torch.abs(mix_stft).unsqueeze(1)
                
                target_s1_mag, _ = get_features(sources_batch[:, 0], N_FFT, HOP_LENGTH, window)
                target_s2_mag, _ = get_features(sources_batch[:, 1], N_FFT, HOP_LENGTH, window)
                target_mag = torch.stack([target_s1_mag, target_s2_mag], dim=1)

                val_loss = pit_spectral_loss(predicted_mag, target_mag)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'--- Epoch {epoch+1}/{EPOCHS} ---')
        print(f'Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "bilstm_separator_best_log_spectral.pth")
            print(f'Model updated and saved to bilstm_separator_best_log_spectral.pth (Validation Loss: {best_val_loss:.4f})')

    print("\n--- Training Complete ---")

if __name__ == '__main__':
    main()