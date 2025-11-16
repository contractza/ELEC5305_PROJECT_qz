"""
Dual-Path RNN for Speech Separation
Custom implementation based on DPRNN architecture for MiniLibriMix dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


# ===================== Dataset =====================
class MiniLibriMixDataset(Dataset):
    """MiniLibriMix dataset loader"""
    
    def __init__(self, root_dir, split='train', mix_type='both', 
                 sample_rate=8000, segment_length=4.0):
        """
        Args:
            root_dir: Dataset root directory
            split: 'train' or 'val'
            mix_type: 'both' or 'clean'
            sample_rate: Audio sample rate
            segment_length: Audio segment length in seconds
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.mix_type = mix_type
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_length)
        
        # Load metadata
        metadata_file = self.root_dir / 'metadata' / f'mixture_{split}_mix_{mix_type}.csv'
        self.metadata = pd.read_csv(metadata_file)
        
        # Construct file paths
        self.mix_dir = self.root_dir / split / f'mix_{mix_type}'
        self.s1_dir = self.root_dir / split / 's1'
        self.s2_dir = self.root_dir / split / 's2'
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Load mixture audio and separated source audios"""
        row = self.metadata.iloc[idx]
        
        # Get mixture_ID
        mixture_id = row['mixture_ID']
        
        # Construct file paths
        mix_path = self.mix_dir / f"{mixture_id}.wav"
        s1_path = self.s1_dir / f"{mixture_id}.wav"
        s2_path = self.s2_dir / f"{mixture_id}.wav"
        
        # Load audio files
        try:
            mixture, sr = torchaudio.load(str(mix_path))
            source1, _ = torchaudio.load(str(s1_path))
            source2, _ = torchaudio.load(str(s2_path))
        except Exception as e:
            raise RuntimeError(f"Error loading audio for {mixture_id}: {e}")
        
        # Convert to mono
        if mixture.shape[0] > 1:
            mixture = mixture.mean(dim=0, keepdim=True)
        if source1.shape[0] > 1:
            source1 = source1.mean(dim=0, keepdim=True)
        if source2.shape[0] > 1:
            source2 = source2.mean(dim=0, keepdim=True)
        
        # Random crop to fixed length
        if mixture.shape[1] > self.segment_samples:
            start = np.random.randint(0, mixture.shape[1] - self.segment_samples)
            mixture = mixture[:, start:start + self.segment_samples]
            source1 = source1[:, start:start + self.segment_samples]
            source2 = source2[:, start:start + self.segment_samples]
        else:
            # Padding
            pad_len = self.segment_samples - mixture.shape[1]
            mixture = F.pad(mixture, (0, pad_len))
            source1 = F.pad(source1, (0, pad_len))
            source2 = F.pad(source2, (0, pad_len))
        
        # Stack two sources
        sources = torch.cat([source1, source2], dim=0)  # [2, T]
        
        return mixture.squeeze(0), sources  # [T], [2, T]


# ===================== Model Components =====================
class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization - supports 3D and 4D tensors"""
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1, 1))
        
    def forward(self, x):
        # x: [B, N, K, S] for 4D or [B, N, K] for 3D
        if x.dim() == 4:
            # 4D: [B, N, K, S]
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        else:
            # 3D: [B, N, K]
            mean = x.mean(dim=(1, 2), keepdim=True)
            var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        
        x = (x - mean) / torch.sqrt(var + 1e-8)
        
        # Adjust gamma and beta dimensions to match input
        if x.dim() == 3:
            gamma = self.gamma.squeeze(-1)
            beta = self.beta.squeeze(-1)
        else:
            gamma = self.gamma
            beta = self.beta
            
        return x * gamma + beta


class DPRNNBlock(nn.Module):
    """Dual-Path RNN Block: contains intra-chunk and inter-chunk RNN"""
    
    def __init__(self, hidden_dim, rnn_type='LSTM', bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # Intra-chunk RNN (along time dimension)
        if rnn_type == 'LSTM':
            self.intra_rnn = nn.LSTM(hidden_dim, hidden_dim, 
                                     bidirectional=bidirectional, 
                                     batch_first=True)
        else:
            self.intra_rnn = nn.GRU(hidden_dim, hidden_dim,
                                    bidirectional=bidirectional,
                                    batch_first=True)
        
        # Inter-chunk RNN (along frequency/chunk dimension)
        if rnn_type == 'LSTM':
            self.inter_rnn = nn.LSTM(hidden_dim, hidden_dim,
                                     bidirectional=bidirectional,
                                     batch_first=True)
        else:
            self.inter_rnn = nn.GRU(hidden_dim, hidden_dim,
                                    bidirectional=bidirectional,
                                    batch_first=True)
        
        # Fully connected layers
        rnn_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.intra_fc = nn.Linear(rnn_out_dim, hidden_dim)
        self.inter_fc = nn.Linear(rnn_out_dim, hidden_dim)
        
        # Layer normalization
        self.intra_norm = GlobalLayerNorm(hidden_dim)
        self.inter_norm = GlobalLayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, K, S] - Batch, Channels, Chunks, Chunk_size
        """
        B, N, K, S = x.shape
        
        # Intra-chunk processing
        intra_input = x.permute(0, 2, 3, 1).contiguous()  # [B, K, S, N]
        intra_input = intra_input.view(B * K, S, N)  # [B*K, S, N]
        
        intra_out, _ = self.intra_rnn(intra_input)  # [B*K, S, N*2]
        intra_out = self.intra_fc(intra_out)  # [B*K, S, N]
        
        intra_out = intra_out.view(B, K, S, N)  # [B, K, S, N]
        intra_out = intra_out.permute(0, 3, 1, 2).contiguous()  # [B, N, K, S]
        
        # Residual connection
        x = self.intra_norm(x + intra_out)
        
        # Inter-chunk processing
        inter_input = x.permute(0, 3, 2, 1).contiguous()  # [B, S, K, N]
        inter_input = inter_input.view(B * S, K, N)  # [B*S, K, N]
        
        inter_out, _ = self.inter_rnn(inter_input)  # [B*S, K, N*2]
        inter_out = self.inter_fc(inter_out)  # [B*S, K, N]
        
        inter_out = inter_out.view(B, S, K, N)  # [B, S, K, N]
        inter_out = inter_out.permute(0, 3, 2, 1).contiguous()  # [B, N, K, S]
        
        # Residual connection
        x = self.inter_norm(x + inter_out)
        
        return x


class DPRNN(nn.Module):
    """Dual-Path RNN Speech Separation Model"""
    
    def __init__(self, 
                 n_src=2,           # Number of sources to separate
                 n_fft=256,         # FFT size
                 hop_length=64,     # Hop length
                 hidden_dim=128,    # Hidden dimension
                 num_blocks=6,      # Number of DPRNN blocks
                 chunk_size=100,    # Number of frames per chunk
                 rnn_type='LSTM'):
        super().__init__()
        
        self.n_src = n_src
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_size = chunk_size
        
        # Encoder: convolutional layer for feature extraction
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=n_fft, 
                     stride=hop_length, padding=n_fft // 2),
            nn.ReLU()
        )
        
        # Bottleneck layer
        self.bottleneck = nn.Conv1d(hidden_dim, hidden_dim, 1)
        
        # DPRNN modules
        self.dprnn_blocks = nn.ModuleList([
            DPRNNBlock(hidden_dim, rnn_type=rnn_type)
            for _ in range(num_blocks)
        ])
        
        # Mask generation
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * n_src, 1),
            nn.Sigmoid()
        )
        
        # Decoder
        self.decoder = nn.ConvTranspose1d(
            hidden_dim, 1, kernel_size=n_fft,
            stride=hop_length, padding=n_fft // 2
        )
        
    def pad_signal(self, x):
        """Pad signal to make it divisible by chunk_size"""
        batch, _, length = x.shape
        
        # Calculate required total length
        rest = length % (self.chunk_size * self.hop_length)
        if rest != 0:
            pad_len = self.chunk_size * self.hop_length - rest
            x = F.pad(x, (0, pad_len))
        
        return x, length
    
    def forward(self, mixture):
        """
        Args:
            mixture: [B, T] Mixed audio
        Returns:
            separated: [B, n_src, T] Separated audio
        """
        # Add channel dimension
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)  # [B, 1, T]
        
        # Padding
        mixture, orig_length = self.pad_signal(mixture)
        
        # Encoding
        enc_output = self.encoder(mixture)  # [B, N, L]
        bottleneck = self.bottleneck(enc_output)  # [B, N, L]
        
        B, N, L = bottleneck.shape
        
        # Reshape into chunks
        K = L // self.chunk_size  # Number of chunks
        if K * self.chunk_size < L:
            K += 1
            bottleneck = F.pad(bottleneck, (0, K * self.chunk_size - L))
        
        # [B, N, K, S] - K chunks, each chunk has S frames
        x = bottleneck.view(B, N, K, self.chunk_size)
        
        # DPRNN processing
        for block in self.dprnn_blocks:
            x = block(x)
        
        # Reshape back to original shape
        x = x.view(B, N, -1)[:, :, :L]  # [B, N, L]
        
        # Generate masks
        masks = self.mask_net(x.unsqueeze(-1))  # [B, N*n_src, L, 1]
        masks = masks.view(B, self.n_src, N, L)  # [B, n_src, N, L]
        
        # Apply masks
        separated = []
        for i in range(self.n_src):
            mask = masks[:, i]  # [B, N, L]
            masked = enc_output * mask  # [B, N, L]
            decoded = self.decoder(masked)  # [B, 1, T]
            separated.append(decoded)
        
        separated = torch.stack(separated, dim=1)  # [B, n_src, 1, T]
        separated = separated.squeeze(2)  # [B, n_src, T]
        
        # Crop to original length
        separated = separated[:, :, :orig_length]
        
        return separated


# ===================== Loss Function =====================
class SISNRLoss(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, est_sources, target_sources):
        """
        Args:
            est_sources: [B, n_src, T]
            target_sources: [B, n_src, T]
        """
        # Zero-mean normalization
        est_sources = est_sources - est_sources.mean(dim=-1, keepdim=True)
        target_sources = target_sources - target_sources.mean(dim=-1, keepdim=True)
        
        # Calculate projection
        s_target = (torch.sum(est_sources * target_sources, dim=-1, keepdim=True) / 
                   (torch.sum(target_sources ** 2, dim=-1, keepdim=True) + 1e-8)) * target_sources
        
        # Noise
        e_noise = est_sources - s_target
        
        # SI-SNR
        si_snr = 10 * torch.log10(
            (torch.sum(s_target ** 2, dim=-1) + 1e-8) / 
            (torch.sum(e_noise ** 2, dim=-1) + 1e-8)
        )
        
        return -si_snr.mean()


def permutation_invariant_loss(loss_fn, est_sources, target_sources):
    """Permutation Invariant Loss - handles source permutation problem"""
    # est_sources: [B, n_src, T]
    # target_sources: [B, n_src, T]
    
    n_src = est_sources.shape[1]
    
    if n_src == 2:
        # For 2 sources, only need to compare two permutations
        loss1 = loss_fn(est_sources, target_sources)
        loss2 = loss_fn(est_sources[:, [1, 0], :], target_sources)
        return torch.min(loss1, loss2)
    else:
        # More sources require iterating through all permutations
        import itertools
        min_loss = float('inf')
        for perm in itertools.permutations(range(n_src)):
            perm_est = est_sources[:, list(perm), :]
            loss = loss_fn(perm_est, target_sources)
            min_loss = min(min_loss, loss)
        return min_loss


# ===================== Training Functions =====================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for mixture, sources in pbar:
        mixture = mixture.to(device)  # [B, T]
        sources = sources.to(device)  # [B, 2, T]
        
        # Forward pass
        est_sources = model(mixture)  # [B, 2, T]
        
        # Calculate permutation invariant loss
        loss = permutation_invariant_loss(criterion, est_sources, sources)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for mixture, sources in tqdm(dataloader, desc='Evaluating'):
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            est_sources = model(mixture)
            loss = permutation_invariant_loss(criterion, est_sources, sources)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# ===================== Main Training Loop =====================
def main():
    # Hyperparameters
    config = {
        'data_root': './MiniLibriMix',
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'hidden_dim': 128,
        'num_blocks': 6,
        'chunk_size': 100,
        'n_fft': 256,
        'hop_length': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Create datasets
    train_dataset = MiniLibriMixDataset(
        root_dir=config['data_root'],
        split='train',
        mix_type='both'
    )
    
    val_dataset = MiniLibriMixDataset(
        root_dir=config['data_root'],
        split='val',
        mix_type='both'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = DPRNN(
        n_src=2,
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        hidden_dim=config['hidden_dim'],
        num_blocks=config['num_blocks'],
        chunk_size=config['chunk_size']
    ).to(config['device'])
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = SISNRLoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        train_losses.append(train_loss)
        
        # Validation
        val_loss = evaluate(model, val_loader, criterion, config['device'])
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Learning rate adjustment
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, 'best_dprnn_model.pth')
            print(f"Saved best model with val loss: {val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Negative SI-SNR)')
    plt.legend()
    plt.title('Training Progress')
    plt.savefig('training_curve.png')
    plt.close()
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
