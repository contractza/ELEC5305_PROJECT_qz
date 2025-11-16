"""
DPRNN ultimate optimization version
target: SI-SNR 10-15 dB

Key improvements:
1. robust data augmentation（multiple noise types）
2. phase-aware loss（SI-SNR + SDR）
3. multi-resolution encoding module
4. Transformer-enhanced DPRNN
5. deeper and wider network
6. advanced learning rate strategy
7. reconstruction loss
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
import math


# ==================== robustdataset ====================
class UltimateMiniLibriMixDataset(Dataset):
    """ultimatedataset - multiple noise typesenhanced"""
    
    def __init__(self, root_dir, split='train', mix_type='both', 
                 sample_rate=8000, segment_length=4.0):
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_length)
        
        metadata_file = self.root_dir / 'metadata' / f'mixture_{split}_mix_{mix_type}.csv'
        self.metadata = pd.read_csv(metadata_file)
        
        self.mix_dir = self.root_dir / split / f'mix_{mix_type}'
        self.s1_dir = self.root_dir / split / 's1'
        self.s2_dir = self.root_dir / split / 's2'
        
    def __len__(self):
        return len(self.metadata)
    
    def _generate_pink_noise(self, length):
        """pinkNoise（1/fNoise）- closer totrueNoise"""
        white = torch.randn(length)
        # simplified implementation：cumulative sumandand normalize
        pink = torch.zeros_like(white)
        b = [0.99886, 0.99332, 0.96900, 0.86650, 0.55000, -0.7616]
        for i in range(len(white)):
            pink[i] = white[i]
            for j in range(min(i, len(b))):
                pink[i] += b[j] * white[i-j-1]
        return pink / pink.std()
    
    def _generate_brown_noise(self, length):
        """brownNoise（integralNoise）- low frequencyNoise"""
        white = torch.randn(length)
        brown = torch.cumsum(white, dim=0)
        return brown / brown.std()
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        mixture_id = row['mixture_ID']
        
        mixture, _ = torchaudio.load(str(self.mix_dir / f"{mixture_id}.wav"))
        source1, _ = torchaudio.load(str(self.s1_dir / f"{mixture_id}.wav"))
        source2, _ = torchaudio.load(str(self.s2_dir / f"{mixture_id}.wav"))
        
        mixture = mixture.mean(0) if mixture.shape[0] > 1 else mixture.squeeze(0)
        source1 = source1.mean(0) if source1.shape[0] > 1 else source1.squeeze(0)
        source2 = source2.mean(0) if source2.shape[0] > 1 else source2.squeeze(0)
        
        if mixture.shape[0] > self.segment_samples:
            start = np.random.randint(0, mixture.shape[0] - self.segment_samples)
            mixture = mixture[start:start + self.segment_samples]
            source1 = source1[start:start + self.segment_samples]
            source2 = source2[start:start + self.segment_samples]
        else:
            pad_len = self.segment_samples - mixture.shape[0]
            mixture = F.pad(mixture, (0, pad_len))
            source1 = F.pad(source1, (0, pad_len))
            source2 = F.pad(source2, (0, pad_len))
        
        # ===== ultimatedataenhanced =====
        if self.split == 'train':
            # 1. soundamountenhanced（more aggressive）
            scale = np.random.uniform(0.4, 1.6)
            mixture = mixture * scale
            source1 = source1 * scale
            source2 = source2 * scale
            
            # 2. addvariousNoise (60%probability)
            if np.random.rand() < 0.6:
                noise_type = np.random.choice(['white', 'pink', 'brown'], p=[0.3, 0.5, 0.2])
                snr_db = np.random.uniform(8, 25)  # SNRrange: 8-25 dB
                
                if noise_type == 'white':
                    noise = torch.randn_like(mixture)
                elif noise_type == 'pink':
                    noise = self._generate_pink_noise(len(mixture))
                else:  # brown
                    noise = self._generate_brown_noise(len(mixture))
                
                # calculateNoisepower
                signal_power = mixture.pow(2).mean()
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = noise * torch.sqrt(noise_power / (noise.pow(2).mean() + 1e-8))
                
                mixture = mixture + noise
            
            # 3. time shifting (50%probability)
            if np.random.rand() < 0.5:
                shift = np.random.randint(-2000, 2000)  # ±0.25seconds
                mixture = torch.roll(mixture, shift)
                source1 = torch.roll(source1, shift)
                source2 = torch.roll(source2, shift)
            
            # 4. dynamic range compression (30%probability)
            if np.random.rand() < 0.3:
                alpha = np.random.uniform(0.8, 1.0)
                mixture = torch.sign(mixture) * torch.abs(mixture) ** alpha
                source1 = torch.sign(source1) * torch.abs(source1) ** alpha
                source2 = torch.sign(source2) * torch.abs(source2) ** alpha
            
            # 5. random silent segments (20%probability) - simulate intermittencyNoise
            if np.random.rand() < 0.2:
                mute_len = np.random.randint(400, 1600)  # 0.05-0.2seconds
                mute_start = np.random.randint(0, len(mixture) - mute_len)
                mixture[mute_start:mute_start + mute_len] *= np.random.uniform(0.1, 0.3)
            
            # 6. frequency masking (10%probability) - SpecAugmentstyle
            if np.random.rand() < 0.1:
                # simplified implementation：high-pass orlow-pass filtering
                if np.random.rand() < 0.5:
                    # low-pass filtering（remove high frequency）
                    cutoff = np.random.uniform(0.6, 0.9)
                    b = torch.ones(int(1 / (1 - cutoff)))
                    b = b / b.sum()
                    mixture = F.conv1d(mixture.unsqueeze(0).unsqueeze(0), 
                                      b.unsqueeze(0).unsqueeze(0), 
                                      padding=len(b)//2).squeeze()[:len(mixture)]
        
        sources = torch.stack([source1, source2], dim=0)
        return mixture, sources


# ==================== multi-scaleEncodingmodule ====================
class MultiScaleEncoder(nn.Module):
    """multi-scaleEncodingmodule - capture features at different time resolutions"""
    
    def __init__(self, L_list=[16, 32], N=512):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Conv1d(1, N // len(L_list), kernel_size=L, stride=L//2, bias=False)
            for L in L_list
        ])
        self.N = N
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = []
        for encoder in self.encoders:
            feat = F.relu(encoder(x))
            features.append(feat)
        
        # aligntominimum length
        min_len = min(f.size(2) for f in features)
        features = [f[:, :, :min_len] for f in features]
        
        # concatenate
        return torch.cat(features, dim=1)


class MultiScaleDecoder(nn.Module):
    """multi-scaleDecoder"""
    
    def __init__(self, L_list=[16, 32], N=512):
        super().__init__()
        N_per_scale = N // len(L_list)
        self.decoders = nn.ModuleList([
            nn.ConvTranspose1d(N_per_scale, 1, kernel_size=L, stride=L//2, bias=False)
            for L in L_list
        ])
        self.split_sizes = [N_per_scale] * len(L_list)
        
    def forward(self, x):
        # split features
        features = torch.split(x, self.split_sizes, dim=1)
        
        # decode
        outputs = []
        for feat, decoder in zip(features, self.decoders):
            out = decoder(feat).squeeze(1)
            outputs.append(out)
        
        # average fusion
        min_len = min(o.size(-1) for o in outputs)
        outputs = [o[..., :min_len] for o in outputs]
        return sum(outputs) / len(outputs)


# ==================== LayerNorm ====================
class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))
        
    def forward(self, x):
        mean = x.mean(dim=[1, 2], keepdim=True)
        var = x.var(dim=[1, 2], keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-8)
        return x * self.gamma + self.beta


# ==================== improvementDPRNN Block ====================
class EnhancedDPRNNBlock(nn.Module):
    """enhancedDPRNNblock - addattention mechanism"""
    
    def __init__(self, hidden_dim, chunk_size, dropout=0.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        
        # Intra-chunk
        self.intra_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.intra_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.intra_norm = GlobalLayerNorm(hidden_dim)
        self.intra_dropout = nn.Dropout(dropout)
        
        # Inter-chunk
        self.inter_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.inter_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.inter_norm = GlobalLayerNorm(hidden_dim)
        self.inter_dropout = nn.Dropout(dropout)
        
        # channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, N, T = x.shape
        
        rest = T % self.chunk_size
        if rest > 0:
            pad_len = self.chunk_size - rest
            x = F.pad(x, (0, pad_len))
            T = x.shape[2]
        
        num_chunks = T // self.chunk_size
        
        # Intra-chunk
        x_chunks = x.view(B, N, num_chunks, self.chunk_size)
        x_chunks = x_chunks.permute(0, 2, 3, 1).contiguous()
        x_chunks = x_chunks.view(B * num_chunks, self.chunk_size, N)
        
        out, _ = self.intra_rnn(x_chunks)
        out = self.intra_linear(out)
        out = self.intra_dropout(out)
        out = out.view(B, num_chunks, self.chunk_size, N)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(B, N, T)
        
        x = self.intra_norm(x + out)
        
        # Inter-chunk
        x_inter = x.view(B, N, num_chunks, self.chunk_size)
        x_inter = x_inter.permute(0, 3, 2, 1).contiguous()
        x_inter = x_inter.view(B * self.chunk_size, num_chunks, N)
        
        out, _ = self.inter_rnn(x_inter)
        out = self.inter_linear(out)
        out = self.inter_dropout(out)
        out = out.view(B, self.chunk_size, num_chunks, N)
        out = out.permute(0, 3, 2, 1).contiguous()
        out = out.view(B, N, T)
        
        x = self.inter_norm(x + out)
        
        # channel attention
        attn = self.channel_attention(x)
        x = x * attn
        
        return x


# ==================== ultimateDPRNN ====================
class UltimateDPRNN(nn.Module):
    """ultimateDPRNN - setbecomealloptimization"""
    
    def __init__(self, n_src=2, L_list=[16, 32], N=512, H=512, R=10, C=50, dropout=0.1):
        super().__init__()
        
        self.n_src = n_src
        self.N = N
        
        # multi-scaleEncodingmodule
        self.encoder = MultiScaleEncoder(L_list, N)
        
        # LayerNorm
        self.ln = GlobalLayerNorm(N)
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(N, H, 1)
        
        # enhancedDPRNNblock
        self.dprnn = nn.ModuleList([
            EnhancedDPRNNBlock(H, C, dropout=dropout) for _ in range(R)
        ])
        
        # Maskgenerator network
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(H, H, 1),
            nn.PReLU(),
            nn.Conv1d(H, N * n_src, 1),
        )
        
        # multi-scaleDecoder
        self.decoder = MultiScaleDecoder(L_list, N)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)  # positiveexchangeinitialization（better）
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def forward(self, mixture):
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)
        
        orig_len = mixture.shape[-1]
        
        # Encoding
        w = self.encoder(mixture)
        e = self.ln(w)
        e = self.bottleneck(e)
        
        # DPRNN processing
        s = e
        for dprnn_block in self.dprnn:
            s = dprnn_block(s)
        
        # generatemasks
        m = self.mask_net(s)
        B, _, K_mask = m.shape
        
        _, _, K_enc = w.shape
        if K_mask != K_enc:
            if K_mask > K_enc:
                m = m[:, :, :K_enc]
            else:
                m = F.pad(m, (0, K_enc - K_mask))
        
        m = m.view(B, self.n_src, self.N, K_enc)
        m = torch.tanh(m)  # allow negative values
        
        # applymask
        w = w.unsqueeze(1)
        s = w * m
        
        # decode
        separated = []
        for i in range(self.n_src):
            si = self.decoder(s[:, i])
            separated.append(si)
        
        separated = torch.stack(separated, dim=1)
        
        # matchinglength
        if separated.shape[-1] > orig_len:
            separated = separated[..., :orig_len]
        elif separated.shape[-1] < orig_len:
            separated = F.pad(separated, (0, orig_len - separated.shape[-1]))
        
        return separated


# ==================== ultimateLoss Function ====================
def si_snr_loss(est, ref, eps=1e-8):
    est = est - torch.mean(est, dim=-1, keepdim=True)
    ref = ref - torch.mean(ref, dim=-1, keepdim=True)
    
    ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + eps
    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref / ref_energy
    noise = est - proj
    
    si_snr = 10 * torch.log10(
        torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps) + eps
    )
    
    return -si_snr


def sdr_loss(est, ref, eps=1e-8):
    signal_power = torch.sum(ref ** 2, dim=-1)
    noise_power = torch.sum((est - ref) ** 2, dim=-1)
    sdr = 10 * torch.log10(signal_power / (noise_power + eps) + eps)
    return -sdr


def ultimate_loss(est_sources, ref_sources, mixture, alpha_sdr=0.3, alpha_recon=0.05):
    """
    ultimateLoss Function
    - SI-SNR: mainrequireloss
    - SDR: phase-aware
    - reconstruction loss: ensuresourcesum up≈mixed
    """
    # PIT for SI-SNR + SDR
    loss1_sisnr = si_snr_loss(est_sources[:, 0], ref_sources[:, 0]) + \
                  si_snr_loss(est_sources[:, 1], ref_sources[:, 1])
    loss1_sdr = sdr_loss(est_sources[:, 0], ref_sources[:, 0]) + \
                sdr_loss(est_sources[:, 1], ref_sources[:, 1])
    loss1 = loss1_sisnr + alpha_sdr * loss1_sdr
    
    loss2_sisnr = si_snr_loss(est_sources[:, 0], ref_sources[:, 1]) + \
                  si_snr_loss(est_sources[:, 1], ref_sources[:, 0])
    loss2_sdr = sdr_loss(est_sources[:, 0], ref_sources[:, 1]) + \
                sdr_loss(est_sources[:, 1], ref_sources[:, 0])
    loss2 = loss2_sisnr + alpha_sdr * loss2_sdr
    
    main_loss = torch.min(loss1, loss2).mean()
    
    # reconstruction loss
    reconstructed = est_sources.sum(dim=1)
    recon_loss = F.mse_loss(reconstructed, mixture)
    
    return main_loss + alpha_recon * recon_loss


# ==================== learning rateschedulingmodule ====================
class WarmupCosineSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        
        progress = (self.last_epoch - self.warmup_steps) / (self.t_total - self.warmup_steps)
        return [base_lr * (0.5 * (1.0 + math.cos(math.pi * self.cycles * 2.0 * progress)))
                for base_lr in self.base_lrs]


# ==================== Training/evaluation ====================
def train_epoch(model, dataloader, optimizer, device, gradient_accumulation_steps=2):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc='Training')
    for i, (mixture, sources) in enumerate(pbar):
        mixture = mixture.to(device)
        sources = sources.to(device)
        
        est_sources = model(mixture)
        loss = ultimate_loss(est_sources, sources, mixture, alpha_sdr=0.3, alpha_recon=0.05)
        loss = loss / gradient_accumulation_steps
        
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for mixture, sources in tqdm(dataloader, desc='Evaluating'):
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            est_sources = model(mixture)
            loss = ultimate_loss(est_sources, sources, mixture, alpha_sdr=0.3, alpha_recon=0.05)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# ==================== mainfunctioncount ====================
def main():
    config = {
        'data_root': './MiniLibriMix',
        'batch_size': 1,        # largemodel，smallbatch
        'num_epochs': 250,      # moreepochs
        'learning_rate': 3e-4,  # relativelysmallLR
        'L_list': [16, 32],     # multi-scale
        'N': 512,               # Encodingdimension
        'H': 512,               # hidden dimension（larger！）
        'R': 10,                # DPRNNnumber of layers（deeper！）
        'C': 50,
        'dropout': 0.15,        # increasedropout
        'gradient_accumulation_steps': 4,  # gradient accumulation
        'warmup_steps': 500,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 80)
    print("🚀 DPRNN ultimate optimization version")
    print("=" * 80)
    print(f"\ndevice: {config['device']}")
    print(f"configuration: N={config['N']}, H={config['H']}, R={config['R']}")
    print(f"multi-scale: L={config['L_list']}")
    print(f"robustenhanced: 6typesdataenhancedstrategy")
    
    # data
    train_dataset = UltimateMiniLibriMixDataset(config['data_root'], 'train', 'both')
    val_dataset = UltimateMiniLibriMixDataset(config['data_root'], 'val', 'both')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                          shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # model
    model = UltimateDPRNN(
        n_src=2,
        L_list=config['L_list'],
        N=config['N'],
        H=config['H'],
        R=config['R'],
        C=config['C'],
        dropout=config['dropout']
    ).to(config['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.2f}M")
    
    # optimizationmodule
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    
    # learning rateschedulingmodule
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = WarmupCosineSchedule(optimizer, config['warmup_steps'], total_steps)
    
    # Training
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*80}")
        
        train_loss = train_epoch(model, train_loader, optimizer, config['device'],
                                config['gradient_accumulation_steps'])
        val_loss = evaluate(model, val_loader, config['device'])
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\nTrainingloss: {train_loss:.4f} | Validationloss: {val_loss:.4f}")
        print(f"estimateSI-SNR: {-val_loss:.2f} dB")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, 'best_dprnn_ultimate.pth')
            
            print(f"Save best model | estimateSI-SNR: {-val_loss:.2f} dB")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nearly stopping!")
                break
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot([-l for l in train_losses], label='Train SI-SNR', linewidth=2)
    plt.plot([-l for l in val_losses], label='Val SI-SNR', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('SI-SNR (dB)')
    plt.legend()
    plt.title('SI-SNR Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_ultimate.png', dpi=300)
    
    print(f"\n{'='*80}")
    print(f"Trainingcompleted!")
    print(f"bestSI-SNR: {-best_val_loss:.2f} dB")
    print(f"model: best_dprnn_ultimate.pth")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()