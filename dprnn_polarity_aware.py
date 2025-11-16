"""
DPRNN fixphaseinvertversion
keyimprovement:
1. inLoss Functioninaddphase/polarity-aware
2. usingSDRauxiliaryloss
3. improvementPITlosscalculate
4. post-process automatic phase correction
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


# ==================== dataset ====================
class EnhancedMiniLibriMixDataset(Dataset):
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
        
        if self.split == 'train':
            scale = np.random.uniform(0.5, 1.5)
            mixture = mixture * scale
            source1 = source1 * scale
            source2 = source2 * scale
            
            if np.random.rand() < 0.3:
                noise_level = np.random.uniform(0.005, 0.02)
                noise = torch.randn_like(mixture) * noise_level
                mixture = mixture + noise
            
            if np.random.rand() < 0.5:
                shift = np.random.randint(-1600, 1600)
                mixture = torch.roll(mixture, shift)
                source1 = torch.roll(source1, shift)
                source2 = torch.roll(source2, shift)
        
        sources = torch.stack([source1, source2], dim=0)
        return mixture, sources


# ==================== Encodingmodule/Decoder ====================
class Encoder(nn.Module):
    def __init__(self, L, N):
        super().__init__()
        self.L = L
        self.N = N
        self.conv1d = nn.Conv1d(1, N, kernel_size=L, stride=L//2, bias=False)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        w = self.conv1d(x)
        return F.relu(w)


class Decoder(nn.Module):
    def __init__(self, L, N):
        super().__init__()
        self.L = L
        self.N = N
        self.deconv = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L//2, bias=False)
        
    def forward(self, x):
        out = self.deconv(x)
        return out.squeeze(1)


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


# ==================== DPRNN Block ====================
class DPRNNBlock(nn.Module):
    def __init__(self, hidden_dim, chunk_size, dropout=0.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        
        self.intra_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.intra_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.intra_norm = GlobalLayerNorm(hidden_dim)
        self.intra_dropout = nn.Dropout(dropout)
        
        self.inter_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.inter_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.inter_norm = GlobalLayerNorm(hidden_dim)
        self.inter_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, T = x.shape
        
        rest = T % self.chunk_size
        if rest > 0:
            pad_len = self.chunk_size - rest
            x = F.pad(x, (0, pad_len))
            T = x.shape[2]
        
        num_chunks = T // self.chunk_size
        
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
        
        return x


# ==================== mainmodel ====================
class PolarityAwareDPRNN(nn.Module):
    """phase-awareDPRNN - fixmirror problem"""
    
    def __init__(self, n_src=2, L=20, N=512, H=256, R=8, C=50, dropout=0.0):
        super().__init__()
        
        self.n_src = n_src
        self.N = N
        self.L = L
        
        self.encoder = Encoder(L, N)
        self.ln = GlobalLayerNorm(N)
        self.bottleneck = nn.Conv1d(N, H, 1)
        
        self.dprnn = nn.ModuleList([
            DPRNNBlock(H, C, dropout=dropout) for _ in range(R)
        ])
        
        # improvementmasknetwork - output sign and magnitude
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(H, N * n_src, 1),
        )
        
        self.decoder = Decoder(L, N)
        
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
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def forward(self, mixture):
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)
        
        orig_len = mixture.shape[-1]
        
        w = self.encoder(mixture)
        e = self.ln(w)
        e = self.bottleneck(e)
        
        s = e
        for dprnn_block in self.dprnn:
            s = dprnn_block(s)
        
        m = self.mask_net(s)
        B, _, K_mask = m.shape
        
        _, _, K_enc = w.shape
        if K_mask != K_enc:
            if K_mask > K_enc:
                m = m[:, :, :K_enc]
            else:
                m = F.pad(m, (0, K_enc - K_mask))
        
        m = m.view(B, self.n_src, self.N, K_enc)
        
        # usingtanhinstead ofsigmoid - allow negative values！
        m = torch.tanh(m)  # output range[-1, 1]
        
        w = w.unsqueeze(1)
        s = w * m
        
        separated = []
        for i in range(self.n_src):
            si = self.decoder(s[:, i])
            separated.append(si)
        
        separated = torch.stack(separated, dim=1)
        
        if separated.shape[-1] > orig_len:
            separated = separated[..., :orig_len]
        elif separated.shape[-1] < orig_len:
            separated = F.pad(separated, (0, orig_len - separated.shape[-1]))
        
        return separated


# ==================== improvementLoss Function ====================
def si_snr_loss(est, ref, eps=1e-8):
    """SI-SNRloss - phase invariant"""
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
    """SDRloss - phase sensitive"""
    signal_power = torch.sum(ref ** 2, dim=-1)
    noise_power = torch.sum((est - ref) ** 2, dim=-1)
    sdr = 10 * torch.log10(signal_power / (noise_power + eps) + eps)
    return -sdr


def polarity_aware_loss(est_sources, ref_sources, alpha_sdr=0.3):
    """
    phase-aware loss
    
    Args:
        alpha_sdr: SDRlossweight (for penalizing phaseinvert)
    """
    # permutation1: est[0]->ref[0], est[1]->ref[1]
    si_snr_1 = si_snr_loss(est_sources[:, 0], ref_sources[:, 0]) + \
               si_snr_loss(est_sources[:, 1], ref_sources[:, 1])
    
    sdr_1 = sdr_loss(est_sources[:, 0], ref_sources[:, 0]) + \
            sdr_loss(est_sources[:, 1], ref_sources[:, 1])
    
    loss_1 = si_snr_1 + alpha_sdr * sdr_1
    
    # permutation2: est[0]->ref[1], est[1]->ref[0]
    si_snr_2 = si_snr_loss(est_sources[:, 0], ref_sources[:, 1]) + \
               si_snr_loss(est_sources[:, 1], ref_sources[:, 0])
    
    sdr_2 = sdr_loss(est_sources[:, 0], ref_sources[:, 1]) + \
            sdr_loss(est_sources[:, 1], ref_sources[:, 0])
    
    loss_2 = si_snr_2 + alpha_sdr * sdr_2
    
    # select smallerloss
    loss = torch.min(loss_1, loss_2)
    
    return loss.mean()


# ==================== phase correctionafterprocessing ====================
def auto_correct_polarity(est_sources, ref_sources):
    """
    automaticcorrectpositivephaseinvert
    induring inferenceusing
    """
    corrected = est_sources.clone()
    
    for b in range(est_sources.shape[0]):
        for s in range(est_sources.shape[1]):
            est = est_sources[b, s]
            ref = ref_sources[b, s]
            
            # calculatepositivephaseandinversephasecorrelation
            corr_positive = torch.sum(est * ref)
            corr_negative = torch.sum((-est) * ref)
            
            # ifinversephasecorrelationhigher，then flip
            if corr_negative > corr_positive:
                corrected[b, s] = -est
    
    return corrected


# ==================== learning rateschedulingmodule ====================
class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, warmup_steps=0, 
                 max_lr=1e-3, min_lr=1e-6, gamma=1.0):
        self.first_cycle_steps = first_cycle_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.gamma = gamma
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = self.max_lr * (self.last_epoch + 1) / self.warmup_steps
            return [lr for _ in self.optimizer.param_groups]
        
        progress = (self.last_epoch - self.warmup_steps) / (self.first_cycle_steps - self.warmup_steps)
        lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        return [lr for _ in self.optimizer.param_groups]


# ==================== Training Functions ====================
def train_epoch(model, dataloader, optimizer, device, gradient_accumulation_steps=1, alpha_sdr=0.3):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc='Training')
    for i, (mixture, sources) in enumerate(pbar):
        mixture = mixture.to(device)
        sources = sources.to(device)
        
        est_sources = model(mixture)
        loss = polarity_aware_loss(est_sources, sources, alpha_sdr=alpha_sdr) / gradient_accumulation_steps
        
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, alpha_sdr=0.3):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for mixture, sources in tqdm(dataloader, desc='Evaluating'):
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            est_sources = model(mixture)
            loss = polarity_aware_loss(est_sources, sources, alpha_sdr=alpha_sdr)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# ==================== mainfunctioncount ====================
def main():
    config = {
        'data_root': './MiniLibriMix',
        'batch_size': 2,
        'num_epochs': 200,
        'learning_rate': 5e-4,
        'L': 20,
        'N': 512,
        'H': 256,
        'R': 8,
        'C': 50,
        'dropout': 0.1,
        'warmup_epochs': 10,
        'gradient_accumulation_steps': 2,
        'alpha_sdr': 0.3,  # SDRlossweight - keyparameters!
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 80)
    print("🚀 DPRNN phase-awareversion - fixmirror problem")
    print("=" * 80)
    print(f"\ndevice: {config['device']}")
    print(f"configuration: L={config['L']}, N={config['N']}, H={config['H']}, R={config['R']}, C={config['C']}")
    print(f"key: alpha_sdr={config['alpha_sdr']} (SDRweightforfixphase)")
    
    # data
    train_dataset = EnhancedMiniLibriMixDataset(config['data_root'], 'train', 'both')
    val_dataset = EnhancedMiniLibriMixDataset(config['data_root'], 'val', 'both')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                          shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # model
    model = PolarityAwareDPRNN(
        n_src=2,
        L=config['L'],
        N=config['N'],
        H=config['H'],
        R=config['R'],
        C=config['C'],
        dropout=config['dropout']
    ).to(config['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.2f}M")
    
    # optimizationmodule
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=config['num_epochs'],
        warmup_steps=config['warmup_epochs'],
        max_lr=config['learning_rate'],
        min_lr=1e-6
    )
    
    # Training
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*80}")
        
        train_loss = train_epoch(model, train_loader, optimizer, config['device'],
                                config['gradient_accumulation_steps'], config['alpha_sdr'])
        val_loss = evaluate(model, val_loader, config['device'], config['alpha_sdr'])
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\nTrainingloss: {train_loss:.4f} | Validationloss: {val_loss:.4f}")
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
            }, 'best_dprnn_polarity_aware.pth')
            
            print(f"Save best model | loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nearly stopping! consecutive{max_patience}epochno improvement")
                break
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss (SI-SNR + SDR)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('training_polarity_aware.png', dpi=300)
    
    print(f"\n{'='*80}")
    print(f"Trainingcompleted!")
    print(f"bestloss: {best_val_loss:.4f}")
    print(f"model: best_dprnn_polarity_aware.pth")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()