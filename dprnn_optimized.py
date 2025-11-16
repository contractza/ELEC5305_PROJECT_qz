"""
DPRNNfinal optimized version - fix allbug + performance optimization
keyimprovement:
1. fixverboseparameterserror
2. optimizationHyperparametersconfiguration（N=512, L=20etc）
3. usingSigmoidactivationfunctioncount
4. addlearning ratewarmup
5. improvementTrainingpipeline
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


# ==================== dataset ====================
class MiniLibriMixDataset(Dataset):
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
        
        # dataenhanced（onlyTrainingset）
        if self.split == 'train':
            # random volume scaling
            scale = np.random.uniform(0.7, 1.3)
            mixture = mixture * scale
            source1 = source1 * scale
            source2 = source2 * scale
        
        sources = torch.stack([source1, source2], dim=0)
        return mixture, sources


# ==================== Encodingmodule/Decoder ====================
class Encoder(nn.Module):
    """TasNetstyleEncodingmodule"""
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
    """improvementDecoder - usingConvTranspose1densuredimensionmatching"""
    def __init__(self, L, N):
        super().__init__()
        self.L = L
        self.N = N
        # usingConvTranspose1d，parametersvsEncodercorresponding
        self.deconv = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L//2, bias=False)
        
    def forward(self, x):
        # x: [B, N, K]
        out = self.deconv(x)  # [B, 1, T']
        return out.squeeze(1)  # [B, T']


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
    def __init__(self, hidden_dim, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        
        # Intra-chunk
        self.intra_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.intra_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.intra_norm = GlobalLayerNorm(hidden_dim)
        
        # Inter-chunk
        self.inter_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.inter_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.inter_norm = GlobalLayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: [B, N, T]
        B, N, T = x.shape
        
        # Pad to be divisible by chunk_size
        rest = T % self.chunk_size
        if rest > 0:
            pad_len = self.chunk_size - rest
            x = F.pad(x, (0, pad_len))
            T = x.shape[2]
        
        num_chunks = T // self.chunk_size
        
        # Intra-chunk
        x_chunks = x.view(B, N, num_chunks, self.chunk_size)  # [B, N, K, S]
        x_chunks = x_chunks.permute(0, 2, 3, 1).contiguous()  # [B, K, S, N]
        x_chunks = x_chunks.view(B * num_chunks, self.chunk_size, N)  # [B*K, S, N]
        
        out, _ = self.intra_rnn(x_chunks)
        out = self.intra_linear(out)  # [B*K, S, N]
        out = out.view(B, num_chunks, self.chunk_size, N)
        out = out.permute(0, 3, 1, 2).contiguous()  # [B, N, K, S]
        out = out.view(B, N, T)
        
        x = self.intra_norm(x + out)
        
        # Inter-chunk
        x_inter = x.view(B, N, num_chunks, self.chunk_size)  # [B, N, K, S]
        x_inter = x_inter.permute(0, 3, 2, 1).contiguous()  # [B, S, K, N]
        x_inter = x_inter.view(B * self.chunk_size, num_chunks, N)
        
        out, _ = self.inter_rnn(x_inter)
        out = self.inter_linear(out)
        out = out.view(B, self.chunk_size, num_chunks, N)
        out = out.permute(0, 3, 2, 1).contiguous()  # [B, N, K, S]
        out = out.view(B, N, T)
        
        x = self.inter_norm(x + out)
        
        return x


# ==================== mainmodel ====================
class DPRNN_TasNet(nn.Module):
    """TasNetstyleDPRNN"""
    
    def __init__(self, n_src=2, L=20, N=512, H=128, R=6, C=50):
        """
        Args:
            n_src: sourcecountamount
            L: Encodingmodulewindow length
            N: Encodingmoduleoutputdimension
            H: DPRNNhidden layer dimension
            R: DPRNNNumber of chunks
            C: chunksize
        """
        super().__init__()
        
        self.n_src = n_src
        self.N = N
        self.L = L
        
        # Encodingmodule
        self.encoder = Encoder(L, N)
        
        # Layer Norm
        self.ln = GlobalLayerNorm(N)
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(N, H, 1)
        
        # DPRNNmodule
        self.dprnn = nn.Sequential(*[
            DPRNNBlock(H, C) for _ in range(R)
        ])
        
        # Maskgenerator network
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(H, N * n_src, 1),
        )
        
        # Decoder
        self.decoder = Decoder(L, N)
        
        # initializationweight
        self._init_weights()
        
    def _init_weights(self):
        """Xavierinitialization"""
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
        """
        Args:
            mixture: [B, T]
        Returns:
            [B, n_src, T]
        """
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)
        
        orig_len = mixture.shape[-1]
        
        # Encoding
        w = self.encoder(mixture)  # [B, N, K]
        
        # LayerNorm + Bottleneck
        e = self.ln(w)
        e = self.bottleneck(e)  # [B, H, K]
        
        # DPRNN processing
        s = self.dprnn(e)  # [B, H, K]
        
        # generatemasks
        m = self.mask_net(s)  # [B, N*n_src, K_mask]
        B, _, K_mask = m.shape
        
        # ensureK_maskvsEncodingmoduleoutputKdimensionmatching
        _, _, K_enc = w.shape
        if K_mask != K_enc:
            # adjustmaskKdimensiontomatchingw
            if K_mask > K_enc:
                m = m[:, :, :K_enc]
            else:
                # pad toK_enc
                m = F.pad(m, (0, K_enc - K_mask))
        
        m = m.view(B, self.n_src, self.N, K_enc)  # [B, n_src, N, K]
        
        # usingSigmoidactivation（improvement：ratioReLUmore stable）
        m = torch.sigmoid(m)
        
        # applymask
        w = w.unsqueeze(1)  # [B, 1, N, K]
        s = w * m  # [B, n_src, N, K]
        
        # decode eachsource
        separated = []
        for i in range(self.n_src):
            si = self.decoder(s[:, i])  # [B, T']
            separated.append(si)
        
        separated = torch.stack(separated, dim=1)  # [B, n_src, T']
        
        # Crop to original length
        if separated.shape[-1] > orig_len:
            separated = separated[..., :orig_len]
        elif separated.shape[-1] < orig_len:
            separated = F.pad(separated, (0, orig_len - separated.shape[-1]))
        
        return separated


# ==================== Loss Function ====================
def si_snr_loss(est, ref, eps=1e-8):
    """SI-SNRloss（optimized）"""
    # zero mean
    est = est - torch.mean(est, dim=-1, keepdim=True)
    ref = ref - torch.mean(ref, dim=-1, keepdim=True)
    
    # projection
    ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + eps
    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref / ref_energy
    
    # Noise
    noise = est - proj
    
    # SI-SNR
    si_snr = 10 * torch.log10(
        torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps) + eps
    )
    
    return -si_snr


def pit_loss(est_sources, ref_sources):
    """permutationinvariantloss"""
    # est_sources, ref_sources: [B, 2, T]
    loss1 = si_snr_loss(est_sources[:, 0], ref_sources[:, 0]) + \
            si_snr_loss(est_sources[:, 1], ref_sources[:, 1])
    
    loss2 = si_snr_loss(est_sources[:, 0], ref_sources[:, 1]) + \
            si_snr_loss(est_sources[:, 1], ref_sources[:, 0])
    
    loss = torch.min(loss1, loss2)
    return loss.mean()


# ==================== Training ====================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for mixture, sources in pbar:
        mixture = mixture.to(device)
        sources = sources.to(device)
        
        optimizer.zero_grad()
        
        est_sources = model(mixture)
        loss = pit_loss(est_sources, sources)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for mixture, sources in tqdm(dataloader, desc='Evaluating'):
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            est_sources = model(mixture)
            loss = pit_loss(est_sources, sources)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# ==================== mainfunctioncount ====================
def main():
    # optimizationafterconfiguration
    config = {
        'data_root': './MiniLibriMix',
        'batch_size': 4,
        'num_epochs': 150,
        'learning_rate': 1e-3,
        'L': 20,        # window length：20 samples @ 8kHz = 2.5ms
        'N': 512,       # Encodingdimension：512 (improvementcrucial!)
        'H': 128,       # DPRNNhidden dimension：128
        'R': 6,         # DPRNNblockcount：6
        'C': 50,        # chunksize：50
        'warmup_epochs': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    print(f"Model Configuration: L={config['L']}, N={config['N']}, H={config['H']}, R={config['R']}, C={config['C']}")
    
    # data
    train_dataset = MiniLibriMixDataset(config['data_root'], 'train', 'both')
    val_dataset = MiniLibriMixDataset(config['data_root'], 'val', 'both')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                          shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Trainingsample: {len(train_dataset)}, Validationsample: {len(val_dataset)}")
    
    # model
    model = DPRNN_TasNet(
        n_src=2,
        L=config['L'],
        N=config['N'],
        H=config['H'],
        R=config['R'],
        C=config['C']
    ).to(config['device'])
    
    # calculateParameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"modelParameters: {total_params / 1e6:.2f}M")
    
    # optimizationmodule
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # learning ratewarmup scheduler
    warmup_epochs = config['warmup_epochs']
    def get_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
    
    # ReduceLROnPlateau scheduler (fix：removeverboseparameters)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, config['device'])
        val_loss = evaluate(model, val_loader, config['device'])
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # calculatetrueSI-SNR（dB）
        train_sisnr = -train_loss
        val_sisnr = -val_loss
        
        print(f"\nTrainingloss: {train_loss:.4f} | TrainingSI-SNR: {train_sisnr:.4f} dB")
        print(f"Validationloss: {val_loss:.4f} | ValidationSI-SNR: {val_sisnr:.4f} dB")
        
        # learning ratescheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
            print(f"Warmup LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            plateau_scheduler.step(val_loss)
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
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
            }, 'best_dprnn_optimized.pth')
            
            print(f"Save best model | SI-SNR: {val_sisnr:.4f} dB")
        else:
            patience_counter += 1
            print(f"Early stopping: {patience_counter}/{max_patience}")
            
            if patience_counter >= max_patience:
                print("\nearly stoppingTraining！")
                break
    
    # Plotting
    plt.figure(figsize=(14, 5))
    
    # losscurve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Neg SI-SNR, dB)', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # SI-SNRcurve
    plt.subplot(1, 2, 2)
    plt.plot([-l for l in train_losses], label='Train SI-SNR', linewidth=2)
    plt.plot([-l for l in val_losses], label='Val SI-SNR', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('SI-SNR (dB)', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('SI-SNR Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_optimized.png', dpi=300, bbox_inches='tight')
    
    print(f"\n{'='*60}")
    print(f"Trainingcompleted！")
    print(f"bestValidationSI-SNR: {-best_val_loss:.4f} dB")
    print(f"Trainingcurvealreadysave: training_optimized.png")
    print(f"modelalreadysave: best_dprnn_optimized.pth")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
