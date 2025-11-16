"""
Evaluate performance of optimized DPRNN model
Supports evaluation of best_dprnn_optimized.pth model
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys

# Import model from dprnn_final_optimized.py
sys.path.insert(0, str(Path(__file__).parent))
from dprnn_final_optimized import DPRNN_TasNet, MiniLibriMixDataset


def calculate_sisnr(estimated, reference):
    """Calculate SI-SNR (dB)"""
    if torch.is_tensor(estimated):
        estimated = estimated.cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.cpu().numpy()
    
    estimated = estimated - np.mean(estimated)
    reference = reference - np.mean(reference)
    
    s_target = (np.sum(estimated * reference) / (np.sum(reference ** 2) + 1e-8)) * reference
    e_noise = estimated - s_target
    si_snr = 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8))
    
    return si_snr


def calculate_sdr(estimated, reference):
    """Calculate SDR (dB)"""
    if torch.is_tensor(estimated):
        estimated = estimated.cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.cpu().numpy()
    
    signal_energy = np.sum(reference ** 2)
    noise_energy = np.sum((estimated - reference) ** 2)
    sdr = 10 * np.log10(signal_energy / (noise_energy + 1e-8))
    
    return sdr


def find_best_permutation(est_sources, ref_sources):
    """Find best permutation"""
    sisnr_1 = calculate_sisnr(est_sources[0], ref_sources[0]) + \
              calculate_sisnr(est_sources[1], ref_sources[1])
    
    sisnr_2 = calculate_sisnr(est_sources[0], ref_sources[1]) + \
              calculate_sisnr(est_sources[1], ref_sources[0])
    
    if sisnr_1 > sisnr_2:
        return [0, 1], sisnr_1 / 2
    else:
        return [1, 0], sisnr_2 / 2


def evaluate_model(model, dataloader, device):
    """Complete evaluation"""
    model.eval()
    
    all_sisnr = []
    all_sdr = []
    all_sisnr_mixture = []
    
    with torch.no_grad():
        for mixture, sources in tqdm(dataloader, desc='Evaluating'):
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            est_sources = model(mixture)
            
            for b in range(mixture.shape[0]):
                mix = mixture[b].cpu().numpy()
                ref = sources[b].cpu().numpy()
                est = est_sources[b].cpu().numpy()
                
                # Find best permutation
                perm, _ = find_best_permutation(est, ref)
                est_aligned = est[perm]
                
                # Calculate metrics
                sisnr_s1 = calculate_sisnr(est_aligned[0], ref[0])
                sisnr_s2 = calculate_sisnr(est_aligned[1], ref[1])
                sdr_s1 = calculate_sdr(est_aligned[0], ref[0])
                sdr_s2 = calculate_sdr(est_aligned[1], ref[1])
                
                # Baseline (mixture)
                sisnr_mix_s1 = calculate_sisnr(mix, ref[0])
                sisnr_mix_s2 = calculate_sisnr(mix, ref[1])
                
                all_sisnr.append((sisnr_s1 + sisnr_s2) / 2)
                all_sdr.append((sdr_s1 + sdr_s2) / 2)
                all_sisnr_mixture.append((sisnr_mix_s1 + sisnr_mix_s2) / 2)
    
    metrics = {
        'SI-SNR (dB)': {
            'mean': np.mean(all_sisnr),
            'std': np.std(all_sisnr),
            'median': np.median(all_sisnr),
            'min': np.min(all_sisnr),
            'max': np.max(all_sisnr)
        },
        'SDR (dB)': {
            'mean': np.mean(all_sdr),
            'std': np.std(all_sdr),
            'median': np.median(all_sdr),
            'min': np.min(all_sdr),
            'max': np.max(all_sdr)
        },
        'SI-SNR Improvement (dB)': {
            'mean': np.mean(all_sisnr) - np.mean(all_sisnr_mixture),
        },
        'Mixture SI-SNR (dB)': {
            'mean': np.mean(all_sisnr_mixture)
        }
    }
    
    return metrics, all_sisnr, all_sdr


def visualize_results(model, dataset, device, num_samples=3):
    """Visualize separation results"""
    model.eval()
    
    save_dir = Path('./results_optimized')
    save_dir.mkdir(exist_ok=True)
    
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            mixture, sources = dataset[idx]
            mixture_batch = mixture.unsqueeze(0).to(device)
            est_sources = model(mixture_batch).squeeze(0).cpu()
            
            # Find best permutation
            perm, _ = find_best_permutation(est_sources.numpy(), sources.numpy())
            est_sources = est_sources[perm]
            
            # Calculate SI-SNR
            sisnr_s1 = calculate_sisnr(est_sources[0], sources[0])
            sisnr_s2 = calculate_sisnr(est_sources[1], sources[1])
            
            # Plotting
            fig, axes = plt.subplots(5, 1, figsize=(15, 12))
            time_axis = np.arange(len(mixture)) / 8000
            
            axes[0].plot(time_axis, mixture.numpy(), linewidth=0.5, alpha=0.8, color='black')
            axes[0].set_title('Mixed Signal', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(time_axis, sources[0].numpy(), linewidth=0.5, color='green', alpha=0.8)
            axes[1].set_title('True Source 1', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(time_axis, est_sources[0].numpy(), linewidth=0.5, color='blue', alpha=0.8)
            axes[2].set_title(f'Estimated Source 1 (SI-SNR: {sisnr_s1:.2f} dB)', 
                            fontsize=14, fontweight='bold', color='darkblue')
            axes[2].set_ylabel('Amplitude')
            axes[2].grid(True, alpha=0.3)
            
            axes[3].plot(time_axis, sources[1].numpy(), linewidth=0.5, color='green', alpha=0.8)
            axes[3].set_title('True Source 2', fontsize=14, fontweight='bold')
            axes[3].set_ylabel('Amplitude')
            axes[3].grid(True, alpha=0.3)
            
            axes[4].plot(time_axis, est_sources[1].numpy(), linewidth=0.5, color='blue', alpha=0.8)
            axes[4].set_title(f'Estimated Source 2 (SI-SNR: {sisnr_s2:.2f} dB)', 
                            fontsize=14, fontweight='bold', color='darkblue')
            axes[4].set_ylabel('Amplitude')
            axes[4].set_xlabel('Time (s)')
            axes[4].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'separation_sample_{idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save audio
            audio_dir = save_dir / 'audio'
            audio_dir.mkdir(exist_ok=True)
            
            torchaudio.save(str(audio_dir / f'sample_{idx}_mixture.wav'), 
                          mixture.unsqueeze(0), 8000)
            torchaudio.save(str(audio_dir / f'sample_{idx}_source1_true.wav'), 
                          sources[0].unsqueeze(0), 8000)
            torchaudio.save(str(audio_dir / f'sample_{idx}_source1_est.wav'), 
                          est_sources[0].unsqueeze(0), 8000)
            torchaudio.save(str(audio_dir / f'sample_{idx}_source2_true.wav'), 
                          sources[1].unsqueeze(0), 8000)
            torchaudio.save(str(audio_dir / f'sample_{idx}_source2_est.wav'), 
                          est_sources[1].unsqueeze(0), 8000)
    
    print(f"\nVisualizations saved to: {save_dir}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = './MiniLibriMix'
    model_path = './best_dprnn_optimized.pth'
    
    print("=" * 70)
    print("Optimized DPRNN Model Evaluation")
    print("=" * 70)
    
    # Loading model
    print(f"\nLoading model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"Error: Error: Model file not found {model_path}")
        print(f"Tip: Please run dprnn_final_optimized.py Trainingmodel")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    print(f"\nModel Configuration:")
    print(f"   L={config['L']}, N={config['N']}, H={config['H']}, R={config['R']}, C={config['C']}")
    
    model = DPRNN_TasNet(
        n_src=2,
        L=config['L'],
        N=config['N'],
        H=config['H'],
        R=config['R'],
        C=config['C']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel loaded successfully!")
    print(f"   - Epoch: {checkpoint['epoch'] + 1}")
    print(f"   - Trainingloss: {checkpoint['train_loss']:.4f} dB")
    print(f"   - Validationloss: {checkpoint['val_loss']:.4f} dB")
    print(f"   - Parameters: {total_params / 1e6:.2f}M")
    
    # loadingdata
    print(f"\nloadingValidationset...")
    val_dataset = MiniLibriMixDataset(data_root, 'val', 'both')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    
    print(f"Validationsetsamplecount: {len(val_dataset)}")
    
    # evaluation
    print(f"\nStarting evaluation...")
    metrics, all_sisnr, all_sdr = evaluate_model(model, val_loader, device)
    
    # printresults
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    for metric_name, values in metrics.items():
        print(f"\n{metric_name}:")
        for stat_name, stat_value in values.items():
            print(f"  {stat_name:10s}: {stat_value:8.3f}")
    
    # vsbaselinecomparison
    print("\n" + "=" * 70)
    print("Performance Comparison (vs. Original model)")
    print("=" * 70)
    
    baseline_sisnr = 2.46  # From baseline
    baseline_sdr = -21.76  # From baseline
    
    print(f"\nSI-SNR:")
    print(f"  Original model: {baseline_sisnr:.2f} dB")
    print(f"  Optimized model: {metrics['SI-SNR (dB)']['mean']:.2f} dB")
    improvement_sisnr = metrics['SI-SNR (dB)']['mean'] - baseline_sisnr
    print(f"  Improvement: {improvement_sisnr:+.2f} dB {'(improved)' if improvement_sisnr > 0 else '(decreased)'}")
    
    print(f"\nSDR:")
    print(f"  Original model: {baseline_sdr:.2f} dB")
    print(f"  Optimized model: {metrics['SDR (dB)']['mean']:.2f} dB")
    improvement_sdr = metrics['SDR (dB)']['mean'] - baseline_sdr
    print(f"  Improvement: {improvement_sdr:+.2f} dB {'(improved)' if improvement_sdr > 0 else '(decreased)'}")
    
    print(f"\nSI-SNR Improvement:")
    print(f"  {metrics['SI-SNR Improvement (dB)']['mean']:.2f} dB (compared to mixture)")
    
    # plot distribution
    print(f"\nGenerating metric distribution plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(all_sisnr, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(np.mean(all_sisnr), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(all_sisnr):.2f} dB')
    axes[0].axvline(baseline_sisnr, color='orange', linestyle=':', 
                   linewidth=2, label=f'Baseline: {baseline_sisnr:.2f} dB')
    axes[0].set_xlabel('SI-SNR (dB)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('SI-SNR Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(all_sdr, bins=30, alpha=0.7, color='seagreen', edgecolor='black')
    axes[1].axvline(np.mean(all_sdr), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(all_sdr):.2f} dB')
    axes[1].axvline(baseline_sdr, color='orange', linestyle=':', 
                   linewidth=2, label=f'Baseline: {baseline_sdr:.2f} dB')
    axes[1].set_xlabel('SDR (dB)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('SDR Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics_optimized.png', dpi=150, bbox_inches='tight')
    print(f"Metric distribution plot: evaluation_metrics_optimized.png")
    
    # visualization
    print(f"\nGenerating separation visualizations...")
    visualize_results(model, val_dataset, device, num_samples=5)
    
    # saveEvaluation Resultsto text
    result_file = 'evaluation_results.txt'
    with open(result_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Optimized DPRNN Model Evaluationresults\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Model Configuration: L={config['L']}, N={config['N']}, H={config['H']}, R={config['R']}, C={config['C']}\n")
        f.write(f"Parameters: {total_params / 1e6:.2f}M\n")
        f.write(f"TrainingEpoch: {checkpoint['epoch'] + 1}\n\n")
        
        for metric_name, values in metrics.items():
            f.write(f"{metric_name}:\n")
            for stat_name, stat_value in values.items():
                f.write(f"  {stat_name:10s}: {stat_value:8.3f}\n")
            f.write("\n")
        
        f.write("\nPerformance Comparison:\n")
        f.write(f"  SI-SNR Improvement: {improvement_sisnr:+.2f} dB\n")
        f.write(f"  SDR Improvement: {improvement_sdr:+.2f} dB\n")
    
    print(f"Evaluation ResultsSaved to: {result_file}")
    
    print("\n" + "=" * 70)
    print("Evaluation completed!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  evaluation_metrics_optimized.png - Metric distribution plot")
    print(f"  evaluation_results.txt - detailedEvaluation Results")
    print(f"  results_optimized/ - Visualization results and audio")


if __name__ == '__main__':
    main()
