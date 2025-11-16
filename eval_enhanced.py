"""
evaluationEnhanced DPRNN Model
supportevaluationbest_dprnn_enhanced.pthmodel
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys

# Import model
sys.path.insert(0, str(Path(__file__).parent))
from dprnn_enhanced import EnhancedDPRNN_TasNet, EnhancedMiniLibriMixDataset


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
    all_sisnr_s1 = []
    all_sisnr_s2 = []
    
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
                all_sisnr_s1.append(sisnr_s1)
                all_sisnr_s2.append(sisnr_s2)
    
    metrics = {
        'SI-SNR (dB)': {
            'mean': np.mean(all_sisnr),
            'std': np.std(all_sisnr),
            'median': np.median(all_sisnr),
            'min': np.min(all_sisnr),
            'max': np.max(all_sisnr),
            'percentile_25': np.percentile(all_sisnr, 25),
            'percentile_75': np.percentile(all_sisnr, 75),
        },
        'SDR (dB)': {
            'mean': np.mean(all_sdr),
            'std': np.std(all_sdr),
            'median': np.median(all_sdr),
            'min': np.min(all_sdr),
            'max': np.max(all_sdr),
            'percentile_25': np.percentile(all_sdr, 25),
            'percentile_75': np.percentile(all_sdr, 75),
        },
        'SI-SNR Source 1 (dB)': {
            'mean': np.mean(all_sisnr_s1),
        },
        'SI-SNR Source 2 (dB)': {
            'mean': np.mean(all_sisnr_s2),
        },
        'SI-SNR Improvement (dB)': {
            'mean': np.mean(all_sisnr) - np.mean(all_sisnr_mixture),
        },
        'Mixture SI-SNR (dB)': {
            'mean': np.mean(all_sisnr_mixture)
        }
    }
    
    return metrics, all_sisnr, all_sdr


def visualize_results(model, dataset, device, num_samples=5):
    """Visualize separation results"""
    model.eval()
    
    save_dir = Path('./results_enhanced')
    save_dir.mkdir(exist_ok=True)
    
    # randomly selectsample
    np.random.seed(42)
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
            sisnr_avg = (sisnr_s1 + sisnr_s2) / 2
            
            # calculateSDR
            sdr_s1 = calculate_sdr(est_sources[0], sources[0])
            sdr_s2 = calculate_sdr(est_sources[1], sources[1])
            sdr_avg = (sdr_s1 + sdr_s2) / 2
            
            # Plotting
            fig, axes = plt.subplots(5, 1, figsize=(16, 13))
            time_axis = np.arange(len(mixture)) / 8000
            
            # Mixed Signal
            axes[0].plot(time_axis, mixture.numpy(), linewidth=0.5, alpha=0.8, color='black')
            axes[0].set_title('Mixed Signal', fontsize=15, fontweight='bold')
            axes[0].set_ylabel('Amplitude', fontsize=11)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim([0, time_axis[-1]])
            
            # True Source 1
            axes[1].plot(time_axis, sources[0].numpy(), linewidth=0.5, color='green', alpha=0.8)
            axes[1].set_title('True Source 1', fontsize=15, fontweight='bold', color='darkgreen')
            axes[1].set_ylabel('Amplitude', fontsize=11)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim([0, time_axis[-1]])
            
            # Estimated Source 1
            axes[2].plot(time_axis, est_sources[0].numpy(), linewidth=0.5, color='blue', alpha=0.8)
            axes[2].set_title(f'Estimated Source 1 | SI-SNR: {sisnr_s1:.2f} dB | SDR: {sdr_s1:.2f} dB', 
                            fontsize=15, fontweight='bold', color='darkblue')
            axes[2].set_ylabel('Amplitude', fontsize=11)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_xlim([0, time_axis[-1]])
            
            # True Source 2
            axes[3].plot(time_axis, sources[1].numpy(), linewidth=0.5, color='green', alpha=0.8)
            axes[3].set_title('True Source 2', fontsize=15, fontweight='bold', color='darkgreen')
            axes[3].set_ylabel('Amplitude', fontsize=11)
            axes[3].grid(True, alpha=0.3)
            axes[3].set_xlim([0, time_axis[-1]])
            
            # Estimated Source 2
            axes[4].plot(time_axis, est_sources[1].numpy(), linewidth=0.5, color='blue', alpha=0.8)
            axes[4].set_title(f'Estimated Source 2 | SI-SNR: {sisnr_s2:.2f} dB | SDR: {sdr_s2:.2f} dB', 
                            fontsize=15, fontweight='bold', color='darkblue')
            axes[4].set_ylabel('Amplitude', fontsize=11)
            axes[4].set_xlabel('Time (s)', fontsize=12)
            axes[4].grid(True, alpha=0.3)
            axes[4].set_xlim([0, time_axis[-1]])
            
            # add overall performance annotation
            fig.suptitle(f'Sample {idx} | Average SI-SNR: {sisnr_avg:.2f} dB | Average SDR: {sdr_avg:.2f} dB',
                        fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout(rect=[0, 0, 1, 0.99])
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
    print(f"  - {num_samples}samplewaveform plot")
    print(f"  - {num_samples * 5}audiofile")


def plot_comprehensive_analysis(metrics, all_sisnr, all_sdr, save_path='evaluation_enhanced.png'):
    """plotcomprehensive analysis plot"""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    baseline_sisnr = 4.63
    baseline_sdr = 2.36
    
    # 1. SI-SNRdistribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts, bins, patches = ax1.hist(all_sisnr, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(np.mean(all_sisnr), color='red', linestyle='--', 
               linewidth=2.5, label=f'Mean: {np.mean(all_sisnr):.2f} dB')
    ax1.axvline(baseline_sisnr, color='orange', linestyle=':', 
               linewidth=2.5, label=f'Previous: {baseline_sisnr:.2f} dB')
    ax1.set_xlabel('SI-SNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('SI-SNR Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. SDRdistribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(all_sdr, bins=40, alpha=0.7, color='seagreen', edgecolor='black')
    ax2.axvline(np.mean(all_sdr), color='red', linestyle='--', 
               linewidth=2.5, label=f'Mean: {np.mean(all_sdr):.2f} dB')
    ax2.axvline(baseline_sdr, color='orange', linestyle=':', 
               linewidth=2.5, label=f'Previous: {baseline_sdr:.2f} dB')
    ax2.set_xlabel('SDR (dB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('SDR Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance Comparisonbar chart
    ax3 = fig.add_subplot(gs[0, 2])
    metrics_names = ['SI-SNR', 'SDR']
    previous_values = [baseline_sisnr, baseline_sdr]
    current_values = [np.mean(all_sisnr), np.mean(all_sdr)]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, previous_values, width, label='Previous Model', 
                   color='orange', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x + width/2, current_values, width, label='Enhanced Model', 
                   color='steelblue', alpha=0.8, edgecolor='black')
    
    ax3.set_ylabel('Performance (dB)', fontsize=12, fontweight='bold')
    ax3.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. SI-SNRbox plot
    ax4 = fig.add_subplot(gs[1, 0])
    bp1 = ax4.boxplot([all_sisnr], positions=[1], widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='steelblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    ax4.set_ylabel('SI-SNR (dB)', fontsize=12, fontweight='bold')
    ax4.set_title('SI-SNR Box Plot', fontsize=14, fontweight='bold')
    ax4.set_xticks([1])
    ax4.set_xticklabels(['Enhanced Model'], fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(baseline_sisnr, color='orange', linestyle=':', 
               linewidth=2, label=f'Previous: {baseline_sisnr:.2f} dB')
    ax4.legend(fontsize=10)
    
    # 5. statistical information text
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    stats_text = f"""
    Enhanced Model Statistics
    
    SI-SNR:
      Mean:        {metrics['SI-SNR (dB)']['mean']:.3f} dB
      Std:         {metrics['SI-SNR (dB)']['std']:.3f} dB
      Median:      {metrics['SI-SNR (dB)']['median']:.3f} dB
      Min:         {metrics['SI-SNR (dB)']['min']:.3f} dB
      Max:         {metrics['SI-SNR (dB)']['max']:.3f} dB
      25th %ile:   {metrics['SI-SNR (dB)']['percentile_25']:.3f} dB
      75th %ile:   {metrics['SI-SNR (dB)']['percentile_75']:.3f} dB
    
    SDR:
      Mean:        {metrics['SDR (dB)']['mean']:.3f} dB
      Std:         {metrics['SDR (dB)']['std']:.3f} dB
      Median:      {metrics['SDR (dB)']['median']:.3f} dB
    
    Improvement:
      SI-SNR Δ:    {metrics['SI-SNR (dB)']['mean'] - baseline_sisnr:+.3f} dB
      SDR Δ:       {metrics['SDR (dB)']['mean'] - baseline_sdr:+.3f} dB
      vs Mixture:  {metrics['SI-SNR Improvement (dB)']['mean']:+.3f} dB
    """
    
    ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 6. improvementmagnitudevisualization
    ax6 = fig.add_subplot(gs[1, 2])
    improvements = {
        'SI-SNR': metrics['SI-SNR (dB)']['mean'] - baseline_sisnr,
        'SDR': metrics['SDR (dB)']['mean'] - baseline_sdr,
    }
    
    colors = ['green' if v > 0 else 'red' for v in improvements.values()]
    bars = ax6.barh(list(improvements.keys()), list(improvements.values()), 
                    color=colors, alpha=0.7, edgecolor='black')
    ax6.axvline(0, color='black', linewidth=1)
    ax6.set_xlabel('Improvement (dB)', fontsize=12, fontweight='bold')
    ax6.set_title('Performance Improvement', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # add value labels
    for i, (metric, value) in enumerate(improvements.items()):
        ax6.text(value, i, f'  {value:+.2f} dB', 
                va='center', fontsize=11, fontweight='bold')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"comprehensive analysis plot: {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = './MiniLibriMix'
    model_path = './best_dprnn_enhanced.pth'
    
    print("=" * 80)
    print("Enhanced DPRNN Modelevaluation")
    print("=" * 80)
    
    # Loading model
    print(f"\nLoading model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"Error: Error: Model file not found {model_path}")
        print(f"Tip: Please run dprnn_enhanced.py Trainingmodel")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    print(f"\nModel Configuration:")
    print(f"   L={config['L']}, N={config['N']}, H={config['H']}, R={config['R']}, C={config['C']}")
    print(f"   Dropout={config['dropout']}, Gradient Accumulation={config['gradient_accumulation_steps']}")
    
    model = EnhancedDPRNN_TasNet(
        n_src=2,
        L=config['L'],
        N=config['N'],
        H=config['H'],
        R=config['R'],
        C=config['C'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel loaded successfully!")
    print(f"   - Epoch: {checkpoint['epoch'] + 1}")
    print(f"   - Trainingloss: {checkpoint['train_loss']:.4f} dB")
    print(f"   - Validationloss: {checkpoint['val_loss']:.4f} dB")
    print(f"   - TrainingSI-SNR: {-checkpoint['train_loss']:.4f} dB")
    print(f"   - ValidationSI-SNR: {-checkpoint['val_loss']:.4f} dB")
    print(f"   - Parameters: {total_params / 1e6:.2f}M")
    
    # loadingdata
    print(f"\nloadingValidationset...")
    val_dataset = EnhancedMiniLibriMixDataset(data_root, 'val', 'both')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    
    print(f"Validationsetsamplecount: {len(val_dataset)}")
    
    # evaluation
    print(f"\nStarting evaluation...")
    metrics, all_sisnr, all_sdr = evaluate_model(model, val_loader, device)
    
    # printresults
    print("\n" + "=" * 80)
    print("detailedEvaluation Results")
    print("=" * 80)
    
    for metric_name, values in metrics.items():
        print(f"\n{metric_name}:")
        for stat_name, stat_value in values.items():
            print(f"  {stat_name:15s}: {stat_value:8.3f}")
    
    # vsbaselinecomparison
    print("\n" + "=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    
    baseline_sisnr = 4.63
    baseline_sdr = 2.36
    
    print(f"\n{'metrics':<15} {'Original model':<15} {'enhancedmodel':<15} {'improvement':<15}")
    print("-" * 60)
    
    sisnr_current = metrics['SI-SNR (dB)']['mean']
    sdr_current = metrics['SDR (dB)']['mean']
    
    sisnr_improvement = sisnr_current - baseline_sisnr
    sdr_improvement = sdr_current - baseline_sdr
    
    print(f"{'SI-SNR (dB)':<15} {baseline_sisnr:<15.2f} {sisnr_current:<15.2f} {sisnr_improvement:+.2f} dB")
    print(f"{'SDR (dB)':<15} {baseline_sdr:<15.2f} {sdr_current:<15.2f} {sdr_improvement:+.2f} dB")
    
    print(f"\n{'SI-SNR Improvement (vs mixture):':<40} {metrics['SI-SNR Improvement (dB)']['mean']:+.2f} dB")
    
    # performanceevaluation
    print("\n" + "=" * 80)
    print("performanceevaluation")
    print("=" * 80)
    
    if sisnr_current >= 12:
        grade = "🌟🌟🌟 excellent!"
    elif sisnr_current >= 10:
        grade = "🌟🌟 good!"
    elif sisnr_current >= 7:
        grade = "🌟 acceptable"
    else:
        grade = "⚠️  needimprovement"
    
    print(f"\nmodeletcgrade: {grade}")
    print(f"SI-SNR: {sisnr_current:.2f} dB")
    
    if sisnr_improvement > 0:
        print(f"relativelyratioprevious versionimprovement {sisnr_improvement:.2f} dB")
    else:
        print(f"⚠️  relativelyratioprevious versiondecreased {abs(sisnr_improvement):.2f} dB")
    
    # plotcomprehensive analysis plot
    print(f"\ngenerate comprehensive analysis plot...")
    plot_comprehensive_analysis(metrics, all_sisnr, all_sdr, 'evaluation_enhanced.png')
    
    # Visualize separation results
    print(f"\nGenerating separation visualizations...")
    visualize_results(model, val_dataset, device, num_samples=5)
    
    # saveEvaluation Results
    result_file = 'evaluation_enhanced_results.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Enhanced DPRNN ModelEvaluation Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model Configuration: L={config['L']}, N={config['N']}, H={config['H']}, R={config['R']}, C={config['C']}\n")
        f.write(f"Parameters: {total_params / 1e6:.2f}M\n")
        f.write(f"TrainingEpoch: {checkpoint['epoch'] + 1}\n")
        f.write(f"Dropout: {config['dropout']}\n\n")
        
        f.write("detailedmetrics:\n")
        f.write("-" * 80 + "\n")
        for metric_name, values in metrics.items():
            f.write(f"\n{metric_name}:\n")
            for stat_name, stat_value in values.items():
                f.write(f"  {stat_name:15s}: {stat_value:8.3f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Performance Comparison:\n")
        f.write("=" * 80 + "\n")
        f.write(f"SI-SNR: {baseline_sisnr:.2f} → {sisnr_current:.2f} dB (Δ {sisnr_improvement:+.2f} dB)\n")
        f.write(f"SDR:    {baseline_sdr:.2f} → {sdr_current:.2f} dB (Δ {sdr_improvement:+.2f} dB)\n")
        f.write(f"\nmodeletcgrade: {grade}\n")
    
    print(f"Evaluation ResultsSaved to: {result_file}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  evaluation_enhanced.png - comprehensive analysis plot")
    print(f"  evaluation_enhanced_results.txt - detailedEvaluation Results")
    print(f"  results_enhanced/ - Visualization results and audio")
    print(f"     ├── separation_sample_*.png - waveform plot")
    print(f"     └── audio/ - audiofile")
    
    # ifperformancesignificantlyimprovement，give congratulations
    if sisnr_improvement >= 3:
        print(f"\n🎉 congratulations! modelperformanceimprovementsignificantly (+{sisnr_improvement:.2f} dB)")
    elif sisnr_improvement >= 1:
        print(f"\n👍 good! modelperformancesomewhatimprovement (+{sisnr_improvement:.2f} dB)")


if __name__ == '__main__':
    main()
