"""
evaluation phase-awareDPRNN - with automatic phase correction
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from dprnn_polarity_aware import PolarityAwareDPRNN, EnhancedMiniLibriMixDataset


def auto_correct_polarity_batch(est_sources, ref_sources):
    """batch automatic phase correction"""
    corrected = est_sources.clone()
    
    for b in range(est_sources.shape[0]):
        # attempt4permutations
        perms = [
            ([0, 1], [1, 1]),   # est[0]->ref[0], est[1]->ref[1], both positive
            ([0, 1], [-1, 1]),  # est[0]invert->ref[0], est[1]->ref[1]
            ([0, 1], [1, -1]),  # est[0]->ref[0], est[1]invert->ref[1]
            ([0, 1], [-1, -1]), # bothinvert
            ([1, 0], [1, 1]),   # swap + all positive
            ([1, 0], [-1, 1]),
            ([1, 0], [1, -1]),
            ([1, 0], [-1, -1]),
        ]
        
        best_corr = float('-inf')
        best_est = None
        
        for perm_idx, signs in perms:
            est_perm = torch.stack([
                signs[0] * est_sources[b, perm_idx[0]],
                signs[1] * est_sources[b, perm_idx[1]]
            ])
            
            # calculate correlation
            corr = (torch.sum(est_perm[0] * ref_sources[b, 0]) + 
                   torch.sum(est_perm[1] * ref_sources[b, 1]))
            
            if corr > best_corr:
                best_corr = corr
                best_est = est_perm
        
        corrected[b] = best_est
    
    return corrected


def calculate_sisnr(estimated, reference):
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
    if torch.is_tensor(estimated):
        estimated = estimated.cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.cpu().numpy()
    
    signal_energy = np.sum(reference ** 2)
    noise_energy = np.sum((estimated - reference) ** 2)
    sdr = 10 * np.log10(signal_energy / (noise_energy + 1e-8))
    
    return sdr


def evaluate_model(model, dataloader, device, use_polarity_correction=True):
    """Evaluate model - supportphase correction"""
    model.eval()
    
    all_sisnr = []
    all_sdr = []
    all_sisnr_before = []
    all_sdr_before = []
    
    with torch.no_grad():
        for mixture, sources in tqdm(dataloader, desc='Evaluating'):
            mixture = mixture.to(device)
            sources = sources.to(device)
            
            est_sources_raw = model(mixture)
            
            # applyphase correction
            if use_polarity_correction:
                est_sources = auto_correct_polarity_batch(est_sources_raw, sources)
            else:
                est_sources = est_sources_raw
            
            for b in range(mixture.shape[0]):
                ref = sources[b].cpu().numpy()
                est_raw = est_sources_raw[b].cpu().numpy()
                est = est_sources[b].cpu().numpy()
                
                # before correction
                sisnr_before = (calculate_sisnr(est_raw[0], ref[0]) + 
                               calculate_sisnr(est_raw[1], ref[1])) / 2
                sdr_before = (calculate_sdr(est_raw[0], ref[0]) + 
                             calculate_sdr(est_raw[1], ref[1])) / 2
                
                all_sisnr_before.append(sisnr_before)
                all_sdr_before.append(sdr_before)
                
                # after correction
                sisnr_after = (calculate_sisnr(est[0], ref[0]) + 
                              calculate_sisnr(est[1], ref[1])) / 2
                sdr_after = (calculate_sdr(est[0], ref[0]) + 
                            calculate_sdr(est[1], ref[1])) / 2
                
                all_sisnr.append(sisnr_after)
                all_sdr.append(sdr_after)
    
    metrics = {
        'Before Correction': {
            'SI-SNR': {'mean': np.mean(all_sisnr_before), 'std': np.std(all_sisnr_before)},
            'SDR': {'mean': np.mean(all_sdr_before), 'std': np.std(all_sdr_before)},
        },
        'After Correction': {
            'SI-SNR': {'mean': np.mean(all_sisnr), 'std': np.std(all_sisnr)},
            'SDR': {'mean': np.mean(all_sdr), 'std': np.std(all_sdr)},
        },
        'Improvement': {
            'SI-SNR': np.mean(all_sisnr) - np.mean(all_sisnr_before),
            'SDR': np.mean(all_sdr) - np.mean(all_sdr_before),
        }
    }
    
    return metrics, all_sisnr, all_sdr, all_sisnr_before, all_sdr_before


def visualize_results(model, dataset, device, num_samples=5):
    """Visualize separation results - withaudiosave"""
    model.eval()
    
    save_dir = Path('./results_polarity_corrected')
    save_dir.mkdir(exist_ok=True)
    
    # create audio directory
    audio_dir = save_dir / 'audio'
    audio_dir.mkdir(exist_ok=True)
    
    # randomly selectsample
    np.random.seed(42)
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            mixture, sources = dataset[idx]
            mixture_batch = mixture.unsqueeze(0).to(device)
            sources_batch = sources.unsqueeze(0).to(device)
            
            # modeloutput
            est_sources_raw = model(mixture_batch).squeeze(0).cpu()
            
            # phase correction
            est_sources = auto_correct_polarity_batch(
                est_sources_raw.unsqueeze(0), 
                sources.unsqueeze(0)
            ).squeeze(0)
            
            # Calculate metrics
            sisnr_s1 = calculate_sisnr(est_sources[0], sources[0])
            sisnr_s2 = calculate_sisnr(est_sources[1], sources[1])
            sdr_s1 = calculate_sdr(est_sources[0], sources[0])
            sdr_s2 = calculate_sdr(est_sources[1], sources[1])
            sisnr_avg = (sisnr_s1 + sisnr_s2) / 2
            sdr_avg = (sdr_s1 + sdr_s2) / 2
            
            # Plotting - 5row layout（vsconsistent with previous）
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
            
            # Save audiofile
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
    print(f"  - {len(indices)}samplewaveform plot")
    print(f"  - {len(indices) * 5}audiofile")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = './MiniLibriMix'
    model_path = './best_dprnn_polarity_aware.pth'
    
    print("=" * 80)
    print("phase-awareDPRNNevaluation - automaticphase correction")
    print("=" * 80)
    
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        print(f"Tip: Please run dprnn_polarity_aware.py")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    print(f"\nconfiguration: L={config['L']}, N={config['N']}, H={config['H']}, R={config['R']}")
    print(f"   alpha_sdr={config['alpha_sdr']} (SDRlossweight)")
    
    model = PolarityAwareDPRNN(
        n_src=2,
        L=config['L'],
        N=config['N'],
        H=config['H'],
        R=config['R'],
        C=config['C'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully (Epoch {checkpoint['epoch'] + 1})")
    
    # data
    val_dataset = EnhancedMiniLibriMixDataset(data_root, 'val', 'both')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4
    )
    
    print(f"Validationset: {len(val_dataset)} sample")
    
    # evaluation
    print(f"\nevaluationin (with automatic phase correction)...")
    metrics, all_sisnr, all_sdr, all_sisnr_before, all_sdr_before = \
        evaluate_model(model, val_loader, device, use_polarity_correction=True)
    
    # printresults
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    
    print(f"\n{'metrics':<20} {'before correction':<15} {'after correction':<15} {'improvement':<15}")
    print("-" * 65)
    
    sisnr_before = metrics['Before Correction']['SI-SNR']['mean']
    sisnr_after = metrics['After Correction']['SI-SNR']['mean']
    sdr_before = metrics['Before Correction']['SDR']['mean']
    sdr_after = metrics['After Correction']['SDR']['mean']
    
    print(f"{'SI-SNR (dB)':<20} {sisnr_before:<15.2f} {sisnr_after:<15.2f} {metrics['Improvement']['SI-SNR']:+.2f}")
    print(f"{'SDR (dB)':<20} {sdr_before:<15.2f} {sdr_after:<15.2f} {metrics['Improvement']['SDR']:+.2f}")
    
    # comparisonbaseline
    print("\n" + "=" * 80)
    print("vsBaselinecomparison")
    print("=" * 80)
    
    baseline_sisnr = 7.20
    baseline_sdr = -6.01
    
    print(f"\nprevious version (enhanced version):")
    print(f"  SI-SNR: {baseline_sisnr:.2f} dB")
    print(f"  SDR:    {baseline_sdr:.2f} dB")
    
    print(f"\ncurrent version (phase correctionafter):")
    print(f"  SI-SNR: {sisnr_after:.2f} dB ({sisnr_after - baseline_sisnr:+.2f})")
    print(f"  SDR:    {sdr_after:.2f} dB ({sdr_after - baseline_sdr:+.2f})")
    
    if sdr_after > 0:
        print(f"\n🎉 SDRcorrected to positive! phase issue resolved!")
    
    # visualization
    print(f"\nGenerating separation visualizations...")
    visualize_results(model, val_dataset, device, num_samples=5)
    
    # plotperformancedistributionplot（simplified version，onlyshow finalresults）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SI-SNRdistribution
    axes[0].hist(all_sisnr, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(np.mean(all_sisnr), color='red', linestyle='--', 
                   linewidth=2.5, label=f'Mean: {np.mean(all_sisnr):.2f} dB')
    axes[0].set_xlabel('SI-SNR (dB)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title('SI-SNR Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # SDRdistribution
    axes[1].hist(all_sdr, bins=40, alpha=0.7, color='seagreen', edgecolor='black')
    axes[1].axvline(np.mean(all_sdr), color='red', linestyle='--', 
                   linewidth=2.5, label=f'Mean: {np.mean(all_sdr):.2f} dB')
    axes[1].set_xlabel('SDR (dB)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1].set_title('SDR Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_polarity_aware.png', dpi=150, bbox_inches='tight')
    print(f"performancedistributionplot: evaluation_polarity_aware.png")
    
    # savedetailedresultsto textfile
    result_file = 'evaluation_polarity_aware_results.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("phase-awareDPRNN Model Evaluationresults (with automatic phase correction)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model Configuration:\n")
        f.write(f"  L={config['L']}, N={config['N']}, H={config['H']}, R={config['R']}, C={config['C']}\n")
        f.write(f"  alpha_sdr={config['alpha_sdr']} (SDRlossweight)\n")
        f.write(f"  dropout={config['dropout']}\n")
        f.write(f"  TrainingEpoch: {checkpoint['epoch'] + 1}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("before correctionaftercomparison:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'metrics':<20} {'before correction':<15} {'after correction':<15} {'improvement':<15}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'SI-SNR (dB)':<20} {sisnr_before:<15.2f} {sisnr_after:<15.2f} {metrics['Improvement']['SI-SNR']:+.2f}\n")
        f.write(f"{'SDR (dB)':<20} {sdr_before:<15.2f} {sdr_after:<15.2f} {metrics['Improvement']['SDR']:+.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("detailedstatistics:\n")
        f.write("=" * 80 + "\n")
        for stage, data in metrics.items():
            if stage != 'Improvement':
                f.write(f"\n{stage}:\n")
                for metric, values in data.items():
                    f.write(f"  {metric}:\n")
                    for stat, val in values.items():
                        f.write(f"    {stat}: {val:.3f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("performanceevaluation:\n")
        f.write("=" * 80 + "\n")
        if sdr_after > 0:
            f.write(f"SDRcorrected to positive! phase issue resolved!\n")
        else:
            f.write(f"⚠️  SDRstill negative，recommend increasingalpha_sdrparameters\n")
        
        if sisnr_after >= 10:
            f.write(f"🌟🌟🌟 excellent! SI-SNR={sisnr_after:.2f} dB\n")
        elif sisnr_after >= 7:
            f.write(f"🌟🌟 good! SI-SNR={sisnr_after:.2f} dB\n")
        else:
            f.write(f"🌟 acceptable! SI-SNR={sisnr_after:.2f} dB\n")
    
    print(f"detailedresultsSaved to: {result_file}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  evaluation_polarity_aware.png - performancedistributionplot")
    print(f"  evaluation_polarity_aware_results.txt - detailedEvaluation Results")
    print(f"  results_polarity_corrected/ - Visualization results and audio")
    print(f"     ├── separation_sample_*.png - waveform plot ({len(all_sisnr)} samples)")
    print(f"     └── audio/")
    print(f"         ├── sample_*_mixture.wav")
    print(f"         ├── sample_*_source1_true.wav")
    print(f"         ├── sample_*_source1_est.wav")
    print(f"         ├── sample_*_source2_true.wav")
    print(f"         └── sample_*_source2_est.wav")
    
    # performanceevaluation
    if sisnr_after >= 10:
        print(f"\n🌟🌟🌟 excellent! SI-SNRreached {sisnr_after:.2f} dB")
    elif sisnr_after >= 7:
        print(f"\n🌟🌟 good! SI-SNRreached {sisnr_after:.2f} dB")
    elif sisnr_after >= 5:
        print(f"\n🌟 acceptable! SI-SNRreached {sisnr_after:.2f} dB")
    
    if sdr_after > 0:
        print(f"SDRcorrected to positive! {sdr_after:.2f} dB - phase issue resolved!")
    else:
        print(f"⚠️  SDRstill negative {sdr_after:.2f} dB - may need adjustmentalpha_sdrparameters")


if __name__ == '__main__':
    main()
