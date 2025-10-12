# dataset.py (Corrected and Cleaned Version)

import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class MiniLibriMixDataset(Dataset):
    """
    用于加载MiniLibriMix数据集的自定义PyTorch Dataset类。
    """
    def __init__(self, dataset_path, split='train', mix_type='mix_clean', sample_rate=16000):
        super().__init__()
        
        if split not in ['train', 'val']:
            raise ValueError("split参数必须是 'train' 或 'val'")
        if mix_type not in ['mix_clean', 'mix_both']:
            raise ValueError("mix_type参数必须是 'mix_clean' 或 'mix_both'")
            
        self.sample_rate = sample_rate

        self.mix_path = os.path.join(dataset_path, split, mix_type)
        self.s1_path = os.path.join(dataset_path, split, 's1')
        self.s2_path = os.path.join(dataset_path, split, 's2')

        self.filenames = [f for f in os.listdir(self.mix_path) if f.endswith('.wav')]
        
        print(f"成功找到 {len(self.filenames)} 个位于 '{split}/{mix_type}' 的样本。")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        mix_filepath = os.path.join(self.mix_path, filename)
        s1_filepath = os.path.join(self.s1_path, filename)
        s2_filepath = os.path.join(self.s2_path, filename)

        mix_audio, _ = librosa.load(mix_filepath, sr=self.sample_rate, mono=True)
        s1_audio, _ = librosa.load(s1_filepath, sr=self.sample_rate, mono=True)
        s2_audio, _ = librosa.load(s2_filepath, sr=self.sample_rate, mono=True)

        sources = np.stack([s1_audio, s2_audio], axis=0)
        
        return torch.from_numpy(mix_audio).float(), torch.from_numpy(sources).float()