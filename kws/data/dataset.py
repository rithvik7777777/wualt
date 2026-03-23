"""
PyTorch Dataset and DataLoader for KWS training.

Handles loading audio from Google Speech Commands and custom keyword directories,
extracting MFCC features, applying augmentation, and batching.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import librosa
import soundfile as sf
from typing import Optional, List, Tuple, Dict
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    AudioConfig, DataConfig, LABEL_TO_IDX, LABELS
)
from data.augmentation import AudioAugmentor, SpecAugment


def extract_mfcc(audio: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """
    Extract MFCC features from raw audio.
    Returns shape: (n_mfcc, time_steps)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    return mfcc.astype(np.float32)


def load_and_preprocess_audio(filepath: str, cfg: AudioConfig) -> np.ndarray:
    """
    Load an audio file and preprocess to a fixed-length, normalized array.
    """
    audio, sr = sf.read(filepath, dtype="float32")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != cfg.sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.sample_rate)

    # Pad or trim to exactly n_samples
    if len(audio) < cfg.n_samples:
        audio = np.pad(audio, (0, cfg.n_samples - len(audio)), mode="constant")
    elif len(audio) > cfg.n_samples:
        audio = audio[: cfg.n_samples]

    # Normalize to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio


class KWSDataset(Dataset):
    """
    Dataset for keyword spotting.

    Loads audio files, applies augmentation (training only),
    extracts MFCC features, and returns (features, label) pairs.
    """

    def __init__(
        self,
        file_list: List[Tuple[str, int]],
        audio_cfg: AudioConfig,
        augmentor: Optional[AudioAugmentor] = None,
        spec_augment: Optional[SpecAugment] = None,
    ):
        self.file_list = file_list  # List of (filepath, label_idx)
        self.audio_cfg = audio_cfg
        self.augmentor = augmentor
        self.spec_augment = spec_augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, label = self.file_list[idx]

        # Load audio
        audio = load_and_preprocess_audio(filepath, self.audio_cfg)

        # Apply waveform augmentation (training only)
        if self.augmentor is not None:
            audio = self.augmentor(audio)

        # Extract MFCC features
        mfcc = extract_mfcc(audio, self.audio_cfg)

        # Apply SpecAugment (training only)
        if self.spec_augment is not None:
            mfcc = self.spec_augment(mfcc)

        # Add channel dimension: (1, n_mfcc, time_steps)
        features = torch.from_numpy(mfcc).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return features, label


def build_file_list(data_cfg: DataConfig) -> List[Tuple[str, int]]:
    """
    Scan data directories and build a list of (filepath, label_idx) tuples.
    """
    file_list = []

    # 1. Custom keywords (help, danger, call_911)
    for keyword in data_cfg.target_keywords:
        keyword_dir = os.path.join(data_cfg.custom_data_dir, keyword)
        if not os.path.isdir(keyword_dir):
            print(f"WARNING: Custom keyword directory not found: {keyword_dir}")
            continue
        label_idx = LABEL_TO_IDX[keyword]
        wav_files = list(Path(keyword_dir).glob("*.wav"))
        for f in wav_files:
            file_list.append((str(f), label_idx))
        print(f"  {keyword}: {len(wav_files)} samples")

    # 2. Silence
    silence_dir = os.path.join(data_cfg.custom_data_dir, "silence")
    if os.path.isdir(silence_dir):
        silence_files = list(Path(silence_dir).glob("*.wav"))
        label_idx = LABEL_TO_IDX["silence"]
        for f in silence_files:
            file_list.append((str(f), label_idx))
        print(f"  silence: {len(silence_files)} samples")

    # 3. Unknown words from Speech Commands (subsampled for balance)
    speech_dir = os.path.join(data_cfg.data_dir, "speech_commands")
    unknown_label = LABEL_TO_IDX["unknown"]
    unknown_count = 0

    for word in data_cfg.background_words:
        word_dir = os.path.join(speech_dir, word)
        if not os.path.isdir(word_dir):
            continue
        wav_files = list(Path(word_dir).glob("*.wav"))
        # Subsample to keep balanced
        if len(wav_files) > data_cfg.max_unknown_per_class:
            wav_files = list(
                np.random.choice(
                    wav_files, data_cfg.max_unknown_per_class, replace=False
                )
            )
        for f in wav_files:
            file_list.append((str(f), unknown_label))
        unknown_count += len(wav_files)

    print(f"  unknown: {unknown_count} samples (from {len(data_cfg.background_words)} words)")

    # Also check if "help" exists in Speech Commands (it doesn't in v2, but check)
    # If a target keyword exists in Speech Commands too, include those samples
    for keyword in data_cfg.target_keywords:
        sc_keyword_dir = os.path.join(speech_dir, keyword)
        if os.path.isdir(sc_keyword_dir):
            label_idx = LABEL_TO_IDX[keyword]
            wav_files = list(Path(sc_keyword_dir).glob("*.wav"))
            for f in wav_files:
                file_list.append((str(f), label_idx))
            print(f"  {keyword} (from Speech Commands): {len(wav_files)} extra samples")

    print(f"\nTotal: {len(file_list)} samples across {len(LABELS)} classes")
    return file_list


def create_dataloaders(
    data_cfg: DataConfig,
    audio_cfg: AudioConfig,
    train_cfg=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders with augmentation and class balancing.
    """
    print("Building file list...")
    all_files = build_file_list(data_cfg)

    if len(all_files) == 0:
        raise RuntimeError(
            "No audio files found. Run data preparation first:\n"
            "  python -m data.download"
        )

    # Split into train/val/test
    filepaths = [f[0] for f in all_files]
    labels = [f[1] for f in all_files]

    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, labels,
        test_size=data_cfg.test_split,
        stratify=labels,
        random_state=42,
    )

    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels,
        test_size=data_cfg.val_split / (1 - data_cfg.test_split),
        stratify=train_labels,
        random_state=42,
    )

    print(f"\nSplit: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # Compute class weights for balanced sampling
    train_label_array = np.array([f[1] for f in train_files])
    class_counts = np.bincount(train_label_array, minlength=len(LABELS))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[train_label_array]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Augmentation for training only
    augmentor = AudioAugmentor(
        noise_snr_range=data_cfg.noise_snr_db_range,
        time_shift_ms=data_cfg.time_shift_ms,
        speed_range=data_cfg.speed_range,
        sample_rate=audio_cfg.sample_rate,
    )
    spec_aug = SpecAugment(freq_mask_param=5, time_mask_param=10)

    batch_size = train_cfg.batch_size if train_cfg else 64
    num_workers = train_cfg.num_workers if train_cfg else 4

    train_dataset = KWSDataset(train_files, audio_cfg, augmentor, spec_aug)
    val_dataset = KWSDataset(val_files, audio_cfg)
    test_dataset = KWSDataset(test_files, audio_cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
