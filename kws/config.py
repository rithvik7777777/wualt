"""
Central configuration for the KWS system.
All hyperparameters, paths, and constants live here.
"""
import os
from dataclasses import dataclass, field
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration_sec: float = 1.0
    n_samples: int = 16000  # sample_rate * duration_sec
    # Feature extraction
    n_mfcc: int = 40
    n_fft: int = 512
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 480  # 30ms at 16kHz
    n_mels: int = 40
    fmin: int = 20
    fmax: int = 4000
    # Resulting spectrogram shape: (n_mfcc, time_steps)
    # time_steps = ceil(n_samples / hop_length) = 101


@dataclass
class DataConfig:
    data_dir: str = os.path.join(BASE_DIR, "datasets", "speech_commands")
    custom_data_dir: str = os.path.join(BASE_DIR, "datasets", "custom")
    # Target keywords for emergency detection
    target_keywords: List[str] = field(
        default_factory=lambda: ["help", "danger", "call_911"]
    )
    # Background/known words from Google Speech Commands used as "unknown"
    background_words: List[str] = field(
        default_factory=lambda: [
            "yes", "no", "up", "down", "left", "right",
            "on", "off", "stop", "go", "zero", "one",
            "two", "three", "four", "five", "six", "seven",
            "eight", "nine",
        ]
    )
    # Max samples per background class to keep dataset balanced
    max_unknown_per_class: int = 200
    silence_samples: int = 1000
    val_split: float = 0.1
    test_split: float = 0.1
    # Augmentation
    noise_snr_db_range: tuple = (5, 20)
    time_shift_ms: int = 100
    speed_range: tuple = (0.9, 1.1)


@dataclass
class ModelConfig:
    # DS-CNN architecture parameters
    n_classes: int = 3  # emergency, unknown, silence
    input_channels: int = 1
    # First conv layer
    first_conv_filters: int = 64
    first_conv_kernel: tuple = (4, 10)
    first_conv_stride: tuple = (2, 2)
    # Depthwise separable conv blocks
    ds_conv_filters: List[int] = field(
        default_factory=lambda: [64, 64, 64, 64]
    )
    ds_conv_kernels: List[tuple] = field(
        default_factory=lambda: [(3, 3), (3, 3), (3, 3), (3, 3)]
    )
    dropout: float = 0.2


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_scheduler_step: int = 20
    lr_scheduler_gamma: float = 0.1
    checkpoint_dir: str = os.path.join(BASE_DIR, "checkpoints")
    log_dir: str = os.path.join(BASE_DIR, "logs")
    seed: int = 42
    num_workers: int = 4


@dataclass
class InferenceConfig:
    # Sliding window for streaming
    window_duration_sec: float = 1.0
    hop_duration_sec: float = 0.5  # 50% overlap
    # Detection thresholds
    confidence_threshold: float = 0.85
    # Smoothing: require N consecutive detections
    smoothing_window: int = 3
    # Cooldown after alert to avoid repeated triggers (seconds)
    cooldown_sec: float = 3.0
    # Model path
    model_path: str = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")
    tflite_model_path: str = os.path.join(BASE_DIR, "checkpoints", "kws_model.tflite")
    quantized_model_path: str = os.path.join(
        BASE_DIR, "checkpoints", "kws_model_int8.tflite"
    )


# Label mapping (3-class: emergency combines help/danger/call_911)
LABELS = ["emergency", "unknown", "silence"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}
