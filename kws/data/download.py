"""
Download Google Speech Commands v2 dataset and prepare custom emergency keyword data.

The Google Speech Commands dataset contains ~105,000 one-second WAV files
of 35 spoken words. We use a subset as "unknown" class and map target
keywords to our emergency classes.

For custom keywords ("help", "danger", "call 911"), this module provides:
  1. Instructions for recording your own samples
  2. A synthetic data generator using TTS (for bootstrapping)
  3. A noise-based silence generator
"""
import os
import tarfile
import hashlib
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

SPEECH_COMMANDS_URL = (
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
)
SPEECH_COMMANDS_MD5 = "6b74f3901214cb2c2934e98196829835"


def _md5_checksum(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_speech_commands(data_dir: str) -> str:
    """
    Download and extract Google Speech Commands v2 dataset.
    Returns the path to the extracted directory.
    """
    os.makedirs(data_dir, exist_ok=True)
    archive_path = os.path.join(data_dir, "speech_commands_v0.02.tar.gz")

    if not os.path.exists(archive_path):
        print(f"Downloading Speech Commands dataset to {archive_path}...")
        urlretrieve(SPEECH_COMMANDS_URL, archive_path, _progress_hook)
        print("\nDownload complete.")
    else:
        print("Archive already exists, skipping download.")

    # Verify checksum
    if _md5_checksum(archive_path) != SPEECH_COMMANDS_MD5:
        print("WARNING: MD5 checksum mismatch. File may be corrupted.")

    # Extract
    extract_dir = os.path.join(data_dir, "speech_commands")
    if not os.path.exists(extract_dir):
        print(f"Extracting to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

    return extract_dir


def _progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    print(f"\r  {percent}%", end="", flush=True)


def generate_silence_samples(
    output_dir: str, n_samples: int = 1000, sample_rate: int = 16000,
    duration: float = 1.0
):
    """
    Generate silence samples with low-level ambient noise.
    These train the model to distinguish silence from speech.
    """
    silence_dir = os.path.join(output_dir, "silence")
    os.makedirs(silence_dir, exist_ok=True)

    existing = len(list(Path(silence_dir).glob("*.wav")))
    if existing >= n_samples:
        print(f"Silence samples already exist ({existing} files).")
        return silence_dir

    n_samples_audio = int(sample_rate * duration)
    print(f"Generating {n_samples} silence samples...")

    for i in tqdm(range(n_samples), desc="Silence"):
        # Low-amplitude Gaussian noise simulating ambient room noise
        noise_level = np.random.uniform(0.001, 0.01)
        audio = np.random.randn(n_samples_audio).astype(np.float32) * noise_level
        filepath = os.path.join(silence_dir, f"silence_{i:05d}.wav")
        sf.write(filepath, audio, sample_rate)

    return silence_dir


def generate_custom_data(custom_dir: str, sample_rate: int = 16000):
    """
    Create placeholder structure for custom emergency keywords.

    In production, you would:
      1. Record real samples from multiple speakers
      2. Use TTS engines (Google TTS, Coqui TTS) to generate synthetic samples
      3. Apply augmentation to increase dataset size

    This function creates the directory structure and generates
    synthetic placeholder samples using simple signal processing.
    """
    keywords = {
        "help": _generate_placeholder_word,
        "danger": _generate_placeholder_word,
        "call_911": _generate_placeholder_word,
    }

    for keyword, generator in keywords.items():
        keyword_dir = os.path.join(custom_dir, keyword)
        os.makedirs(keyword_dir, exist_ok=True)

        existing = len(list(Path(keyword_dir).glob("*.wav")))
        if existing >= 100:
            print(f"Custom '{keyword}' samples already exist ({existing} files).")
            continue

        print(f"Generating placeholder samples for '{keyword}'...")
        print(f"  NOTE: Replace these with real recordings for production use!")
        print(f"  Record 200+ samples per keyword from diverse speakers.")

        for i in tqdm(range(200), desc=keyword):
            audio = generator(sample_rate)
            filepath = os.path.join(keyword_dir, f"{keyword}_{i:05d}.wav")
            sf.write(filepath, audio, sample_rate)

    return custom_dir


def _generate_placeholder_word(sample_rate: int = 16000) -> np.ndarray:
    """
    Generate a synthetic speech-like audio signal as a placeholder.
    This creates a modulated tone that roughly resembles a spoken word
    in terms of spectral characteristics (NOT actual speech).

    Replace with real TTS or recorded audio for production.
    """
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # Randomize fundamental frequency (simulating different speakers)
    f0 = np.random.uniform(100, 300)

    # Create harmonic stack (speech-like spectral content)
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        amp = 1.0 / harmonic
        signal += amp * np.sin(2 * np.pi * f0 * harmonic * t)

    # Amplitude envelope (attack-sustain-decay like a spoken word)
    word_start = np.random.uniform(0.05, 0.2)
    word_end = np.random.uniform(0.6, 0.9)
    envelope = np.zeros_like(t)
    attack_end = word_start + 0.05
    decay_start = word_end - 0.05

    for i, ti in enumerate(t):
        if ti < word_start:
            envelope[i] = 0.0
        elif ti < attack_end:
            envelope[i] = (ti - word_start) / (attack_end - word_start)
        elif ti < decay_start:
            envelope[i] = 1.0
        elif ti < word_end:
            envelope[i] = (word_end - ti) / (word_end - decay_start)
        else:
            envelope[i] = 0.0

    signal *= envelope

    # Add slight noise
    signal += np.random.randn(len(signal)).astype(np.float32) * 0.01

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.8

    return signal


def prepare_all_data(data_dir: str, custom_dir: str, sample_rate: int = 16000):
    """One-stop function to prepare all data needed for training."""
    print("=" * 60)
    print("STEP 1: Download Google Speech Commands dataset")
    print("=" * 60)
    speech_dir = download_speech_commands(data_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Generate silence samples")
    print("=" * 60)
    generate_silence_samples(custom_dir, n_samples=1000, sample_rate=sample_rate)

    print("\n" + "=" * 60)
    print("STEP 3: Generate custom keyword placeholders")
    print("=" * 60)
    generate_custom_data(custom_dir, sample_rate=sample_rate)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nSpeech Commands: {speech_dir}")
    print(f"Custom data: {custom_dir}")
    print("\nIMPORTANT: For production, replace placeholder custom keyword")
    print("samples with real recordings from diverse speakers.")

    return speech_dir, custom_dir


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import DataConfig

    cfg = DataConfig()
    prepare_all_data(cfg.data_dir, cfg.custom_data_dir)
