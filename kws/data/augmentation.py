"""
Audio augmentation pipeline for KWS training.

Augmentation is critical for edge-deployed models because real-world audio
contains noise, reverberation, and speaker variability. These transforms
make the model robust to deployment conditions.
"""
import numpy as np
import random


class AudioAugmentor:
    """
    Applies a chain of audio augmentations during training.
    Each augmentation is applied with a configurable probability.
    """

    def __init__(
        self,
        noise_snr_range=(5, 20),
        time_shift_ms=100,
        speed_range=(0.9, 1.1),
        sample_rate=16000,
        p_noise=0.5,
        p_shift=0.5,
        p_speed=0.3,
        p_volume=0.3,
        p_spec_augment=0.3,
    ):
        self.noise_snr_range = noise_snr_range
        self.time_shift_samples = int(time_shift_ms * sample_rate / 1000)
        self.speed_range = speed_range
        self.sample_rate = sample_rate
        self.p_noise = p_noise
        self.p_shift = p_shift
        self.p_speed = p_speed
        self.p_volume = p_volume
        self.p_spec_augment = p_spec_augment

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a 1D audio array."""
        if random.random() < self.p_noise:
            audio = self.add_noise(audio)
        if random.random() < self.p_shift:
            audio = self.time_shift(audio)
        if random.random() < self.p_speed:
            audio = self.speed_perturb(audio)
        if random.random() < self.p_volume:
            audio = self.volume_perturb(audio)
        return audio

    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise at a random SNR within the configured range.
        SNR (dB) = 10 * log10(signal_power / noise_power)
        """
        signal_power = np.mean(audio ** 2)
        if signal_power < 1e-10:
            return audio

        snr_db = np.random.uniform(*self.noise_snr_range)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
        return audio + noise

    def time_shift(self, audio: np.ndarray) -> np.ndarray:
        """
        Shift audio left or right by a random amount.
        Simulates words being spoken at slightly different times in the window.
        """
        shift = np.random.randint(-self.time_shift_samples, self.time_shift_samples)
        if shift > 0:
            audio = np.pad(audio, (shift, 0), mode="constant")[:len(audio)]
        elif shift < 0:
            audio = np.pad(audio, (0, -shift), mode="constant")[-shift : -shift + len(audio)]
        return audio

    def speed_perturb(self, audio: np.ndarray) -> np.ndarray:
        """
        Change playback speed by resampling.
        Speed < 1.0 = slower (lower pitch), speed > 1.0 = faster (higher pitch).
        """
        speed = np.random.uniform(*self.speed_range)
        indices = np.round(np.arange(0, len(audio), speed)).astype(int)
        indices = indices[indices < len(audio)]
        stretched = audio[indices]

        # Pad or trim to original length
        target_len = len(audio)
        if len(stretched) > target_len:
            stretched = stretched[:target_len]
        elif len(stretched) < target_len:
            stretched = np.pad(
                stretched, (0, target_len - len(stretched)), mode="constant"
            )
        return stretched

    def volume_perturb(self, audio: np.ndarray) -> np.ndarray:
        """Random volume scaling between 0.7x and 1.3x."""
        gain = np.random.uniform(0.7, 1.3)
        return audio * gain


class SpecAugment:
    """
    SpecAugment: frequency and time masking on spectrograms.
    Applied after feature extraction during training.

    Reference: Park et al., "SpecAugment" (2019)
    """

    def __init__(self, freq_mask_param=5, time_mask_param=10, n_freq_masks=1, n_time_masks=1):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply frequency and time masking to a spectrogram.
        Input shape: (n_mels/n_mfcc, time_steps)
        """
        spec = spectrogram.copy()
        n_freq, n_time = spec.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, n_freq))
            f0 = np.random.randint(0, max(1, n_freq - f))
            spec[f0 : f0 + f, :] = 0.0

        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, min(self.time_mask_param, n_time))
            t0 = np.random.randint(0, max(1, n_time - t))
            spec[:, t0 : t0 + t] = 0.0

        return spec
