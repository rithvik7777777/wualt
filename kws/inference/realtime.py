"""
Real-time keyword spotting from microphone input.

This module implements streaming inference with:
  - Continuous microphone capture in a background thread
  - Sliding window MFCC extraction
  - DS-CNN inference on each window
  - Smoothing logic to reduce false positives
  - Cooldown period to prevent repeated alerts

Architecture:
  Mic -> Ring Buffer -> [1s window] -> MFCC -> DS-CNN -> Smoothing -> Alert
           ^                                                    |
           |__________________hop (0.5s)________________________|
"""
import os
import sys
import time
import threading
import collections
import numpy as np
import torch
import pyaudio
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    AudioConfig, InferenceConfig, ModelConfig,
    IDX_TO_LABEL, LABELS, LABEL_TO_IDX,
)
from model.ds_cnn import DSCNN
from data.dataset import extract_mfcc


class RealtimeKWS:
    """
    Real-time keyword spotting engine.

    Usage:
        kws = RealtimeKWS(model_path="checkpoints/best_model.pt")
        kws.start()
        # ... listens continuously, prints alerts ...
        kws.stop()
    """

    def __init__(
        self,
        model_path: str = None,
        audio_cfg: AudioConfig = None,
        infer_cfg: InferenceConfig = None,
        model_cfg: ModelConfig = None,
        alert_callback=None,
        use_onnx: bool = False,
        onnx_path: str = None,
    ):
        self.audio_cfg = audio_cfg or AudioConfig()
        self.infer_cfg = infer_cfg or InferenceConfig()
        self.model_cfg = model_cfg or ModelConfig()

        model_path = model_path or self.infer_cfg.model_path

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load model
        if use_onnx and onnx_path:
            self._load_onnx(onnx_path)
        else:
            self._load_pytorch(model_path)

        # Audio streaming parameters
        self.sample_rate = self.audio_cfg.sample_rate
        self.window_samples = int(self.infer_cfg.window_duration_sec * self.sample_rate)
        self.hop_samples = int(self.infer_cfg.hop_duration_sec * self.sample_rate)
        self.chunk_size = 1024  # PyAudio read chunk

        # Ring buffer to accumulate audio
        self.audio_buffer = collections.deque(maxlen=self.window_samples)

        # Smoothing: track recent predictions
        self.prediction_history = collections.deque(
            maxlen=self.infer_cfg.smoothing_window
        )

        # Cooldown tracking per keyword
        self.last_alert_time = {}
        for keyword in LABELS:
            self.last_alert_time[keyword] = 0.0

        # Alert callback
        self.alert_callback = alert_callback or self._default_alert

        # Threading
        self._running = False
        self._audio_thread = None
        self._inference_thread = None
        self._lock = threading.Lock()

        # ONNX runtime (if used)
        self._ort_session = None

    def _load_pytorch(self, model_path: str):
        """Load trained PyTorch model."""
        print(f"Loading PyTorch model from {model_path}...")
        self.model = DSCNN(
            n_classes=self.model_cfg.n_classes,
            n_mfcc=self.audio_cfg.n_mfcc,
            first_filters=self.model_cfg.first_conv_filters,
            first_kernel=self.model_cfg.first_conv_kernel,
            first_stride=self.model_cfg.first_conv_stride,
            ds_filters=self.model_cfg.ds_conv_filters,
            ds_kernels=self.model_cfg.ds_conv_kernels,
            dropout=0.0,  # No dropout at inference
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.use_onnx = False
        print("Model loaded successfully.")

    def _load_onnx(self, onnx_path: str):
        """Load ONNX model for faster CPU inference."""
        import onnxruntime as ort

        print(f"Loading ONNX model from {onnx_path}...")
        self._ort_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self.use_onnx = True
        self.model = None
        print("ONNX model loaded successfully.")

    @torch.no_grad()
    def _infer(self, mfcc: np.ndarray) -> tuple:
        """
        Run inference on MFCC features.
        Returns (predicted_label, confidence, all_probs).
        """
        if self.use_onnx and self._ort_session is not None:
            # ONNX inference
            input_tensor = mfcc[np.newaxis, np.newaxis, :, :].astype(np.float32)
            input_name = self._ort_session.get_inputs()[0].name
            outputs = self._ort_session.run(None, {input_name: input_tensor})
            logits = outputs[0][0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
        else:
            # PyTorch inference
            input_tensor = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = IDX_TO_LABEL[pred_idx]

        return label, confidence, probs

    def _audio_capture_loop(self):
        """Background thread: continuously read from microphone into ring buffer."""
        pa = pyaudio.PyAudio()

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        print("Microphone stream opened. Listening...")

        try:
            while self._running:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                with self._lock:
                    self.audio_buffer.extend(audio_chunk)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def _inference_loop(self):
        """Background thread: run inference on sliding windows from the buffer."""
        # Wait for buffer to fill
        while self._running and len(self.audio_buffer) < self.window_samples:
            time.sleep(0.1)

        print("Buffer filled. Starting inference loop...")
        print(f"Window: {self.infer_cfg.window_duration_sec}s, "
              f"Hop: {self.infer_cfg.hop_duration_sec}s")
        print(f"Confidence threshold: {self.infer_cfg.confidence_threshold}")
        print(f"Smoothing window: {self.infer_cfg.smoothing_window}")
        print(f"Cooldown: {self.infer_cfg.cooldown_sec}s")
        print("\nListening for: " + ", ".join(
            k for k in LABELS if k not in ("unknown", "silence")
        ))
        print("-" * 50)

        while self._running:
            # Get current audio window
            with self._lock:
                if len(self.audio_buffer) < self.window_samples:
                    time.sleep(0.05)
                    continue
                audio_window = np.array(list(self.audio_buffer))[-self.window_samples:]

            # Normalize
            peak = np.max(np.abs(audio_window))
            if peak > 0:
                audio_window = audio_window / peak

            # Extract features
            mfcc = extract_mfcc(audio_window, self.audio_cfg)

            # Inference
            t_start = time.perf_counter()
            label, confidence, probs = self._infer(mfcc)
            latency_ms = (time.perf_counter() - t_start) * 1000

            # Add to prediction history for smoothing
            self.prediction_history.append((label, confidence))

            # Check for keyword detection with smoothing
            self._check_detection(label, confidence, latency_ms)

            # Wait for next hop
            time.sleep(self.infer_cfg.hop_duration_sec)

    def _check_detection(self, label: str, confidence: float, latency_ms: float):
        """
        Apply smoothing logic and trigger alert if conditions are met.

        Detection requires:
          1. Confidence above threshold
          2. Same keyword detected in N consecutive windows (smoothing)
          3. Cooldown period has elapsed since last alert for this keyword
        """
        # Skip non-emergency labels
        if label in ("unknown", "silence"):
            return

        # Check confidence
        if confidence < self.infer_cfg.confidence_threshold:
            return

        # Check smoothing: require consecutive detections
        if len(self.prediction_history) < self.infer_cfg.smoothing_window:
            return

        recent = list(self.prediction_history)
        consistent = all(
            pred == label and conf >= self.infer_cfg.confidence_threshold
            for pred, conf in recent
        )

        if not consistent:
            return

        # Check cooldown
        now = time.time()
        if now - self.last_alert_time.get(label, 0) < self.infer_cfg.cooldown_sec:
            return

        # Trigger alert
        self.last_alert_time[label] = now
        self.prediction_history.clear()
        self.alert_callback(label, confidence, latency_ms)

    def _default_alert(self, keyword: str, confidence: float, latency_ms: float):
        """Default alert handler: print to console."""
        timestamp = time.strftime("%H:%M:%S")
        print(
            f"\n{'!'*50}\n"
            f"  ALERT: '{keyword.upper()}' DETECTED\n"
            f"  Confidence: {confidence:.2%}\n"
            f"  Latency: {latency_ms:.1f} ms\n"
            f"  Time: {timestamp}\n"
            f"{'!'*50}\n"
        )

    def start(self):
        """Start real-time keyword spotting."""
        if self._running:
            print("Already running.")
            return

        self._running = True
        self.audio_buffer.clear()
        self.prediction_history.clear()

        # Start audio capture thread
        self._audio_thread = threading.Thread(
            target=self._audio_capture_loop, daemon=True
        )
        self._audio_thread.start()

        # Start inference thread
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

        print("KWS engine started. Press Ctrl+C to stop.")

    def stop(self):
        """Stop real-time keyword spotting."""
        self._running = False
        if self._audio_thread:
            self._audio_thread.join(timeout=2)
        if self._inference_thread:
            self._inference_thread.join(timeout=2)
        print("KWS engine stopped.")

    def run_forever(self):
        """Convenience method: start and block until Ctrl+C."""
        self.start()
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()


def main():
    """Run real-time KWS from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Keyword Spotting")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model checkpoint (.pt) or ONNX model (.onnx)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Detection confidence threshold (default: 0.85)",
    )
    parser.add_argument(
        "--smoothing", type=int, default=3,
        help="Number of consecutive detections required (default: 3)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=3.0,
        help="Cooldown between alerts in seconds (default: 3.0)",
    )
    args = parser.parse_args()

    infer_cfg = InferenceConfig()
    infer_cfg.confidence_threshold = args.threshold
    infer_cfg.smoothing_window = args.smoothing
    infer_cfg.cooldown_sec = args.cooldown

    model_path = args.model
    use_onnx = model_path is not None and model_path.endswith(".onnx")

    kws = RealtimeKWS(
        model_path=model_path if not use_onnx else None,
        infer_cfg=infer_cfg,
        use_onnx=use_onnx,
        onnx_path=model_path if use_onnx else None,
    )
    kws.run_forever()


if __name__ == "__main__":
    main()
