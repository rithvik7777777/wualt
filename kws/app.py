"""
Streamlit web demo for Emergency Keyword Spotting.

Upload or record audio and detect emergency keywords in real time.
Deployable to Streamlit Cloud, Hugging Face Spaces, or any server.
"""
import os
import sys
import io
import time
import tempfile
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from config import AudioConfig, ModelConfig, InferenceConfig, IDX_TO_LABEL, LABELS
from model.ds_cnn import DSCNN, count_parameters
from data.dataset import extract_mfcc, load_and_preprocess_audio

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Emergency Keyword Spotter",
    page_icon="🚨",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────
AUDIO_CFG = AudioConfig()
MODEL_CFG = ModelConfig()
INFER_CFG = InferenceConfig()

EMERGENCY_KEYWORDS = {"emergency"}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")


# ── Model loading (cached) ──────────────────────────────────
@st.cache_resource
def load_model():
    """Load the DS-CNN model once and cache it."""
    model = DSCNN(
        n_classes=MODEL_CFG.n_classes,
        n_mfcc=AUDIO_CFG.n_mfcc,
        first_filters=MODEL_CFG.first_conv_filters,
        first_kernel=MODEL_CFG.first_conv_kernel,
        first_stride=MODEL_CFG.first_conv_stride,
        ds_filters=MODEL_CFG.ds_conv_filters,
        ds_kernels=MODEL_CFG.ds_conv_kernels,
        dropout=0.0,  # No dropout at inference time
    )

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`. Run `python run_train.py` first.")
        st.stop()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(model, audio: np.ndarray) -> dict:
    """
    Run keyword detection on a 1-second audio clip.
    Returns dict with label, confidence, and per-class probabilities.
    """
    # Pad or trim
    if len(audio) < AUDIO_CFG.n_samples:
        audio = np.pad(audio, (0, AUDIO_CFG.n_samples - len(audio)), mode="constant")
    elif len(audio) > AUDIO_CFG.n_samples:
        audio = audio[:AUDIO_CFG.n_samples]

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    # Extract MFCC
    mfcc = extract_mfcc(audio, AUDIO_CFG)

    # Inference
    input_tensor = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=-1).numpy()[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    pred_idx = int(np.argmax(probs))
    label = IDX_TO_LABEL[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
        "latency_ms": latency_ms,
        "is_emergency": label in EMERGENCY_KEYWORDS,
    }


def analyze_long_audio(model, audio: np.ndarray, sr: int) -> list:
    """
    Sliding-window analysis over audio longer than 1 second.
    Returns list of per-window results.
    """
    # Resample if needed
    if sr != AUDIO_CFG.sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_CFG.sample_rate)
        sr = AUDIO_CFG.sample_rate

    window_size = AUDIO_CFG.n_samples
    hop_size = window_size // 2  # 50% overlap
    results = []

    for start in range(0, len(audio) - window_size + 1, hop_size):
        chunk = audio[start : start + window_size]
        result = predict(model, chunk)
        result["time_start"] = start / sr
        result["time_end"] = (start + window_size) / sr
        results.append(result)

    # Handle remaining audio if any
    if len(audio) > window_size and (len(audio) % hop_size != 0):
        chunk = audio[-window_size:]
        result = predict(model, chunk)
        result["time_start"] = (len(audio) - window_size) / sr
        result["time_end"] = len(audio) / sr
        results.append(result)

    return results


# ── UI ───────────────────────────────────────────────────────
def main():
    st.title("Emergency Keyword Spotter")
    st.markdown(
        "Detects emergency keywords (**help**, **danger**, **call 911**) from audio. "
        "Powered by a DS-CNN model optimized for edge deployment."
    )

    model = load_model()

    # Sidebar: model info
    with st.sidebar:
        st.header("Model Info")
        info = count_parameters(model)
        st.metric("Parameters", f"{info['total_params']:,}")
        st.metric("Size (FP32)", f"{info['size_fp32_kb']:.1f} KB")
        st.metric("Size (INT8)", f"{info['size_int8_kb']:.1f} KB")

        st.divider()
        st.header("Settings")
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5, max_value=0.99, value=0.85, step=0.05,
            help="Minimum confidence to trigger an alert",
        )

        st.divider()
        st.header("Keywords")
        for kw in EMERGENCY_KEYWORDS:
            st.markdown(f"- `{kw}`")

    # Main area: tabs
    tab_upload, tab_record = st.tabs(["Upload Audio", "Record Audio"])

    # ── Tab 1: Upload ────────────────────────────────────────
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a WAV file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="Upload an audio file to analyze for emergency keywords",
        )

        if uploaded_file is not None:
            # Save to temp file and load
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            audio, sr = sf.read(tmp_path, dtype="float32")
            os.unlink(tmp_path)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            st.audio(uploaded_file, format="audio/wav")

            duration = len(audio) / sr
            st.caption(f"Duration: {duration:.2f}s | Sample rate: {sr} Hz")

            if st.button("Analyze", key="analyze_upload", type="primary"):
                with st.spinner("Analyzing..."):
                    if duration <= 1.5:
                        # Single prediction
                        result = predict(model, audio if sr == AUDIO_CFG.sample_rate else
                                         librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_CFG.sample_rate))
                        _show_single_result(result, threshold)
                    else:
                        # Sliding window
                        results = analyze_long_audio(model, audio, sr)
                        _show_timeline_results(results, threshold)

    # ── Tab 2: Record ────────────────────────────────────────
    with tab_record:
        st.markdown(
            "Record a short audio clip using your browser's microphone."
        )

        audio_bytes = st.audio_input("Record audio (say 'help', 'danger', or 'call 911')")

        if audio_bytes is not None:
            # Decode the recorded audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes.getvalue())
                tmp_path = tmp.name

            audio, sr = sf.read(tmp_path, dtype="float32")
            os.unlink(tmp_path)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            duration = len(audio) / sr
            st.caption(f"Recorded: {duration:.2f}s | Sample rate: {sr} Hz")

            if st.button("Analyze Recording", key="analyze_record", type="primary"):
                with st.spinner("Analyzing..."):
                    if duration <= 1.5:
                        resampled = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_CFG.sample_rate) if sr != AUDIO_CFG.sample_rate else audio
                        result = predict(model, resampled)
                        _show_single_result(result, threshold)
                    else:
                        results = analyze_long_audio(model, audio, sr)
                        _show_timeline_results(results, threshold)


def _show_single_result(result: dict, threshold: float):
    """Display a single prediction result."""
    is_alert = result["is_emergency"] and result["confidence"] >= threshold

    if is_alert:
        st.error(
            f"EMERGENCY DETECTED: **{result['label'].upper()}** "
            f"(confidence: {result['confidence']:.1%})",
            icon="🚨",
        )
    elif result["is_emergency"]:
        st.warning(
            f"Possible keyword: **{result['label']}** "
            f"(confidence: {result['confidence']:.1%}, below threshold {threshold:.0%})",
            icon="⚠️",
        )
    else:
        st.success(
            f"No emergency detected. Classified as: **{result['label']}** "
            f"({result['confidence']:.1%})",
            icon="✅",
        )

    # Probabilities bar chart
    st.subheader("Class Probabilities")
    prob_data = result["probabilities"]
    cols = st.columns(len(prob_data))
    for col, (label, prob) in zip(cols, prob_data.items()):
        col.metric(label, f"{prob:.1%}")

    st.caption(f"Inference latency: {result['latency_ms']:.2f} ms")


def _show_timeline_results(results: list, threshold: float):
    """Display sliding-window results for longer audio."""
    # Check for any emergency detections
    emergencies = [
        r for r in results
        if r["is_emergency"] and r["confidence"] >= threshold
    ]

    if emergencies:
        st.error(
            f"EMERGENCY DETECTED in {len(emergencies)} segment(s)!",
            icon="🚨",
        )
    else:
        st.success("No emergency keywords detected.", icon="✅")

    # Results table
    st.subheader("Segment Analysis")
    for i, r in enumerate(results):
        is_alert = r["is_emergency"] and r["confidence"] >= threshold
        icon = "🚨" if is_alert else "✅"

        with st.expander(
            f"{icon} [{r['time_start']:.1f}s - {r['time_end']:.1f}s] "
            f"**{r['label']}** ({r['confidence']:.1%})",
            expanded=is_alert,
        ):
            prob_data = r["probabilities"]
            cols = st.columns(len(prob_data))
            for col, (label, prob) in zip(cols, prob_data.items()):
                col.metric(label, f"{prob:.1%}")
            st.caption(f"Latency: {r['latency_ms']:.2f} ms")

    avg_latency = np.mean([r["latency_ms"] for r in results])
    st.caption(f"Average inference latency: {avg_latency:.2f} ms | {len(results)} segments analyzed")


if __name__ == "__main__":
    main()
