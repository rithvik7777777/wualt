"""
Vercel serverless function for KWS inference.
Receives audio via POST, runs ONNX model, returns predictions.
"""
import os
import io
import time
import json
import tempfile
import numpy as np

# ---------- Configuration ----------
LABELS = ["help", "danger", "call_911", "unknown", "silence"]
SAMPLE_RATE = 16000
DURATION_SEC = 1.0
N_SAMPLES = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 480
N_MELS = 40
FMIN = 20
FMAX = 4000

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "kws_model.onnx")

# ---------- Lazy-loaded globals ----------
_session = None


def _get_session():
    """Load ONNX model once, reuse across invocations (warm starts)."""
    global _session
    if _session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(MODEL_PATH, sess_options=opts)
    return _session


# ---------- Minimal MFCC extraction (no librosa) ----------
def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels, n_fft, sr, fmin, fmax):
    """Create a Mel filterbank matrix."""
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, center):
            if center != left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                fb[i, j] = (right - j) / (right - center)
    return fb


def _extract_mfcc(audio, sr=SAMPLE_RATE):
    """Extract MFCC features matching the training pipeline (librosa-based)."""
    import librosa

    # Pad or trim to exactly N_SAMPLES
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    # Normalize
    mx = np.max(np.abs(audio))
    if mx > 0:
        audio = audio / mx

    # Extract MFCCs using librosa (matches training pipeline exactly)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    # Model expects exactly 101 time steps — pad or trim
    target_time = 101
    if mfcc.shape[1] < target_time:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_time - mfcc.shape[1])))
    elif mfcc.shape[1] > target_time:
        mfcc = mfcc[:, :target_time]

    return mfcc.astype(np.float32)


def _decode_audio(file_bytes: bytes) -> np.ndarray:
    """Decode audio bytes to numpy array at 16kHz mono."""
    import soundfile as sf

    # Try direct read first (works for wav, ogg, flac)
    try:
        audio, sr = sf.read(io.BytesIO(file_bytes), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            # Simple resampling via linear interpolation
            duration = len(audio) / sr
            target_len = int(duration * SAMPLE_RATE)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        return audio
    except Exception:
        pass

    # Fallback: use ffmpeg for webm/mp3
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_in:
        tmp_in.write(file_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace('.webm', '.wav')
    try:
        import subprocess
        subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_in_path, '-ar', str(SAMPLE_RATE),
             '-ac', '1', '-f', 'wav', tmp_out_path],
            capture_output=True, timeout=10
        )
        audio, sr = sf.read(tmp_out_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio
    finally:
        for p in [tmp_in_path, tmp_out_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------- Vercel handler ----------
def handler(request):
    """Handle POST /api/predict with audio file upload."""
    from http.server import BaseHTTPRequestHandler

    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            },
            'body': ''
        }

    if request.method != 'POST':
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'Method not allowed'})
        }

    try:
        # Parse multipart form data
        content_type = request.headers.get('content-type', '')
        if 'multipart/form-data' not in content_type:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Expected multipart/form-data'})
            }

        # Read file from request body
        body = request.body
        if isinstance(body, str):
            body = body.encode()

        # Extract audio file from multipart
        audio_bytes = _parse_multipart(body, content_type)
        if audio_bytes is None:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No audio file found'})
            }

        # Decode audio
        audio = _decode_audio(audio_bytes)

        # Extract features
        mfcc = _extract_mfcc(audio)

        # Run inference
        session = _get_session()
        input_tensor = mfcc[np.newaxis, np.newaxis, :, :]  # (1, 1, n_mfcc, time)
        input_name = session.get_inputs()[0].name

        t0 = time.time()
        outputs = session.run(None, {input_name: input_tensor})
        inference_ms = round((time.time() - t0) * 1000, 1)

        logits = outputs[0][0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        pred_idx = int(np.argmax(probs))
        keyword = LABELS[pred_idx]
        probabilities = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}

        result = {
            'keyword': keyword,
            'confidence': round(float(probs[pred_idx]), 4),
            'probabilities': probabilities,
            'inference_ms': inference_ms,
        }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps(result)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }


def _parse_multipart(body: bytes, content_type: str) -> bytes:
    """Extract the first file from multipart form data."""
    # Get boundary
    for part in content_type.split(';'):
        part = part.strip()
        if part.startswith('boundary='):
            boundary = part[9:].strip('"')
            break
    else:
        return None

    boundary_bytes = boundary.encode()
    parts = body.split(b'--' + boundary_bytes)

    for part in parts:
        if b'filename=' in part:
            # Find the blank line separating headers from content
            header_end = part.find(b'\r\n\r\n')
            if header_end == -1:
                header_end = part.find(b'\n\n')
                if header_end == -1:
                    continue
                content = part[header_end + 2:]
            else:
                content = part[header_end + 4:]

            # Remove trailing boundary markers
            if content.endswith(b'\r\n'):
                content = content[:-2]
            elif content.endswith(b'--\r\n'):
                content = content[:-4]
            if content.endswith(b'--'):
                content = content[:-2]
            if content.endswith(b'\r\n'):
                content = content[:-2]

            return content

    return None
