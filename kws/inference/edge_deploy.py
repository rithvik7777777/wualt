"""
Edge deployment utilities for the KWS model.

Provides:
  1. PyTorch -> ONNX export
  2. ONNX -> TensorFlow Lite conversion
  3. INT8 quantization (post-training)
  4. Benchmarking and latency measurement
  5. Raspberry Pi deployment guide

The deployment pipeline:
  PyTorch (.pt) -> ONNX (.onnx) -> TFLite (.tflite) -> INT8 TFLite (.tflite)
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import AudioConfig, ModelConfig, InferenceConfig, LABELS
from model.ds_cnn import DSCNN, count_parameters


def export_onnx(
    model_path: str,
    output_path: str = None,
    audio_cfg: AudioConfig = None,
    model_cfg: ModelConfig = None,
):
    """
    Export PyTorch model to ONNX format.

    ONNX provides a portable representation that can be:
      - Run with ONNX Runtime (fast CPU inference)
      - Converted to TFLite for microcontrollers
      - Optimized with various backends
    """
    audio_cfg = audio_cfg or AudioConfig()
    model_cfg = model_cfg or ModelConfig()

    if output_path is None:
        output_path = model_path.replace(".pt", ".onnx")

    print(f"Exporting to ONNX: {output_path}")

    # Load model
    model = DSCNN(
        n_classes=model_cfg.n_classes,
        n_mfcc=audio_cfg.n_mfcc,
        first_filters=model_cfg.first_conv_filters,
        first_kernel=model_cfg.first_conv_kernel,
        first_stride=model_cfg.first_conv_stride,
        ds_filters=model_cfg.ds_conv_filters,
        ds_kernels=model_cfg.ds_conv_kernels,
        dropout=0.0,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy input matching the expected MFCC shape
    # time_steps = ceil(n_samples / hop_length) + 1 ≈ 101
    dummy_input = torch.randn(1, 1, audio_cfg.n_mfcc, 101)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    # Run shape inference to fix any missing shape info
    import onnx
    from onnx import shape_inference
    onnx_model = onnx.load(output_path)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, output_path)
    onnx.checker.check_model(onnx_model)

    file_size = os.path.getsize(output_path)
    print(f"ONNX model exported: {file_size / 1024:.1f} KB")
    print(f"Saved to: {output_path}")

    return output_path


def export_tflite(
    onnx_path: str,
    output_path: str = None,
):
    """
    Convert ONNX model to TensorFlow Lite.

    TFLite is the standard format for:
      - Raspberry Pi (via TFLite runtime)
      - Android (via TFLite Android API)
      - Microcontrollers (via TFLite Micro)
    """
    if output_path is None:
        output_path = onnx_path.replace(".onnx", ".tflite")

    print(f"Converting ONNX to TFLite: {output_path}")

    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        # ONNX -> TensorFlow SavedModel
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)

        saved_model_dir = onnx_path.replace(".onnx", "_saved_model")
        tf_rep.export_graph(saved_model_dir)

        # SavedModel -> TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        file_size = os.path.getsize(output_path)
        print(f"TFLite model exported: {file_size / 1024:.1f} KB")

    except ImportError:
        print("TFLite conversion requires: pip install onnx-tf tensorflow")
        print("Falling back to ONNX Runtime for edge inference.")
        print("\nAlternative: Use onnxruntime on Raspberry Pi directly:")
        print("  pip install onnxruntime")
        print("  # ONNX Runtime is well-supported on ARM devices")
        return None

    return output_path


def quantize_tflite(
    tflite_path: str,
    output_path: str = None,
    calibration_data: np.ndarray = None,
):
    """
    Apply INT8 post-training quantization to TFLite model.

    INT8 quantization:
      - Reduces model size by ~4x (FP32 -> INT8)
      - Speeds up inference on devices with INT8 support
      - Minimal accuracy loss (typically < 1%)
    """
    if output_path is None:
        output_path = tflite_path.replace(".tflite", "_int8.tflite")

    try:
        import tensorflow as tf

        # For full integer quantization, we need representative data
        def representative_dataset():
            if calibration_data is not None:
                for i in range(min(100, len(calibration_data))):
                    yield [calibration_data[i:i+1].astype(np.float32)]
            else:
                # Generate random calibration data (replace with real data in production)
                for _ in range(100):
                    yield [np.random.randn(1, 1, 40, 101).astype(np.float32)]

        # Load the base TFLite model's SavedModel
        saved_model_dir = tflite_path.replace(".tflite", "").replace("_int8", "") + "_saved_model"

        if os.path.exists(saved_model_dir):
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        else:
            print(f"SavedModel not found at {saved_model_dir}")
            print("Quantizing from existing TFLite model...")
            # Read existing tflite and re-quantize isn't directly supported,
            # so we use the interpreter-based approach
            return _quantize_onnx_direct(tflite_path, output_path)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        quantized_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(quantized_model)

        file_size = os.path.getsize(output_path)
        print(f"Quantized INT8 model: {file_size / 1024:.1f} KB")
        print(f"Saved to: {output_path}")

    except ImportError:
        print("Quantization requires: pip install tensorflow")
        print("\nFor ONNX-based quantization, use onnxruntime quantization tools.")
        return _quantize_onnx_direct(tflite_path, output_path)

    return output_path


def _quantize_onnx_direct(model_path: str, output_path: str):
    """
    Alternative: quantize ONNX model directly using onnxruntime.
    Works without TensorFlow.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process

        onnx_path = model_path.replace(".tflite", ".onnx")
        quantized_onnx = output_path.replace(".tflite", ".onnx")

        if not os.path.exists(onnx_path):
            print(f"ONNX model not found at {onnx_path}")
            return None

        # Preprocess: run shape inference and optimization before quantization
        preprocessed_path = onnx_path.replace(".onnx", "_preprocessed.onnx")
        try:
            quant_pre_process(onnx_path, preprocessed_path)
            source_path = preprocessed_path
        except Exception as e:
            print(f"Preprocessing failed ({e}), quantizing directly...")
            source_path = onnx_path

        quantize_dynamic(
            source_path,
            quantized_onnx,
            weight_type=QuantType.QInt8,
        )

        # Clean up preprocessed file
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)

        file_size = os.path.getsize(quantized_onnx)
        print(f"Quantized ONNX model: {file_size / 1024:.1f} KB")
        print(f"Saved to: {quantized_onnx}")
        return quantized_onnx

    except ImportError:
        print("Install onnxruntime for quantization: pip install onnxruntime")
        return None


def benchmark_model(model_path: str, n_runs: int = 100, use_onnx: bool = False):
    """
    Benchmark model inference latency.
    Measures average, min, max, and P95 latency over n_runs.
    """
    audio_cfg = AudioConfig()
    dummy_input = np.random.randn(1, 1, audio_cfg.n_mfcc, 101).astype(np.float32)

    if use_onnx or model_path.endswith(".onnx"):
        import onnxruntime as ort

        session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            latencies.append((time.perf_counter() - t0) * 1000)

    else:
        model_cfg = ModelConfig()
        model = DSCNN(
            n_classes=model_cfg.n_classes,
            n_mfcc=audio_cfg.n_mfcc,
            first_filters=model_cfg.first_conv_filters,
            first_kernel=model_cfg.first_conv_kernel,
            first_stride=model_cfg.first_conv_stride,
            ds_filters=model_cfg.ds_conv_filters,
            ds_kernels=model_cfg.ds_conv_kernels,
            dropout=0.0,
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        input_tensor = torch.from_numpy(dummy_input)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(input_tensor)

        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(input_tensor)
                latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)

    print(f"\nBenchmark Results ({n_runs} runs):")
    print(f"  Mean:  {latencies.mean():.2f} ms")
    print(f"  Std:   {latencies.std():.2f} ms")
    print(f"  Min:   {latencies.min():.2f} ms")
    print(f"  Max:   {latencies.max():.2f} ms")
    print(f"  P50:   {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:   {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:   {np.percentile(latencies, 99):.2f} ms")

    return {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def deploy_raspberry_pi():
    """
    Print step-by-step Raspberry Pi deployment instructions.
    """
    instructions = """
╔══════════════════════════════════════════════════════════════╗
║         RASPBERRY PI DEPLOYMENT GUIDE                       ║
╚══════════════════════════════════════════════════════════════╝

PREREQUISITES:
  - Raspberry Pi 3B+ or newer (Pi 4 recommended)
  - Raspbian OS (64-bit recommended for Pi 4)
  - USB microphone or I2S MEMS microphone (e.g., INMP441)
  - Python 3.8+

STEP 1: System Setup
─────────────────────
  sudo apt-get update && sudo apt-get upgrade -y
  sudo apt-get install -y python3-pip python3-venv portaudio19-dev
  sudo apt-get install -y libatlas-base-dev libopenblas-dev

STEP 2: Python Environment
───────────────────────────
  python3 -m venv kws_env
  source kws_env/bin/activate

STEP 3: Install Dependencies
─────────────────────────────
  pip install numpy librosa soundfile pyaudio onnxruntime

  # Note: Use onnxruntime (not torch) on Pi for minimal footprint
  # onnxruntime has prebuilt ARM wheels

STEP 4: Copy Model Files
─────────────────────────
  # From your dev machine:
  scp checkpoints/best_model.onnx pi@<PI_IP>:~/kws/
  scp -r inference/ config.py pi@<PI_IP>:~/kws/

STEP 5: Test Microphone
────────────────────────
  arecord -l                    # List audio devices
  arecord -d 3 test.wav         # Record 3 seconds
  aplay test.wav                # Playback

STEP 6: Run Inference
──────────────────────
  cd ~/kws
  python inference/realtime.py --model best_model.onnx --threshold 0.85

STEP 7: Run as Service (Optional)
──────────────────────────────────
  # Create systemd service for auto-start:
  sudo nano /etc/systemd/system/kws.service

  [Unit]
  Description=Keyword Spotting Service
  After=multi-user.target sound.target

  [Service]
  Type=simple
  User=pi
  WorkingDirectory=/home/pi/kws
  ExecStart=/home/pi/kws/kws_env/bin/python inference/realtime.py --model best_model.onnx
  Restart=always
  RestartSec=5

  [Install]
  WantedBy=multi-user.target

  # Enable and start:
  sudo systemctl enable kws
  sudo systemctl start kws
  sudo systemctl status kws

PERFORMANCE EXPECTATIONS (Raspberry Pi 4):
──────────────────────────────────────────
  - ONNX Runtime inference: ~5-15 ms per window
  - Total pipeline latency: ~50-100 ms
  - CPU usage: ~10-20% single core
  - RAM usage: ~50-100 MB
  - Power: ~2-3W

OPTIMIZATION TIPS:
──────────────────
  1. Use ONNX Runtime (not PyTorch) - 3-5x faster on ARM
  2. INT8 quantized model reduces memory and may speed up inference
  3. Increase hop duration to 0.75s to reduce CPU usage
  4. Use ALSA directly instead of PyAudio for lower audio latency
  5. Pin process to a specific CPU core: taskset -c 0 python ...
  6. Disable GUI (headless mode) to free resources
"""
    print(instructions)


def main():
    """Edge deployment CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="KWS Edge Deployment Tools")
    subparsers = parser.add_subparsers(dest="command")

    # Export ONNX
    export_parser = subparsers.add_parser("export-onnx", help="Export to ONNX")
    export_parser.add_argument("--model", required=True, help="PyTorch model path")
    export_parser.add_argument("--output", help="Output ONNX path")

    # Export TFLite
    tflite_parser = subparsers.add_parser("export-tflite", help="Export to TFLite")
    tflite_parser.add_argument("--onnx", required=True, help="ONNX model path")
    tflite_parser.add_argument("--output", help="Output TFLite path")

    # Quantize
    quant_parser = subparsers.add_parser("quantize", help="INT8 quantization")
    quant_parser.add_argument("--model", required=True, help="Model path")
    quant_parser.add_argument("--output", help="Output path")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark latency")
    bench_parser.add_argument("--model", required=True, help="Model path")
    bench_parser.add_argument("--runs", type=int, default=100)

    # Deploy guide
    subparsers.add_parser("rpi-guide", help="Raspberry Pi deployment guide")

    args = parser.parse_args()

    if args.command == "export-onnx":
        export_onnx(args.model, args.output)
    elif args.command == "export-tflite":
        export_tflite(args.onnx, args.output)
    elif args.command == "quantize":
        if args.model.endswith(".onnx"):
            _quantize_onnx_direct(args.model, args.output or args.model.replace(".onnx", "_int8.onnx"))
        else:
            quantize_tflite(args.model, args.output)
    elif args.command == "benchmark":
        benchmark_model(args.model, n_runs=args.runs)
    elif args.command == "rpi-guide":
        deploy_raspberry_pi()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
