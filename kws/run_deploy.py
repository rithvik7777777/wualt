#!/usr/bin/env python3
"""
Step 4: Export and quantize model for edge deployment.
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from config import InferenceConfig
from inference.edge_deploy import (
    export_onnx, export_tflite, quantize_tflite,
    benchmark_model, deploy_raspberry_pi,
)


def main():
    parser = argparse.ArgumentParser(description="Deploy KWS model to edge")
    parser.add_argument(
        "--model", type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt"),
        help="Path to trained model",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--rpi-guide", action="store_true", help="Show RPi deployment guide")
    args = parser.parse_args()

    if args.rpi_guide:
        deploy_raspberry_pi()
        return

    # Step 1: Export to ONNX
    print("\n" + "=" * 60)
    print("STEP 1: Export to ONNX")
    print("=" * 60)
    onnx_path = export_onnx(args.model)

    # Step 2: Try TFLite conversion
    print("\n" + "=" * 60)
    print("STEP 2: Convert to TFLite")
    print("=" * 60)
    tflite_path = export_tflite(onnx_path)

    # Step 3: Quantize
    print("\n" + "=" * 60)
    print("STEP 3: INT8 Quantization")
    print("=" * 60)
    if tflite_path:
        quantize_tflite(tflite_path)
    else:
        print("Quantizing ONNX model directly...")
        from inference.edge_deploy import _quantize_onnx_direct
        _quantize_onnx_direct(
            onnx_path,
            onnx_path.replace(".onnx", "_int8.onnx"),
        )

    # Step 4: Benchmark
    if args.benchmark:
        print("\n" + "=" * 60)
        print("STEP 4: Benchmark")
        print("=" * 60)
        print("\nPyTorch model:")
        benchmark_model(args.model, use_onnx=False)
        print("\nONNX model:")
        benchmark_model(onnx_path, use_onnx=True)

    print("\n" + "=" * 60)
    print("DEPLOYMENT READY")
    print("=" * 60)
    print(f"\nONNX model:     {onnx_path}")
    if tflite_path:
        print(f"TFLite model:   {tflite_path}")
    print("\nRun with: python run_deploy.py --rpi-guide  for Pi instructions")


if __name__ == "__main__":
    main()
