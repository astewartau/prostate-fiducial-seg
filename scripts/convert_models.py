#!/usr/bin/env python3
"""
Convert prostate gold seed segmentation models from PyTorch to ONNX.

Usage (from project root):
    python scripts/convert_models.py
    python scripts/convert_models.py --quantize   # INT8 quantization (not supported in ORT WASM 1.17)

Requires:
    pip install torch onnx onnxruntime

Input:  PyTorch .pth checkpoints from models/
Output: ONNX models in web/models/
"""

import sys
import os
import glob
import numpy as np

import torch

# Add project root to path so we can import from scripts.models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.models.unet3d import UNet3D

# ==================== Configuration ====================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "web", "models")

# Find best checkpoints by glob pattern
def find_model_checkpoints():
    """Find all *-best.pth model checkpoints in models/."""
    pattern = os.path.join(MODELS_DIR, "T1-*-best.pth")
    paths = sorted(glob.glob(pattern))
    models = {}
    for path in paths:
        # Extract seed from filename like T1-20260218-213325-seed42-best.pth
        basename = os.path.basename(path)
        for part in basename.split("-"):
            if part.startswith("seed"):
                seed = int(part.replace("seed", ""))
                models[seed] = path
                break
    return models


# ==================== Conversion ====================

def load_model(checkpoint_path):
    """Load a UNet3D model from a PyTorch checkpoint."""
    model = UNet3D(in_channels=1, out_channels=3)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle both raw state_dict and checkpoint dict formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_onnx(model, output_path, opset_version=17):
    """Export a PyTorch model to ONNX format using the legacy exporter."""
    # Use a typical prostate volume size (padded to 32)
    dummy_input = torch.randn(1, 1, 256, 256, 32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "depth", 3: "height", 4: "width"},
            "output": {2: "depth", 3: "height", 4: "width"},
        },
        dynamo=False,  # Use legacy exporter for broad ONNX runtime compatibility
    )
    print(f"  Exported ONNX: {output_path}")


def quantize_model(input_path, output_path):
    """Apply INT8 dynamic quantization to an ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"  Quantized: {output_path}")


def verify_model(onnx_path):
    """Verify an ONNX model runs correctly with a dummy input."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 1, 256, 256, 32).astype(np.float32)
    result = session.run(None, {"input": dummy})
    output = result[0]
    print(f"  Verified: output shape {output.shape}, "
          f"range [{output.min():.3f}, {output.max():.3f}]")
    return output.shape == (1, 3, 256, 256, 32)


def main():
    models = find_model_checkpoints()
    if not models:
        print(f"No model checkpoints found in {MODELS_DIR}/")
        print("Expected pattern: T1-*-best.pth")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    quantize = '--quantize' in sys.argv

    print(f"Found {len(models)} model(s): seeds {list(models.keys())}")

    for seed, checkpoint_path in models.items():
        print(f"\n{'='*60}")
        print(f"Processing seed {seed}: {os.path.basename(checkpoint_path)}")
        print(f"{'='*60}")

        # Load model
        print("  Loading PyTorch model...")
        model = load_model(checkpoint_path)

        # Export to ONNX
        output_path = os.path.join(OUTPUT_DIR, f"seedseg-model-seed{seed}.onnx")

        if quantize:
            # Export FP32 first, then quantize
            fp32_path = os.path.join(OUTPUT_DIR, f"seedseg-model-seed{seed}-fp32.onnx")
            print("  Exporting to ONNX (FP32)...")
            export_to_onnx(model, fp32_path)

            print("  Quantizing to INT8...")
            quantize_model(fp32_path, output_path)

            # Remove FP32 intermediate files
            os.remove(fp32_path)
            data_file = fp32_path + ".data"
            if os.path.exists(data_file):
                os.remove(data_file)
        else:
            print("  Exporting to ONNX (FP32)...")
            export_to_onnx(model, output_path)

        # Verify
        print("  Verifying model...")
        ok = verify_model(output_path)
        if ok:
            print(f"  SUCCESS: seed {seed}")
        else:
            print(f"  FAILED: unexpected output shape for seed {seed}")

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f}MB")

    print(f"\nDone! ONNX models saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
