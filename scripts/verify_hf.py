"""
Independent verification script for Hugging Face release:
1. Verify HTTP 200 via Hugging Face Hub API.
2. Snapshot download into clean temporary directory.
3. Compute and assert SHA-256 for all 4 weights.
4. Load YOLOv8n ONNX via ONNX Runtime.
5. Execute CPU smoke inference on a real test image.
6. Verify output tensor shape [1, 5, 1029] and assert no NaN/Inf.
"""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
from huggingface_hub import HfApi, snapshot_download

REPO_ID = "neitpm/vn-plate-detection-benchmark"
TEST_IMG = Path("datasets/Vietnam license-plate.v1i.yolo26/test/images/CarLongPlate115_jpg.rf.122cb025d8830754c49e2956145c18a3.jpg")

EXPECTED_HASHES = {
    "weights/yolov8n_edge.onnx": "28c4c12958f705ae658c6901f2e9e755a7e03f629a73c23750c7d37761af31dd",
    "weights/yolov8n_edge.pt": "873851543beeb9c6483c07e64b99400d1554a70244862b1cd70b62035b6431c2",
    "weights/yolov8l_server.onnx": "21f2e3aefc46dadd0b4b7e4a0200d0443644c41e982f9d756ef17231b74eaaa3",
    "weights/yolov8l_server.pt": "8674e24abccf02522f12c34d6829efc3c0e13ab0c53a3ea4850bd6fe5a0671fd"
}

def letterbox(im, new_shape=224, color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape - new_unpad[0]) / 2.0, (new_shape - new_unpad[1]) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded

def main():
    print(f"--- 1. Checking Hugging Face Repository HTTP Status for '{REPO_ID}' ---")
    api = HfApi()
    try:
        repo_info = api.repo_info(repo_id=REPO_ID, repo_type="model")
        print(f"  [OK] Repository exists and is accessible. ID: {repo_info.id}")
    except Exception as e:
        print(f"ERROR: Could not access repository {REPO_ID}: {e}", file=sys.stderr)
        sys.exit(1)

    temp_dir = Path(tempfile.mkdtemp(prefix="hf_verify_"))
    try:
        print(f"\n--- 2. Snapshot Downloading into clean directory: {temp_dir} ---")
        downloaded_path = Path(snapshot_download(repo_id=REPO_ID, repo_type="model", local_dir=str(temp_dir)))

        print("\n--- 3. Verifying SHA-256 Checksums ---")
        for rel_path, expected_hash in EXPECTED_HASHES.items():
            file_path = downloaded_path / rel_path
            if not file_path.exists():
                print(f"ERROR: Expected file '{rel_path}' not found in download.", file=sys.stderr)
                sys.exit(1)
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                print(f"ERROR: Hash mismatch for '{rel_path}':\n  Expected: {expected_hash}\n  Actual:   {actual_hash}", file=sys.stderr)
                sys.exit(1)
            print(f"  [VERIFIED] {rel_path} -> {actual_hash[:16]}... (MATCH)")

        print("\n--- 4. Loading YOLOv8n ONNX via ONNX Runtime & Running Smoke Inference ---")
        onnx_model_path = downloaded_path / "weights" / "yolov8n_edge.onnx"
        sess = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
        
        im0 = cv2.imread(str(TEST_IMG))
        assert im0 is not None, f"Could not read test image {TEST_IMG}"
        im_lb = letterbox(im0, 224)
        rgb = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        output = sess.run(None, {"images": tensor})[0]
        print(f"  [OK] ONNX Output shape: {output.shape}")
        assert output.shape == (1, 5, 1029), f"Unexpected output shape: {output.shape}"
        assert not np.isnan(output).any(), "Output contains NaN values!"
        assert not np.isinf(output).any(), "Output contains Inf values!"
        
        max_conf = float(output[0, 4].max())
        print(f"  [OK] Smoke inference successful! Max plate confidence: {max_conf * 100:.2f}%")
        assert max_conf > 0.50, f"Expected positive detection, got max conf {max_conf}"

        print("\n==========================================")
        print("ALL HUGGING FACE RELEASE CHECKS PASSED 100%")
        print("==========================================")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
