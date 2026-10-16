"""
Independent standalone Python inference script for Vietnamese License Plate Detection.
Supports both PyTorch (.pt) checkpoints and ONNX Runtime (.onnx) graphs.
Implements exact Ultralytics Letterbox padding and post-NMS inverse coordinate mapping.

Usage:
    python scripts/infer.py --weights vn-plate-app/app/src/main/assets/best.onnx --source datasets/Vietnam\ license-plate.v1i.yolo26/test/images/CarLongPlate115_jpg.rf.122cb025d8830754c49e2956145c18a3.jpg --imgsz 224
    python scripts/infer.py --weights vn-plate-20260805T073942Z-1-001/vn-plate/runs/train/vn_plate_stage2_finetune/weights/best.onnx --source images.jpg --imgsz 640
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

def letterbox(im, new_shape=224, color=(114, 114, 114)):
    """Ultralytics exact letterbox padding."""
    shape = im.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)

def nms(boxes, scores, iou_thresh=0.5):
    """Greedy Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

class Detector:
    def __init__(self, weights_path, imgsz=224, conf_thresh=0.5, iou_thresh=0.5):
        self.weights_path = Path(weights_path)
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.is_onnx = self.weights_path.suffix.lower() == '.onnx'

        if self.is_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(self.weights_path))
            self.input_name = self.session.get_inputs()[0].name
            # Inspect metadata if available
            try:
                import onnx
                m = onnx.load(str(self.weights_path))
                props = {p.key: p.value for p in m.metadata_props}
                if 'imgsz' in props:
                    val = eval(props['imgsz'])
                    self.imgsz = val[0] if isinstance(val, (list, tuple)) else val
            except Exception:
                pass
        else:
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))

    def predict(self, image_path):
        im0 = cv2.imread(str(image_path))
        if im0 is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        h0, w0 = im0.shape[:2]

        t0 = time.perf_counter()

        if self.is_onnx:
            # 1. Preprocess
            im_lb, r, (pad_l, pad_t) = letterbox(im0, self.imgsz)
            rgb = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB)
            tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

            # 2. Forward
            t_infer = time.perf_counter()
            out = self.session.run(None, {self.input_name: tensor})[0]
            infer_ms = (time.perf_counter() - t_infer) * 1000.0

            # 3. Decode & NMS
            # Shape [1, 5, N]
            cxs, cys, ws, hs, confs = out[0, 0], out[0, 1], out[0, 2], out[0, 3], out[0, 4]
            x1s = cxs - ws / 2.0
            y1s = cys - hs / 2.0
            x2s = cxs + ws / 2.0
            y2s = cys + hs / 2.0
            boxes = np.stack([x1s, y1s, x2s, y2s], axis=1)

            mask = confs >= self.conf_thresh
            filtered_boxes = boxes[mask]
            filtered_confs = confs[mask]
            keep = nms(filtered_boxes, filtered_confs, iou_thresh=self.iou_thresh)

            detections = []
            for idx in keep:
                b = filtered_boxes[idx]
                score = float(filtered_confs[idx])
                # Inverse letterbox mapping
                bx1 = float(np.clip((b[0] - pad_l) / r, 0, w0))
                by1 = float(np.clip((b[1] - pad_t) / r, 0, h0))
                bx2 = float(np.clip((b[2] - pad_l) / r, 0, w0))
                by2 = float(np.clip((b[3] - pad_t) / r, 0, h0))
                detections.append({
                    "box": [bx1, by1, bx2, by2],
                    "confidence": score,
                    "label": "plate"
                })
        else:
            t_infer = time.perf_counter()
            results = self.model.predict(im0, imgsz=self.imgsz, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
            infer_ms = (time.perf_counter() - t_infer) * 1000.0
            detections = []
            for b in results[0].boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                detections.append({
                    "box": xyxy,
                    "confidence": conf,
                    "label": "plate"
                })

        total_ms = (time.perf_counter() - t0) * 1000.0
        return detections, im0, infer_ms, total_ms

def main():
    parser = argparse.ArgumentParser(description="Standalone Vietnamese License Plate Detector")
    parser.add_argument("--weights", type=str, default="vn-plate-app/app/src/main/assets/best.onnx", help="Path to .onnx or .pt file")
    parser.add_argument("--source", type=str, required=True, help="Input image path")
    parser.add_argument("--imgsz", type=int, default=224, help="Input resolution (224 for edge nano, 640 for server large)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--output", type=str, default="output_pred.jpg", help="Path to save annotated prediction")
    args = parser.parse_args()

    detector = Detector(args.weights, imgsz=args.imgsz, conf_thresh=args.conf, iou_thresh=args.iou)
    detections, im0, infer_ms, total_ms = detector.predict(args.source)

    print(f"\n--- Inference Summary ---")
    print(f"Weights : {args.weights}")
    print(f"Source  : {args.source}")
    print(f"Input   : {args.imgsz}x{args.imgsz}")
    print(f"Latency : Infer={infer_ms:.2f}ms | Total={total_ms:.2f}ms")
    print(f"Found   : {len(detections)} plate(s)")

    for i, d in enumerate(detections):
        box = [round(x, 1) for x in d["box"]]
        print(f"  [{i+1}] Conf: {d['confidence']*100:.1f}% | Box: {box}")

        # Draw on image
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 230, 64), 3)
        label_text = f"plate {d['confidence']*100:.0f}%"
        cv2.putText(im0, label_text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 64), 2)

    cv2.imwrite(args.output, im0)
    print(f"Annotated output saved to: {args.output}\n")

if __name__ == "__main__":
    main()
