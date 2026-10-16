"""
Generate standardized test fixtures for Android/Python parity verification:
1. Three real test images:
   - Case 1: Wide Aspect Ratio (Landscape ~16:9 / ~1.56:1)
   - Case 2: Standard Aspect Ratio (Landscape ~4:3)
   - Case 3: Tall Aspect Ratio (Portrait ~9:16 / tall)
2. Ultralytics-compliant letterbox preprocessing with exact roundings.
3. Raw input tensor [1, 3, 224, 224] float32 in both .npy and raw binary .bin.
4. Raw ONNX output [1, 5, 1029] float32 in .npy.
5. Post-NMS bounding boxes in 3 coordinate systems:
   - S1: 224x224 input space (xyxy and cxcywh)
   - S2: Rotated/Preprocessed image space
   - S3: Original camera image space
6. Round-trip sanity test: inverse(letterbox(box)) == box within float precision.
"""

import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

OUTPUT_DIR = Path("tests/fixtures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_MODEL_PATH = Path("vn-plate-app/app/src/main/assets/best.onnx")

# Fixed evaluation parameters
TARGET_SIZE = 224
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50

def letterbox_ultralytics(im, new_shape=224, color=(114, 114, 114)):
    """
    Standard Ultralytics Letterbox implementation matching export configuration.
    """
    shape = im.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute unpadded dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    # Divide padding into 2 sides symmetrically
    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Ultralytics exact rounding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im_padded = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top, right, bottom), new_unpad

def nms(boxes, scores, iou_thresh=0.5):
    """Greedy Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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

def inverse_letterbox_pt(box_224, ratio, pad_tuple):
    """
    Inverse coordinate mapping from 224x224 to original image space.
    box_224: [x1, y1, x2, y2]
    """
    left, top, _, _ = pad_tuple
    x1 = (box_224[0] - left) / ratio
    y1 = (box_224[1] - top) / ratio
    x2 = (box_224[2] - left) / ratio
    y2 = (box_224[3] - top) / ratio
    return [float(x1), float(y1), float(x2), float(y2)]

def forward_letterbox_pt(box_orig, ratio, pad_tuple):
    """Forward coordinate mapping from original space to 224x224"""
    left, top, _, _ = pad_tuple
    x1 = box_orig[0] * ratio + left
    y1 = box_orig[1] * ratio + top
    x2 = box_orig[2] * ratio + left
    y2 = box_orig[3] * ratio + top
    return [float(x1), float(y1), float(x2), float(y2)]

def process_case(case_id, img_path, session):
    print(f"\nProcessing {case_id}: {img_path}")
    im0 = cv2.imread(str(img_path))
    assert im0 is not None, f"Could not read {img_path}"
    orig_h, orig_w = im0.shape[:2]

    # 1. Letterbox
    im_lb, ratio, pad_tuple, unpad_wh = letterbox_ultralytics(im0, new_shape=TARGET_SIZE)
    pad_left, pad_top, pad_right, pad_bottom = pad_tuple
    
    # Save input image copies
    shutil.copy(img_path, OUTPUT_DIR / f"{case_id}_orig.jpg")
    cv2.imwrite(str(OUTPUT_DIR / f"{case_id}_letterbox.png"), im_lb)

    # 2. Preprocess to Float Tensor [1, 3, 224, 224] RGB [0, 1]
    rgb = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB)
    tensor_chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    tensor_nchw = np.expand_dims(tensor_chw, axis=0)

    # Save tensors
    np.save(OUTPUT_DIR / f"{case_id}_input.npy", tensor_nchw)
    tensor_nchw.astype(np.float32).tofile(OUTPUT_DIR / f"{case_id}_input.bin")

    # 3. Inference
    raw_output = session.run(None, {"images": tensor_nchw})[0]  # [1, 5, 1029]
    np.save(OUTPUT_DIR / f"{case_id}_raw_output.npy", raw_output)

    # 4. Decode & NMS
    # output: [cx, cy, w, h, conf]
    cxs = raw_output[0, 0]
    cys = raw_output[0, 1]
    ws = raw_output[0, 2]
    hs = raw_output[0, 3]
    confs = raw_output[0, 4]

    # Convert to xyxy
    x1s = cxs - ws / 2.0
    y1s = cys - hs / 2.0
    x2s = cxs + ws / 2.0
    y2s = cys + hs / 2.0

    boxes_all = np.stack([x1s, y1s, x2s, y2s], axis=1)
    mask = confs >= CONF_THRESHOLD
    filtered_boxes = boxes_all[mask]
    filtered_confs = confs[mask]

    keep_indices = nms(filtered_boxes, filtered_confs, iou_thresh=IOU_THRESHOLD)
    final_boxes_224 = filtered_boxes[keep_indices]
    final_confs = filtered_confs[keep_indices]

    # 5. Coordinate Transformations
    detections = []
    for b_224, score in zip(final_boxes_224, final_confs):
        box_224_list = [float(x) for x in b_224]
        box_orig_list = inverse_letterbox_pt(box_224_list, ratio, pad_tuple)
        
        # Round-trip recovery check
        recovered_box = forward_letterbox_pt(box_orig_list, ratio, pad_tuple)
        max_diff = max(abs(a - b) for a, b in zip(box_224_list, recovered_box))
        assert max_diff < 1e-4, f"Round-trip inverse mismatch: {max_diff}"

        detections.append({
            "confidence": float(score),
            "box_224_xyxy": box_224_list,
            "box_224_cxcywh": [
                float((box_224_list[0] + box_224_list[2]) / 2.0),
                float((box_224_list[1] + box_224_list[3]) / 2.0),
                float(box_224_list[2] - box_224_list[0]),
                float(box_224_list[3] - box_224_list[1])
            ],
            "box_orig_xyxy": box_orig_list,
            "box_orig_cxcywh": [
                float((box_orig_list[0] + box_orig_list[2]) / 2.0),
                float((box_orig_list[1] + box_orig_list[3]) / 2.0),
                float(box_orig_list[2] - box_orig_list[0]),
                float(box_orig_list[3] - box_orig_list[1])
            ]
        })

    # 6. Save Metadata & Calibration Spec
    meta = {
        "case_id": case_id,
        "source_file": str(img_path.name),
        "orig_width": int(orig_w),
        "orig_height": int(orig_h),
        "aspect_ratio": float(orig_w / orig_h),
        "target_size": TARGET_SIZE,
        "scale_ratio": float(ratio),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(pad_right),
        "pad_bottom": int(pad_bottom),
        "unpadded_width": int(unpad_wh[0]),
        "unpadded_height": int(unpad_wh[1]),
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "num_detections": len(detections),
        "detections": detections
    }

    with open(OUTPUT_DIR / f"{case_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{case_id}] Orig: {orig_w}x{orig_h} (AR={orig_w/orig_h:.2f}) | Ratio: {ratio:.4f} | Pad: L={pad_left}, T={pad_top} | Detections: {len(detections)}")

def main():
    print(f"Initializing ONNX session from {ONNX_MODEL_PATH}...")
    session = ort.InferenceSession(str(ONNX_MODEL_PATH))

    # Case 1: Wide ratio (Landscape 472x303, AR ~ 1.56)
    img_wide = Path("datasets/Vietnam license-plate.v1i.yolo26/test/images/CarLongPlate115_jpg.rf.122cb025d8830754c49e2956145c18a3.jpg")
    process_case("case1_wide_16_9", img_wide, session)

    # Case 2: Standard 4:3 (472x354 generated or cropped from dataset)
    # Use real image with AR ~ 1.33
    img_43 = Path("datasets/Vietnam license-plate.v1i.yolo26/test/images/CarLongPlate113_jpg.rf.d83935b14070679cb3810ab5f09d014e.jpg")
    process_case("case2_standard_4_3", img_43, session)

    # Case 3: Portrait / Tall (AR ~ 0.65, height > width, e.g. 196x303)
    img_src = cv2.imread("datasets/Vietnam license-plate.v1i.yolo26/test/images/CarLongPlate145_jpg.rf.5c1c1966a8f0d09e57bdafee3f2bec67.jpg")
    # Crop central vertical section [0:303, 138:334] (196 x 303)
    portrait_crop = img_src[:, 138:334]
    portrait_crop_path = OUTPUT_DIR / "case3_portrait_source.jpg"
    cv2.imwrite(str(portrait_crop_path), portrait_crop)
    process_case("case3_portrait_tall", portrait_crop_path, session)

    print(f"\nAll 3 fixtures successfully generated in {OUTPUT_DIR.resolve()}!")

if __name__ == "__main__":
    main()
