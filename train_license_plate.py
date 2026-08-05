"""
Script huấn luyện YOLOv8 nhan dien bien so xe Viet Nam
Dataset: D:/titsmat-parking/datasets/Vietnam license-plate.v1i.yolo26

Cau hinh toi uu cho:
  - GPU  : NVIDIA GeForce RTX (GPU 1) - 15.9 GB VRAM
  - RAM  : 32 GB
  - Disk : SSD NVMe
"""

import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# CẤU HÌNH HUẤN LUYỆN
# Tối ưu cho NVIDIA RTX 15.9 GB VRAM / RAM 32 GB
# ─────────────────────────────────────────────
DATASET_DIR = Path(r"D:\titsmat-parking\datasets\Vietnam license-plate.v1i.yolo26")
DATA_YAML   = DATASET_DIR / "data.yaml"

# yolov8l: cân bằng tốt giữa tốc độ và độ chính xác với VRAM 16 GB
# Đổi sang yolov8x nếu muốn độ chính xác tối đa (chậm hơn ~30%)
MODEL       = "yolov8l.pt"
EPOCHS      = 150
IMG_SIZE    = 1280           # ảnh biển số cần độ phân giải cao để nhận chữ tốt
BATCH_SIZE  = 32             # 15.9 GB VRAM chạy thoải mái batch 32 với img 1280
WORKERS     = 8              # SSD NVMe + RAM 32 GB → tăng workers để data load nhanh
DEVICE      = 0              # RTX 5060 Ti là GPU duy nhất (index 0)
PROJECT     = r"D:\titsmat-parking\runs\train"
RUN_NAME    = "vn_plate_yolov8l_1280"

PATIENCE    = 30             # early stopping – tăng lên vì train lâu hơn
LR0         = 0.01           # learning rate ban đầu
LRF         = 0.005          # cosine decay về 5e-5 cuối quá trình train
OPTIMIZER   = "AdamW"        # AdamW hội tụ ổn định hơn SGD với dataset nhỏ
AMP         = True           # Automatic Mixed Precision – giữ FP16, tận dụng Tensor Core

# ─────────────────────────────────────────────
# KIỂM TRA MÔI TRƯỜNG
# ─────────────────────────────────────────────
def check_env():
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"[OK] ultralytics {ultralytics.__version__}")
    except ImportError:
        print("[ERROR] Chưa cài ultralytics. Chạy: pip install ultralytics")
        sys.exit(1)

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"[OK] PyTorch {torch.__version__} | CUDA: {cuda_ok}")
        if cuda_ok:
            n_gpu = torch.cuda.device_count()
            for i in range(n_gpu):
                vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"     GPU {i}: {torch.cuda.get_device_name(i)} ({vram:.1f} GB VRAM)")
            # Cảnh báo nếu chọn sai device
            if isinstance(DEVICE, int) and DEVICE >= n_gpu:
                print(f"[ERROR] DEVICE={DEVICE} nhưng chỉ có {n_gpu} GPU (index 0~{n_gpu-1})")
                sys.exit(1)
        else:
            print("[WARN] Không tìm thấy GPU – sẽ train trên CPU (chậm hơn nhiều)")
    except ImportError:
        print("[ERROR] Chưa cài torch. Xem: https://pytorch.org/get-started/locally/")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"[ERROR] Không tìm thấy {DATA_YAML}")
        sys.exit(1)
    print(f"[OK] Dataset: {DATA_YAML}")


# ─────────────────────────────────────────────
# FIX DATA.YAML – đổi path tương đối → tuyệt đối
# ─────────────────────────────────────────────
def fix_data_yaml() -> Path:
    """
    data.yaml gốc dùng '../train/images' nhưng thực tế train/valid/test
    nằm ngay bên trong DATASET_DIR. Hàm này tạo data_abs.yaml với path
    tuyệt đối chính xác.
    """
    import yaml

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # train/valid/test thực tế nằm trong DATASET_DIR
    cfg["train"] = (DATASET_DIR / "train"  / "images").as_posix()
    cfg["val"]   = (DATASET_DIR / "valid"  / "images").as_posix()
    cfg["test"]  = (DATASET_DIR / "test"   / "images").as_posix()

    out_yaml = DATASET_DIR / "data_abs.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[OK] Đã tạo data_abs.yaml:")
    print(f"     train: {cfg['train']}")
    print(f"     val  : {cfg['val']}")
    print(f"     test : {cfg['test']}")
    return out_yaml


# ─────────────────────────────────────────────
# HUẤN LUYỆN
# ─────────────────────────────────────────────
def train(data_yaml: Path, use_cache: bool = True):
    from ultralytics import YOLO

    print("\n" + "="*60)
    print("  BẮT ĐẦU HUẤN LUYỆN YOLOv8 – BIỂN SỐ XE VIỆT NAM")
    print("="*60)
    print(f"  Model      : {MODEL}")
    print(f"  Epochs     : {EPOCHS}  (patience={PATIENCE})")
    print(f"  Batch      : {BATCH_SIZE}")
    print(f"  Image size : {IMG_SIZE}")
    print(f"  Device     : {DEVICE}")
    print(f"  Output dir : {PROJECT}/{RUN_NAME}")
    print("="*60 + "\n")

    model = YOLO(MODEL)

    results = model.train(
        data        = str(data_yaml),
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        workers     = WORKERS,
        device      = DEVICE,
        project     = PROJECT,
        name        = RUN_NAME,
        patience    = PATIENCE,
        lr0         = LR0,
        lrf         = LRF,
        optimizer   = OPTIMIZER,
        amp         = AMP,
        # ── Data augmentation tối ưu cho biển số xe ──
        # Màu sắc – biển số VN có màu nền trắng/vàng/xanh
        hsv_h       = 0.015,
        hsv_s       = 0.5,
        hsv_v       = 0.4,
        # Hình học – biển số thường nằm ngang, ít xoay
        degrees     = 8.0,    # xoay nhỏ mô phỏng camera lệch góc
        translate   = 0.1,
        scale       = 0.6,    # zoom in/out mô phỏng khoảng cách xe
        shear       = 3.0,
        perspective = 0.0005, # mô phỏng góc camera hơi nghiêng
        flipud      = 0.0,    # không lật dọc – biển số không bao giờ lộn ngược
        fliplr      = 0.5,
        mosaic      = 1.0,    # mosaic tốt với dataset nhỏ (~700 ảnh)
        mixup       = 0.1,    # mixup nhẹ để tăng tổng quát hoá
        copy_paste  = 0.1,    # copy-paste giả lập nhiều biển số trong 1 ảnh
        # ── Lưu checkpoint ──
        save        = True,
        save_period = 10,
        exist_ok    = True,
        verbose     = True,
        # ── Warmup & scheduler ──
        warmup_epochs  = 3,
        warmup_bias_lr = 0.1,
        cos_lr         = True,   # cosine learning rate schedule
        # ── Cache ảnh vào RAM (32 GB đủ dư) → tăng tốc I/O ──
        cache       = "ram" if use_cache else False,
    )

    return results


# ─────────────────────────────────────────────
# ĐÁNH GIÁ TRÊN TẬP TEST
# ─────────────────────────────────────────────
def evaluate(data_yaml: Path):
    from ultralytics import YOLO

    best_weight = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if not best_weight.exists():
        print(f"[WARN] Không tìm thấy best.pt tại {best_weight}, bỏ qua bước evaluate.")
        return

    print("\n" + "="*60)
    print("  ĐÁNH GIÁ MODEL TRÊN TẬP TEST")
    print("="*60)

    model = YOLO(str(best_weight))
    metrics = model.val(
        data    = str(data_yaml),
        split   = "test",
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = DEVICE,
        project = PROJECT,
        name    = RUN_NAME + "_test",
        verbose = True,
    )

    print(f"\n  mAP50   : {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")


# ─────────────────────────────────────────────
# EXPORT MODEL (tuỳ chọn)
# ─────────────────────────────────────────────
def export_model(fmt: str = "onnx"):
    """
    Export sang ONNX, TensorRT, CoreML, ... để deploy.
    fmt: 'onnx' | 'engine' (TensorRT) | 'coreml' | 'tflite' ...
    """
    from ultralytics import YOLO

    best_weight = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if not best_weight.exists():
        print(f"[WARN] Không tìm thấy best.pt, bỏ qua export.")
        return

    print(f"\n[INFO] Exporting model sang định dạng: {fmt}")
    model = YOLO(str(best_weight))
    model.export(format=fmt, imgsz=IMG_SIZE, dynamic=True, simplify=True)
    print("[OK] Export hoàn tất.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 nhận diện biển số xe VN")
    parser.add_argument("--model",   default=MODEL,      help="Pretrained weight (vd: yolov8l.pt, yolov8x.pt)")
    parser.add_argument("--epochs",  default=EPOCHS,     type=int)
    parser.add_argument("--batch",   default=BATCH_SIZE, type=int)
    parser.add_argument("--imgsz",   default=IMG_SIZE,   type=int)
    parser.add_argument("--device",  default=str(DEVICE))
    parser.add_argument("--no-eval", action="store_true", help="Bỏ qua bước evaluate sau train")
    parser.add_argument("--export",  default=None,        help="Export model sau train (vd: onnx)")
    parser.add_argument("--no-cache",action="store_true", help="Tắt cache RAM (nếu RAM không đủ)")
    args = parser.parse_args()

    # Ghi đè config từ CLI
    MODEL       = args.model
    EPOCHS      = args.epochs
    BATCH_SIZE  = args.batch
    IMG_SIZE    = args.imgsz
    DEVICE      = int(args.device) if args.device.isdigit() else args.device

    check_env()
    data_yaml = fix_data_yaml()
    train(data_yaml, use_cache=not args.no_cache)

    if not args.no_eval:
        evaluate(data_yaml)

    if args.export:
        export_model(args.export)

    print("\n[DONE] Xong! Kết quả lưu tại:", Path(PROJECT) / RUN_NAME)
