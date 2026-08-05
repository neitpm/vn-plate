"""
Script huan luyen YOLOv8 nhan dien bien so xe Viet Nam
Dataset: datasets/Vietnam license-plate.v1i.yolo26

Cau hinh mac dinh toi uu cho:
  - GPU  : NVIDIA GeForce RTX ~16 GB VRAM
  - RAM  : 32 GB
  - Disk : SSD NVMe

Clone va chay:
  git clone https://github.com/kennhope13/vn-plate.git
  cd vn-plate
  pip install ultralytics
  python train_license_plate.py
"""

import sys
from pathlib import Path

# ─────────────────────────────────────────────
# THƯ MỤC GỐC = nơi chứa script này
# Hoạt động đúng dù clone về bất kỳ đâu
# ─────────────────────────────────────────────
ROOT        = Path(__file__).parent.resolve()
DATASET_DIR = ROOT / "datasets" / "Vietnam license-plate.v1i.yolo26"
DATA_YAML   = DATASET_DIR / "data.yaml"

# ─────────────────────────────────────────────
# CẤU HÌNH HUẤN LUYỆN
# ─────────────────────────────────────────────
MODEL       = "yolov8l.pt"   # đổi sang yolov8x.pt nếu muốn độ chính xác tối đa
EPOCHS      = 200
IMG_SIZE    = 640            # 640 ổn định hơn 1280 với dataset ~700 ảnh;
                             # tăng lên 1280 sau khi model đã converge tốt
BATCH_SIZE  = 16             # batch nhỏ hơn → gradient ổn định hơn với dataset nhỏ
WORKERS     = 8
DEVICE      = 0
PROJECT     = str(ROOT / "runs" / "train")
RUN_NAME    = "vn_plate_yolov8l_stable"

PATIENCE    = 50             # cho model đủ thời gian recover
LR0         = 0.001          # learning rate thấp hơn nhiều – quan trọng nhất
LRF         = 0.01           # decay đến LR0 * LRF = 1e-5
OPTIMIZER   = "AdamW"
AMP         = True


# ─────────────────────────────────────────────
# KIỂM TRA MÔI TRƯỜNG
# ─────────────────────────────────────────────
def check_env():
    try:
        import ultralytics
        print(f"[OK] ultralytics {ultralytics.__version__}")
    except ImportError:
        print("[ERROR] Chua cai ultralytics. Chay: pip install ultralytics")
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
            if isinstance(DEVICE, int) and DEVICE >= n_gpu:
                print(f"[ERROR] DEVICE={DEVICE} nhung chi co {n_gpu} GPU (index 0~{n_gpu-1})")
                sys.exit(1)
        else:
            print("[WARN] Khong tim thay GPU – train tren CPU se rat cham")
    except ImportError:
        print("[ERROR] Chua cai torch. Xem: https://pytorch.org/get-started/locally/")
        sys.exit(1)

    if not DATASET_DIR.exists():
        print(f"[ERROR] Khong tim thay dataset tai: {DATASET_DIR}")
        sys.exit(1)

    print(f"[OK] Root     : {ROOT}")
    print(f"[OK] Dataset  : {DATASET_DIR}")
    print(f"[OK] Output   : {PROJECT}/{RUN_NAME}")


# ─────────────────────────────────────────────
# TẠO data_train.yaml VỚI PATH TUYỆT ĐỐI
# (tính từ ROOT lúc runtime, không hardcode)
# ─────────────────────────────────────────────
def make_data_yaml() -> Path:
    """
    Tạo data_train.yaml với path tuyệt đối tính từ vị trí clone thực tế.
    File này không được commit vào git (nằm trong .gitignore).
    """
    import yaml

    cfg = {
        "train": (DATASET_DIR / "train" / "images").as_posix(),
        "val"  : (DATASET_DIR / "valid" / "images").as_posix(),
        "test" : (DATASET_DIR / "test"  / "images").as_posix(),
        "nc"   : 1,
        "names": ["plate"],
    }

    out = ROOT / "data_train.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[OK] data_train.yaml:")
    print(f"     train : {cfg['train']}")
    print(f"     val   : {cfg['val']}")
    print(f"     test  : {cfg['test']}")
    return out


# ─────────────────────────────────────────────
# HUẤN LUYỆN
# ─────────────────────────────────────────────
def train(data_yaml: Path, use_cache: bool = True):
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  BAT DAU HUAN LUYEN YOLOv8 – BIEN SO XE VIET NAM")
    print("=" * 60)
    print(f"  Model      : {MODEL}")
    print(f"  Epochs     : {EPOCHS}  (patience={PATIENCE})")
    print(f"  Batch      : {BATCH_SIZE}")
    print(f"  Image size : {IMG_SIZE}")
    print(f"  Device     : {DEVICE}")
    print(f"  Output     : {PROJECT}/{RUN_NAME}")
    print("=" * 60 + "\n")

    model = YOLO(MODEL)

    model.train(
        data           = str(data_yaml),
        epochs         = EPOCHS,
        imgsz          = IMG_SIZE,
        batch          = BATCH_SIZE,
        workers        = WORKERS,
        device         = DEVICE,
        project        = PROJECT,
        name           = RUN_NAME,
        patience       = PATIENCE,
        lr0            = LR0,
        lrf            = LRF,
        optimizer      = OPTIMIZER,
        amp            = AMP,
        # ── Augmentation – giảm mạnh so với trước ──
        # Màu sắc nhẹ thôi, biển số VN màu khá chuẩn
        hsv_h          = 0.01,
        hsv_s          = 0.3,
        hsv_v          = 0.3,
        # Hình học – biển số gắn cố định, không xoay nhiều
        degrees        = 5.0,
        translate      = 0.05,
        scale          = 0.4,
        shear          = 1.0,
        perspective    = 0.0,
        flipud         = 0.0,   # bien so khong bao gio lon nguoc
        fliplr         = 0.5,
        # Mosaic giữ nhưng tắt mixup và copy_paste
        # – dataset nhỏ, 2 kỹ thuật này cùng lúc gây nhiễu quá mức
        mosaic         = 1.0,
        mixup          = 0.0,   # tắt
        copy_paste     = 0.0,   # tắt
        # ── Checkpoint ──
        save           = True,
        save_period    = 10,
        exist_ok       = True,
        verbose        = True,
        # ── LR schedule ──
        warmup_epochs  = 5,     # warmup dài hơn để gradient ổn định từ đầu
        warmup_bias_lr = 0.01,
        cos_lr         = True,
        # ── Weight decay chặt hơn để tránh overfit ──
        weight_decay   = 0.001,
        # ── Cache ──
        cache          = "ram" if use_cache else False,
    )


# ─────────────────────────────────────────────
# ĐÁNH GIÁ TRÊN TẬP TEST
# ─────────────────────────────────────────────
def evaluate(data_yaml: Path):
    from ultralytics import YOLO

    best_weight = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if not best_weight.exists():
        print(f"[WARN] Khong tim thay best.pt tai {best_weight}, bo qua evaluate.")
        return

    print("\n" + "=" * 60)
    print("  DANH GIA MODEL TREN TAP TEST")
    print("=" * 60)

    model   = YOLO(str(best_weight))
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

    print(f"\n  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")


# ─────────────────────────────────────────────
# EXPORT MODEL
# ─────────────────────────────────────────────
def export_model(fmt: str = "onnx"):
    """Export sang ONNX / TensorRT / CoreML / TFLite ..."""
    from ultralytics import YOLO

    best_weight = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if not best_weight.exists():
        print(f"[WARN] Khong tim thay best.pt, bo qua export.")
        return

    print(f"\n[INFO] Exporting sang: {fmt}")
    model = YOLO(str(best_weight))
    model.export(format=fmt, imgsz=IMG_SIZE, dynamic=True, simplify=True)
    print("[OK] Export hoan tat.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 nhan dien bien so xe VN")
    parser.add_argument("--model",    default=MODEL,           help="vd: yolov8l.pt, yolov8x.pt")
    parser.add_argument("--epochs",   default=EPOCHS,          type=int)
    parser.add_argument("--batch",    default=BATCH_SIZE,      type=int)
    parser.add_argument("--imgsz",    default=IMG_SIZE,        type=int)
    parser.add_argument("--device",   default=str(DEVICE),     help="0 / 1 / cpu")
    parser.add_argument("--no-eval",  action="store_true",     help="Bo qua evaluate sau train")
    parser.add_argument("--export",   default=None,            help="Export format: onnx / engine / tflite")
    parser.add_argument("--no-cache", action="store_true",     help="Tat cache RAM")
    args = parser.parse_args()

    MODEL      = args.model
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch
    IMG_SIZE   = args.imgsz
    DEVICE     = int(args.device) if args.device.isdigit() else args.device

    check_env()
    data_yaml = make_data_yaml()
    train(data_yaml, use_cache=not args.no_cache)

    if not args.no_eval:
        evaluate(data_yaml)

    if args.export:
        export_model(args.export)

    print("\n[DONE] Xong! Ket qua luu tai:", Path(PROJECT) / RUN_NAME)
