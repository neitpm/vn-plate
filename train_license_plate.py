"""
Script huan luyen YOLOv8 nhan dien bien so xe Viet Nam
Dataset: datasets/Vietnam license-plate.v1i.yolo26

Chien luoc: "cham ma chac"
  - Giai doan 1 (freeze): Chi train head, giu nguyen backbone pretrained
                          → model hoc nhanh biet bien so la gi
  - Giai doan 2 (unfreeze): Mo toan bo, fine-tune nhe voi lr rat nho
                            → tinh chinh tung chi tiet

Clone va chay:
  git clone https://github.com/kennhope13/vn-plate.git
  cd vn-plate
  pip install ultralytics
  python train_license_plate.py
"""

import sys
from pathlib import Path

# ─────────────────────────────────────────────
# THƯ MỤC GỐC
# ─────────────────────────────────────────────
ROOT        = Path(__file__).parent.resolve()
DATASET_DIR = ROOT / "datasets" / "Vietnam license-plate.v1i.yolo26"
DATA_YAML   = DATASET_DIR / "data.yaml"

# ─────────────────────────────────────────────
# CẤU HÌNH CHUNG
# ─────────────────────────────────────────────
MODEL      = "yolov8l.pt"
IMG_SIZE   = 640
BATCH_SIZE = 16
WORKERS    = 8
DEVICE     = 0
PROJECT    = str(ROOT / "runs" / "train")
AMP        = True

# ── Giai doan 1: Freeze backbone, chi train detection head ──
# Model giu nguyen kien thuc pretrained, chi hoc "bien so o cho nay"
STAGE1_NAME          = "vn_plate_stage1_freeze"
STAGE1_EPOCHS        = 30       # it epoch vi chi train head, converge rat nhanh
STAGE1_LR0           = 0.001    # lr vua phai cho head
STAGE1_LRF           = 0.1      # decay xuong 1e-4
STAGE1_WARMUP_EPOCHS = 3
STAGE1_FREEZE        = 10       # dong bang 10 layer dau (backbone) cua yolov8l

# ── Giai doan 2: Unfreeze toan bo, fine-tune nhe ──
# Mo backbone ra, chinh sua tung chi tiet voi lr rat nho
STAGE2_NAME          = "vn_plate_stage2_finetune"
STAGE2_EPOCHS        = 300      # nhiều hơn để box regression hội tụ
STAGE2_LR0           = 0.0001
STAGE2_LRF           = 0.01
STAGE2_WARMUP_EPOCHS = 10
STAGE2_PATIENCE      = 80       # kiên nhẫn hơn vì mAP50-95 cải thiện chậm


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
            print("[WARN] Khong co GPU – train tren CPU rat cham")
    except ImportError:
        print("[ERROR] Chua cai torch. Xem: https://pytorch.org/get-started/locally/")
        sys.exit(1)

    if not DATASET_DIR.exists():
        print(f"[ERROR] Khong tim thay dataset: {DATASET_DIR}")
        sys.exit(1)

    print(f"[OK] Root    : {ROOT}")
    print(f"[OK] Dataset : {DATASET_DIR}")
    print(f"[OK] Output  : {PROJECT}")


# ─────────────────────────────────────────────
# TẠO data_train.yaml (path tuyệt đối runtime)
# ─────────────────────────────────────────────
def make_data_yaml() -> Path:
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
# AUGMENTATION NHẸ – dùng chung cả 2 giai đoạn
# ─────────────────────────────────────────────
AUGMENT_ARGS = dict(
    # Màu sắc nhẹ
    hsv_h        = 0.01,
    hsv_s        = 0.2,
    hsv_v        = 0.2,
    # Hình học
    degrees      = 3.0,
    translate    = 0.05,
    scale        = 0.3,
    shear        = 0.5,
    perspective  = 0.0,
    flipud       = 0.0,
    fliplr       = 0.5,
    mosaic       = 0.5,
    mixup        = 0.0,
    copy_paste   = 0.0,
    close_mosaic = 20,
)


# ─────────────────────────────────────────────
# GIAI ĐOẠN 1: FREEZE BACKBONE
# ─────────────────────────────────────────────
def train_stage1(data_yaml: Path, cache_mode: str) -> Path:
    """
    Chi train detection head, backbone bi dong bang.
    Muc tieu: model nhanh chong hoc biet 'bien so xe o cho nao'
    ma khong pha vo kien thuc pretrained.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  GIAI DOAN 1/2 – FREEZE BACKBONE (chi train head)")
    print(f"  Epochs : {STAGE1_EPOCHS} | LR : {STAGE1_LR0} | Freeze : {STAGE1_FREEZE} layers")
    print("=" * 60 + "\n")

    model = YOLO(MODEL)

    model.train(
        data           = str(data_yaml),
        epochs         = STAGE1_EPOCHS,
        imgsz          = IMG_SIZE,
        batch          = BATCH_SIZE,
        workers        = WORKERS,
        device         = DEVICE,
        project        = PROJECT,
        name           = STAGE1_NAME,
        patience       = 15,
        lr0            = STAGE1_LR0,
        lrf            = STAGE1_LRF,
        optimizer      = "SGD",           # SGD + momentum on dinh hon AdamW khi freeze
        momentum       = 0.937,
        amp            = AMP,
        freeze         = STAGE1_FREEZE,
        warmup_epochs  = STAGE1_WARMUP_EPOCHS,
        warmup_momentum= 0.8,             # momentum tang dan trong warmup
        warmup_bias_lr = 0.001,
        cos_lr         = True,
        weight_decay   = 0.0005,
        dropout        = 0.1,             # dropout nhe tranh overfit dataset nho
        save           = True,
        save_period    = 5,
        exist_ok       = True,
        verbose        = True,
        cache          = cache_mode,
        **AUGMENT_ARGS,
    )

    best = Path(PROJECT) / STAGE1_NAME / "weights" / "best.pt"
    print(f"\n[OK] Giai doan 1 hoan tat. Best weight: {best}")
    return best


# ─────────────────────────────────────────────
# GIAI ĐOẠN 2: UNFREEZE – FINE-TUNE TOÀN BỘ
# ─────────────────────────────────────────────
def train_stage2(data_yaml: Path, stage1_weight: Path, cache_mode: str) -> Path:
    """
    Mo toan bo model, fine-tune voi lr rat nho.
    Muc tieu: tinh chinh tung chi tiet, tang do chinh xac len cao nhat.
    """
    from ultralytics import YOLO

    if not stage1_weight.exists():
        print(f"[ERROR] Khong tim thay weight stage 1: {stage1_weight}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  GIAI DOAN 2/2 – FINE-TUNE TOAN BO MODEL")
    print(f"  Epochs : {STAGE2_EPOCHS} | LR : {STAGE2_LR0} | Warmup : {STAGE2_WARMUP_EPOCHS}")
    print(f"  Weight : {stage1_weight}")
    print("=" * 60 + "\n")

    # Load tu best.pt cua stage 1 – giu nguyen tat ca da hoc
    model = YOLO(str(stage1_weight))

    model.train(
        data           = str(data_yaml),
        epochs         = STAGE2_EPOCHS,
        imgsz          = 1280,            # tang len 1280 – box khit hon tren anh lon
        batch          = 8,               # giam batch vi img 1280 nang hon
        workers        = WORKERS,
        device         = DEVICE,
        project        = PROJECT,
        name           = STAGE2_NAME,
        patience       = STAGE2_PATIENCE,
        lr0            = STAGE2_LR0,
        lrf            = STAGE2_LRF,
        optimizer      = "AdamW",
        momentum       = 0.937,
        amp            = AMP,
        freeze         = 0,
        warmup_epochs  = STAGE2_WARMUP_EPOCHS,
        warmup_momentum= 0.8,
        warmup_bias_lr = 0.0001,
        cos_lr         = True,
        weight_decay   = 0.0005,
        dropout        = 0.0,
        # Tang box loss weight de bbox khit hon → mAP50-95 cao hon
        box            = 10.0,            # mac dinh 7.5, tang len 10
        cls            = 0.3,             # giam cls vi chi 1 class
        dfl            = 1.5,
        save           = True,
        save_period    = 10,
        exist_ok       = True,
        verbose        = True,
        cache          = cache_mode,
        **AUGMENT_ARGS,
    )

    best = Path(PROJECT) / STAGE2_NAME / "weights" / "best.pt"
    print(f"\n[OK] Giai doan 2 hoan tat. Best weight: {best}")
    return best


# ─────────────────────────────────────────────
# ĐÁNH GIÁ TRÊN TẬP TEST
# ─────────────────────────────────────────────
def evaluate(data_yaml: Path, weight: Path, run_name: str):
    from ultralytics import YOLO

    if not weight.exists():
        print(f"[WARN] Khong tim thay {weight}, bo qua evaluate.")
        return

    print("\n" + "=" * 60)
    print("  DANH GIA MODEL TREN TAP TEST")
    print("=" * 60)

    model   = YOLO(str(weight))
    metrics = model.val(
        data    = str(data_yaml),
        split   = "test",
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = DEVICE,
        project = PROJECT,
        name    = run_name + "_test",
        verbose = True,
    )

    print(f"\n  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")


# ─────────────────────────────────────────────
# EXPORT MODEL
# ─────────────────────────────────────────────
def export_model(weight: Path, fmt: str = "onnx"):
    from ultralytics import YOLO

    if not weight.exists():
        print(f"[WARN] Khong tim thay {weight}, bo qua export.")
        return

    print(f"\n[INFO] Exporting sang: {fmt}")
    model = YOLO(str(weight))
    model.export(format=fmt, imgsz=IMG_SIZE, dynamic=True, simplify=True)
    print("[OK] Export hoan tat.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train YOLOv8 bien so xe VN – 2 giai doan: freeze → fine-tune"
    )
    parser.add_argument("--model",       default=MODEL,       help="vd: yolov8l.pt, yolov8x.pt")
    parser.add_argument("--batch",       default=BATCH_SIZE,  type=int)
    parser.add_argument("--imgsz",       default=IMG_SIZE,    type=int)
    parser.add_argument("--device",      default=str(DEVICE), help="0 / 1 / cpu")
    parser.add_argument("--no-eval",     action="store_true", help="Bo qua evaluate")
    parser.add_argument("--export",      default=None,        help="onnx / engine / tflite")
    parser.add_argument("--cache-disk",  action="store_true", help="Dung cache disk (cho Colab / it RAM)")
    parser.add_argument("--stage",       default="all",       choices=["all", "1", "2"],
                        help="Chay giai doan nao: all=ca hai, 1=chi freeze, 2=chi finetune")
    parser.add_argument("--stage2-weight", default=None,
                        help="Path den best.pt cua stage 1 (neu chi chay --stage 2)")
    args = parser.parse_args()

    MODEL      = args.model
    BATCH_SIZE = args.batch
    IMG_SIZE   = args.imgsz
    DEVICE     = int(args.device) if args.device.isdigit() else args.device
    cache_mode = "disk" if args.cache_disk else "ram"

    check_env()
    data_yaml = make_data_yaml()

    final_weight = None

    if args.stage in ("all", "1"):
        stage1_best  = train_stage1(data_yaml, cache_mode)
        final_weight = stage1_best

    if args.stage in ("all", "2"):
        if args.stage == "2":
            # Chi chay stage 2 rieng le
            if args.stage2_weight is None:
                print("[ERROR] --stage 2 can them --stage2-weight <path/to/best.pt>")
                sys.exit(1)
            s1_weight = Path(args.stage2_weight)
        else:
            s1_weight = stage1_best  # tu dong tu stage 1

        final_weight = train_stage2(data_yaml, s1_weight, cache_mode)

    if final_weight and not args.no_eval:
        run_name = STAGE2_NAME if args.stage in ("all", "2") else STAGE1_NAME
        evaluate(data_yaml, final_weight, run_name)

    if final_weight and args.export:
        export_model(final_weight, args.export)

    if final_weight:
        print(f"\n[DONE] Xong! Best model: {final_weight}")
