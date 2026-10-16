"""
Script to generate 6 publication-grade technical architecture & benchmark figures
for the Vietnamese License Plate Detection repository:
1. docs/figures/1_dataset_pipeline.png
2. docs/figures/2_yolov8n_architecture.png
3. docs/figures/3_yolov8l_architecture.png
4. docs/figures/4_inference_pipeline.png
5. docs/figures/5_resolution_tradeoff.png
6. docs/figures/6_model_comparison.png

Design specifications:
- Wide canvas, modern high-contrast aesthetic, crisp sans-serif typography.
- Isometric 3D-styled volumetric tensor blocks & explicit dimension labels.
- Zero overlapping labels; high readability.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
import numpy as np

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"

def draw_3d_box(ax, x, y, w, h, depth=0.15, facecolor="#2b5c8f", edgecolor="#1a365d", alpha=0.9, label=None, sublabel=None):
    """Draw an isometric volumetric tensor block."""
    # Front face
    front = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, alpha=alpha)
    ax.add_patch(front)

    # Top face
    top_poly = np.array([
        [x, y + h],
        [x + depth * 0.7, y + h + depth * 0.5],
        [x + w + depth * 0.7, y + h + depth * 0.5],
        [x + w, y + h]
    ])
    top = Polygon(top_poly, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2, alpha=min(alpha + 0.1, 1.0))
    ax.add_patch(top)

    # Right face
    right_poly = np.array([
        [x + w, y],
        [x + w + depth * 0.7, y + depth * 0.5],
        [x + w + depth * 0.7, y + h + depth * 0.5],
        [x + w, y + h]
    ])
    right = Polygon(right_poly, facecolor=edgecolor, edgecolor=edgecolor, linewidth=1.2, alpha=alpha * 0.75)
    ax.add_patch(right)

    if label:
        ax.text(x + w / 2, y + h / 2 + (0.04 if sublabel else 0), label,
                ha="center", va="center", color="#ffffff", fontsize=11, fontweight="bold")
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.05, sublabel,
                ha="center", va="center", color="#e2e8f0", fontsize=8.5, fontweight="normal")

def fig1_dataset_pipeline():
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    # Title
    ax.text(0.5, 0.93, "Dataset Processing & Partitioning Pipeline",
            ha="center", va="center", color="#38bdf8", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.86, "1,005 Raw Images | 1 Single Class ('plate') | Verified Zero Cross-Split Leakage",
            ha="center", va="center", color="#94a3b8", fontsize=11)

    # Stages
    stages = [
        ("Raw Captures", "1,005 Images\nResolution: 600x480 & 472x303", "#1e293b", "#334155", 0.05),
        ("Annotation Audit", "1,006 Ground Truth Bboxes\n100% Normalized [0, 1]", "#0369a1", "#0284c7", 0.28),
        ("Integrity Verif.", "MD5 Cross-Split Hashing\n0 Duplicate Images", "#047857", "#059669", 0.51),
        ("Dataset Splits", "Train (704) | Valid (200) | Test (101)\nClass 0: 'plate'", "#7c3aed", "#8b5cf6", 0.74),
    ]

    for title, desc, bg, border, x in stages:
        box = FancyBboxPatch((x, 0.35), 0.20, 0.40, boxstyle="round,pad=0.02",
                             facecolor=bg, edgecolor=border, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.10, 0.65, title, ha="center", va="center", color="#f8fafc", fontsize=13, fontweight="bold")
        ax.text(x + 0.10, 0.48, desc, ha="center", va="center", color="#cbd5e1", fontsize=9.5, linespacing=1.4)

        if x < 0.70:
            ax.annotate("", xy=(x + 0.22, 0.55), xytext=(x + 0.20, 0.55),
                        arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3))

    # Test scale slice breakdown sub-box
    slice_box = FancyBboxPatch((0.15, 0.08), 0.70, 0.18, boxstyle="round,pad=0.02",
                               facecolor="#1e1e38", edgecolor="#6366f1", linewidth=1.5)
    ax.add_patch(slice_box)
    ax.text(0.5, 0.21, "Test Split Scale Breakdown (N = 101 Images / 101 Plates)",
            ha="center", va="center", color="#a5b4fc", fontsize=11, fontweight="bold")
    ax.text(0.26, 0.13, "Small / Far (<2.5% area)\n34 Images (33.7%)", ha="center", va="center", color="#e0e7ff", fontsize=9)
    ax.text(0.50, 0.13, "Medium (2.5% - 6.0% area)\n49 Images (48.5%)", ha="center", va="center", color="#e0e7ff", fontsize=9)
    ax.text(0.74, 0.13, "Large / Near (>6.0% area)\n18 Images (17.8%)", ha="center", va="center", color="#e0e7ff", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "1_dataset_pipeline.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

def fig2_yolov8n_architecture():
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    ax.set_facecolor("#0b1329")
    fig.patch.set_facecolor("#0b1329")

    ax.text(0.5, 0.93, "YOLOv8n Compact Model Architecture (3.0M Parameters)",
            ha="center", va="center", color="#38bdf8", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.86, "Training: 640x640 (8.1 GFLOPs) | Compact Deployment: 224x224 (1,029 Decoded Anchors)",
            ha="center", va="center", color="#94a3b8", fontsize=11)

    blocks = [
        ("Input", "3x224x224", 0.04, 0.38, 0.11, 0.30, "#1e3a8a", "#3b82f6"),
        ("Stem P1", "Conv 3x3 / s2\n16x112x112", 0.18, 0.40, 0.12, 0.26, "#1e40af", "#60a5fa"),
        ("Backbone\nP2-P5", "C2f + Conv Blocks\nChannels: 32, 64, 128, 256", 0.33, 0.36, 0.16, 0.34, "#0369a1", "#0ea5e9"),
        ("SPPF", "Multi-kernel MaxPool\n1x1, 5x5, 9x9, 13x13", 0.52, 0.41, 0.12, 0.24, "#0d9488", "#14b8a6"),
        ("PAN-FPN", "Multi-scale Neck\nP3, P4, P5 Fusion", 0.67, 0.36, 0.14, 0.34, "#4f46e5", "#818cf8"),
        ("Detect Head", "Decoupled Cls + Reg\nOutput: [1, 5, 1029]", 0.84, 0.37, 0.13, 0.32, "#be185d", "#f43f5e"),
    ]

    for title, desc, x, y, w, h, c1, c2 in blocks:
        draw_3d_box(ax, x, y, w, h, depth=0.03, facecolor=c1, edgecolor=c2, label=title, sublabel=desc)
        if x < 0.80:
            ax.annotate("", xy=(x + w + 0.02, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5))

    # Spec details footer
    spec_box = FancyBboxPatch((0.08, 0.08), 0.84, 0.18, boxstyle="round,pad=0.02",
                              facecolor="#172554", edgecolor="#1d4ed8", linewidth=1.5)
    ax.add_patch(spec_box)
    ax.text(0.5, 0.21, "Key Execution Specifications (Compact Deployment Variant)",
            ha="center", va="center", color="#93c5fd", fontsize=11, fontweight="bold")
    ax.text(0.24, 0.13, "Input Shape: [1, 3, 224, 224]\nDecoded Anchors: 1,029 (5 channels)", ha="center", va="center", color="#dbeafe", fontsize=9.5)
    ax.text(0.50, 0.13, "Stride Levels: 8, 16, 32\nPure ONNX Forward: 3.42 ms (CPU)", ha="center", va="center", color="#dbeafe", fontsize=9.5)
    ax.text(0.76, 0.13, "Model Weights: 5.95 MB (.pt)\nONNX Weights: 11.56 MB (FP32)", ha="center", va="center", color="#dbeafe", fontsize=9.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_yolov8n_architecture.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

def fig3_yolov8l_architecture():
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    ax.set_facecolor("#0b1329")
    fig.patch.set_facecolor("#0b1329")

    ax.text(0.5, 0.93, "YOLOv8l Server Baseline Architecture (43.6M Parameters)",
            ha="center", va="center", color="#a855f7", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.86, "High-Capacity Server Baseline | Input: 640x640 (164.8 GFLOPs) | 8,400 Decoded Anchors",
            ha="center", va="center", color="#94a3b8", fontsize=11)

    blocks = [
        ("Input", "3x640x640", 0.04, 0.36, 0.11, 0.34, "#3b0764", "#7e22ce"),
        ("Deep Stem", "Conv 3x3 / s2\n64x320x320", 0.18, 0.38, 0.12, 0.30, "#581c87", "#9333ea"),
        ("Heavy Backbone", "C2f Deep Blocks\nChannels: 128, 256, 512, 512", 0.33, 0.33, 0.16, 0.40, "#6b21a8", "#a855f7"),
        ("SPPF", "Multi-kernel MaxPool\n1x1, 5x5, 9x9, 13x13", 0.52, 0.40, 0.12, 0.26, "#7e22ce", "#c084fc"),
        ("Dense Neck", "Heavy PAN-FPN\nRich Semantic Feats", 0.67, 0.34, 0.14, 0.38, "#86198f", "#d946ef"),
        ("Detect Head", "Decoupled Cls + Reg\nOutput: [1, 5, 8400]", 0.84, 0.36, 0.13, 0.34, "#9d174d", "#f43f5e"),
    ]

    for title, desc, x, y, w, h, c1, c2 in blocks:
        draw_3d_box(ax, x, y, w, h, depth=0.03, facecolor=c1, edgecolor=c2, label=title, sublabel=desc)
        if x < 0.80:
            ax.annotate("", xy=(x + w + 0.02, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="->", color="#c084fc", lw=2.5))

    # Spec details footer
    spec_box = FancyBboxPatch((0.08, 0.08), 0.84, 0.18, boxstyle="round,pad=0.02",
                              facecolor="#2e1065", edgecolor="#581c87", linewidth=1.5)
    ax.add_patch(spec_box)
    ax.text(0.5, 0.21, "Key Execution Specifications (Server Baseline Variant)",
            ha="center", va="center", color="#d8b4fe", fontsize=11, fontweight="bold")
    ax.text(0.24, 0.13, "Input Shape: [1, 3, 640, 640]\nDecoded Anchors: 8,400 (5 channels)", ha="center", va="center", color="#f3e8ff", fontsize=9.5)
    ax.text(0.50, 0.13, "High Distance Recall: 100.0%\nPure ONNX Forward: 372.07 ms (CPU)", ha="center", va="center", color="#f3e8ff", fontsize=9.5)
    ax.text(0.76, 0.13, "Model Weights: 83.58 MB (.pt)\nONNX Weights: 166.62 MB (FP32)", ha="center", va="center", color="#f3e8ff", fontsize=9.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_yolov8l_architecture.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

def fig4_inference_pipeline():
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=300)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    ax.text(0.5, 0.93, "End-to-End ONNX Preprocessing, Inference & Mapping Contract",
            ha="center", va="center", color="#38bdf8", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.86, "Strict Mathematical Inversion: Zero Distortion | Integer Letterbox Rounding (dh - 0.1 / dh + 0.1)",
            ha="center", va="center", color="#94a3b8", fontsize=11)

    steps = [
        ("Input Frame", "Raw Image\nW x H x 3 (BGR)", 0.03, "#1e293b", "#334155"),
        ("Letterbox", "Aspect-ratio pad (114)\nr = min(S/H, S/W)", 0.20, "#0369a1", "#0284c7"),
        ("Tensor Norm", "BGR->RGB, /255.0\nShape: [1, 3, S, S]", 0.37, "#047857", "#059669"),
        ("ONNX Engine", "CPU ExecutionProvider\nOutput0: [1, 5, N]", 0.54, "#4338ca", "#4f46e5"),
        ("Decode & NMS", "cxcywh -> xyxy\nIoU >= 0.5, Conf >= 0.25", 0.71, "#7c3aed", "#8b5cf6"),
        ("Inverse Map", "x = (x224 - left) / r\nTrue Original Bboxes", 0.88, "#be185d", "#f43f5e"),
    ]

    for title, desc, x, bg, border in steps:
        box = FancyBboxPatch((x, 0.35), 0.10, 0.38, boxstyle="round,pad=0.02",
                             facecolor=bg, edgecolor=border, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.05, 0.64, title, ha="center", va="center", color="#f8fafc", fontsize=11.5, fontweight="bold")
        ax.text(x + 0.05, 0.48, desc, ha="center", va="center", color="#cbd5e1", fontsize=8.5, linespacing=1.35)

        if x < 0.80:
            ax.annotate("", xy=(x + 0.13, 0.54), xytext=(x + 0.11, 0.54),
                        arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5))

    # Formula breakdown footer
    f_box = FancyBboxPatch((0.08, 0.08), 0.84, 0.18, boxstyle="round,pad=0.02",
                           facecolor="#182234", edgecolor="#2563eb", linewidth=1.5)
    ax.add_patch(f_box)
    ax.text(0.5, 0.20, "Mathematical Preprocessing & Inversion Formulas",
            ha="center", va="center", color="#60a5fa", fontsize=11, fontweight="bold")
    ax.text(0.28, 0.13, "Letterbox Scaling:\n$r = \\min(S/H, S/W)$, $unpad = [\\mathrm{round}(W \\cdot r), \\mathrm{round}(H \\cdot r)]$",
            ha="center", va="center", color="#e2e8f0", fontsize=9.5)
    ax.text(0.72, 0.13, "Inverse Coordinate Mapping:\n$x_{\\mathrm{orig}} = (x_S - pad_{left}) / r$, $y_{\\mathrm{orig}} = (y_S - pad_{top}) / r$",
            ha="center", va="center", color="#e2e8f0", fontsize=9.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_inference_pipeline.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

def fig5_resolution_tradeoff():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    ax.text(0.5, 0.93, "Resolution Trade-off: Small Plate Recall Breakdown",
            ha="center", va="center", color="#38bdf8", fontsize=17, fontweight="bold")
    ax.text(0.5, 0.86, "Evaluated on Test Split (N = 101 Images / 101 Ground Truth Bounding Boxes)",
            ha="center", va="center", color="#94a3b8", fontsize=11)

    categories = ["Small / Far (<2.5% area)\n[N = 34]", "Medium (2.5% - 6.0% area)\n[N = 49]", "Large / Near (>6.0% area)\n[N = 18]"]
    x = np.arange(len(categories))
    width = 0.32

    rec_nano = [79.4, 98.0, 100.0]
    rec_large = [100.0, 100.0, 100.0]

    rects1 = ax.bar(x - width/2, rec_nano, width, label="YOLOv8n @ 224x224 (Compact)", color="#38bdf8", edgecolor="#0284c7", linewidth=1.5)
    rects2 = ax.bar(x + width/2, rec_large, width, label="YOLOv8l @ 640x640 (Server)", color="#a855f7", edgecolor="#7e22ce", linewidth=1.5)

    ax.set_ylabel("Detection Recall (%)", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", colors="#94a3b8")
    ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.6)

    # Add values on bars
    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", color="#38bdf8", fontweight="bold", fontsize=11)
    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", color="#a855f7", fontweight="bold", fontsize=11)

    # Highlight box for 20.6% miss rate
    diff_box = FancyBboxPatch((-0.42, 60), 0.35, 15, boxstyle="round,pad=1", facecolor="#450a0a", edgecolor="#ef4444", linewidth=1.5)
    ax.add_patch(diff_box)
    ax.text(-0.24, 66, "-20.6% Recall Drop\nSpatial feature loss", ha="center", va="center", color="#fca5a5", fontsize=9.5, fontweight="bold")

    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", loc="lower right", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "5_resolution_tradeoff.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

def fig6_model_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("#0f172a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0f172a")
        ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#475569")
        ax.spines["bottom"].set_color("#475569")
        ax.tick_params(colors="#94a3b8")

    models = ["YOLOv8n (224x224)\n[Compact]", "YOLOv8l (640x640)\n[Server]"]
    colors = ["#38bdf8", "#a855f7"]
    edges = ["#0284c7", "#7e22ce"]

    # Subplot 1: Latency (ms)
    latencies = [3.70, 375.01]
    b1 = ax1.bar(models, latencies, color=colors, edgecolor=edges, width=0.45, linewidth=1.5)
    ax1.set_title("End-to-End CPU Latency (Lower is Better)\nIntel Core i5-14600K (4 Threads)", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Latency (ms)", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 430)
    for b in b1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 8, f"{h:.2f} ms", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=11)
    ax1.text(0.5, 180, "101.4x CPU Speedup", ha="center", va="center", color="#38bdf8", fontsize=13, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor="#38bdf8"))

    # Subplot 2: mAP@[0.5:0.95]
    maps = [68.82, 78.47]
    b2 = ax2.bar(models, maps, color=colors, edgecolor=edges, width=0.45, linewidth=1.5)
    ax2.set_title("Detection Accuracy (Higher is Better)\nTest Split mAP@[0.5:0.95] (N = 101)", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("mAP@[0.5:0.95] (%)", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 95)
    for b in b2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 2, f"{h:.2f}%", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=11)
    ax2.text(0.5, 45, "+9.65 mAP Gain\nFull spatial detail", ha="center", va="center", color="#a855f7", fontsize=11, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor="#a855f7"))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "6_model_comparison.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

def main():
    print("Generating technical architecture figures into docs/figures/...")
    fig1_dataset_pipeline()
    fig2_yolov8n_architecture()
    fig3_yolov8l_architecture()
    fig4_inference_pipeline()
    fig5_resolution_tradeoff()
    fig6_model_comparison()
    print("Successfully generated all 6 figures!")

if __name__ == "__main__":
    main()
