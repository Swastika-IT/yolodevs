"""
predict.py  –  YOLO26-ResNet50 Inference & Export
=============================================================
Modes:
  image  –  run on a single image file
  export –  export to ONNX or TorchScript

Usage
-----
  # Image inference
  python predict.py image \\
      --weights runs/train/best.pt \\
      --source  samples/rice_sample.jpg \\
      --nc 5

  # ONNX export
  python predict.py export \\
      --weights runs/train/best.pt \\
      --format  onnx \\
      --nc 5
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from ultralytics.nn.yolo26_resnet50 import YOLO26ResNet50

CLASS_NAMES = ["Full", "Broken", "Chalky", "Straw", "Stone"]


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def load_model(weights: str, nc: int, device: torch.device) -> YOLO26ResNet50:
    model = YOLO26ResNet50(nc=nc).to(device)
    ckpt  = torch.load(weights, map_location=device)
    state = ckpt.get("model_state", ckpt)   # handle bare state-dict too
    model.load_state_dict(state)
    model.eval()
    print(f"[predict] Loaded weights from  {weights}")
    return model


def preprocess(img_path: str, imgsz: int = 640, device: torch.device = None):
    """Load an image, letterbox to imgsz, return tensor [1,3,H,W] in [0,1]."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise ImportError("pip install opencv-python numpy")

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # simple resize (replace with letterbox for production)
    img = cv2.resize(img, (imgsz, imgsz))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def decode_predictions(preds, conf_thres: float = 0.25):
    """
    Minimal decode: take the highest-confidence class per scale per location.
    Replace with your full DFL box decoder + NMS for production.
    """
    results = []
    for box_raw, cls_logits in preds:
        probs  = cls_logits.sigmoid()                  # [B, nc, H, W]
        conf, idx = probs.max(dim=1)                   # [B, H, W]
        mask   = conf > conf_thres
        if mask.any():
            for b in range(conf.shape[0]):
                hits = mask[b].nonzero(as_tuple=False) # [K, 2]
                for (gy, gx) in hits:
                    c = idx[b, gy, gx].item()
                    s = conf[b, gy, gx].item()
                    results.append({
                        "class": CLASS_NAMES[c] if c < len(CLASS_NAMES) else c,
                        "conf":  round(s, 3),
                        "grid":  (gy.item(), gx.item()),
                    })
    return results


# ═══════════════════════════════════════════════════════════
#  Modes
# ═══════════════════════════════════════════════════════════

def run_image(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.weights, args.nc, device)

    tensor = preprocess(args.source, args.imgsz, device)
    with torch.no_grad():
        preds = model(tensor)

    detections = decode_predictions(preds, conf_thres=args.conf)
    if detections:
        print(f"\n[predict] {len(detections)} detection(s):")
        for d in detections:
            print(f"  class={d['class']:8s}  conf={d['conf']:.3f}  "
                  f"grid={d['grid']}")
    else:
        print("[predict] No detections above threshold.")


def run_export(args):
    device = torch.device("cpu")
    model  = load_model(args.weights, args.nc, device)
    dummy  = torch.zeros(1, 3, args.imgsz, args.imgsz)

    fmt = args.format.lower()
    if fmt == "onnx":
        try:
            import onnx
        except ImportError:
            raise ImportError("pip install onnx")
        out = Path(args.weights).with_suffix(".onnx")
        torch.onnx.export(
            model, dummy, str(out),
            input_names=["images"],
            output_names=["box_s", "cls_s", "box_m", "cls_m", "box_l", "cls_l"],
            opset_version=17,
            dynamic_axes={"images": {0: "batch"}},
        )
        print(f"[export] ONNX saved → {out}")

    elif fmt == "torchscript":
        out = Path(args.weights).with_suffix(".torchscript.pt")
        traced = torch.jit.trace(model, dummy)
        traced.save(str(out))
        print(f"[export] TorchScript saved → {out}")

    else:
        print(f"[export] Unknown format '{fmt}'.  Supported: onnx, torchscript")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="YOLO26-ResNet50 predict / export")
    sub = p.add_subparsers(dest="mode")

    img = sub.add_parser("image")
    img.add_argument("--weights", required=True)
    img.add_argument("--source",  required=True)
    img.add_argument("--nc",      type=int,   default=5)
    img.add_argument("--imgsz",   type=int,   default=640)
    img.add_argument("--conf",    type=float, default=0.25)

    exp = sub.add_parser("export")
    exp.add_argument("--weights", required=True)
    exp.add_argument("--format",  default="onnx", choices=["onnx","torchscript"])
    exp.add_argument("--nc",      type=int,   default=5)
    exp.add_argument("--imgsz",   type=int,   default=640)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if   args.mode == "image":  run_image(args)
    elif args.mode == "export": run_export(args)
    else:
        print("Usage: python predict.py {image|export} --help")
