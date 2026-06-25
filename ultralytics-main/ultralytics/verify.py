"""
verify.py  –  Sanity-check YOLO26-ResNet50 without any data
=============================================================
Run this first to confirm the model builds, channels match,
and a forward pass completes without errors.

  python verify.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from ultralytics.nn.yolo26_resnet50 import YOLO26ResNet50
from ultralytics.nn.resnet50_backbone import ResNet50Backbone


def hr(char="─", n=60):
    print(char * n)


def verify_backbone():
    hr()
    print("1. ResNet50Backbone  (no pretrained weights)")
    hr()
    bb = ResNet50Backbone(pretrained=False)
    bb.eval()

    dummy = torch.zeros(2, 3, 640, 640)
    with torch.no_grad():
        p3, p4, p5 = bb(dummy)

    print(f"   Input : {tuple(dummy.shape)}")
    print(f"   P3    : {tuple(p3.shape)}   expected (2, 512,  80, 80)")
    print(f"   P4    : {tuple(p4.shape)}   expected (2,1024,  40, 40)")
    print(f"   P5    : {tuple(p5.shape)}   expected (2,2048,  20, 20)")

    assert p3.shape == (2,  512, 80, 80), f"P3 shape mismatch: {p3.shape}"
    assert p4.shape == (2, 1024, 40, 40), f"P4 shape mismatch: {p4.shape}"
    assert p5.shape == (2, 2048, 20, 20), f"P5 shape mismatch: {p5.shape}"
    print("   ✓  All backbone shapes correct.\n")


def verify_full_model():
    hr()
    print("2. YOLO26ResNet50  (nc=5, pretrained=False)")
    hr()
    model = YOLO26ResNet50(nc=5, pretrained=False)
    model.eval()

    info = model.num_params()
    print(f"   Params  total={info['total']:,}   trainable={info['trainable']:,}")

    dummy = torch.zeros(2, 3, 640, 640)
    with torch.no_grad():
        preds = model(dummy)

    print(f"   Head outputs: {len(preds)} scales")
    for i, (box, cls) in enumerate(preds):
        label = ["P3 (80×80)", "P4 (40×40)", "P5 (20×20)"][i]
        print(f"   Scale {i+1}  {label:<12s}  box={tuple(box.shape)}  cls={tuple(cls.shape)}")

    # Shape checks
    assert len(preds) == 3
    assert preds[0][1].shape[1] == 5   # nc=5 classes at P3
    assert preds[1][1].shape[1] == 5   # nc=5 classes at P4
    assert preds[2][1].shape[1] == 5   # nc=5 classes at P5
    print("   ✓  All head shapes correct.\n")


def verify_freeze_unfreeze():
    hr()
    print("3. Freeze / unfreeze backbone")
    hr()
    model = YOLO26ResNet50(nc=5)

    model.freeze_backbone()
    frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    print(f"   Frozen backbone params: {frozen}")
    assert frozen > 0

    model.unfreeze_backbone()
    unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    print(f"   Unfrozen backbone params: {unfrozen}")
    assert unfrozen > 0
    print("   ✓  Freeze / unfreeze working.\n")


def verify_skip_connections():
    hr()
    print("4. Skip connection gradient flow (vanishing-gradient check)")
    hr()
    model = YOLO26ResNet50(nc=5)
    model.train()

    dummy   = torch.randn(1, 3, 320, 320)
    preds   = model(dummy)
    loss    = sum(b.mean() + c.mean() for b, c in preds)
    loss.backward()

    # Check the very first conv in the backbone still has a gradient
    first_grad = model.backbone.stem[0].weight.grad
    assert first_grad is not None, "No gradient reached the backbone stem!"
    grad_norm = first_grad.norm().item()
    print(f"   Stem conv grad norm = {grad_norm:.6f}")
    assert grad_norm > 0, "Gradient is zero – possible vanishing gradient!"
    print("   ✓  Gradients flow through skip connections to backbone stem.\n")


if __name__ == "__main__":
    print("\n══════  YOLO26-ResNet50 Verification Suite  ══════\n")
    verify_backbone()
    verify_full_model()
    verify_freeze_unfreeze()
    verify_skip_connections()
    hr("═")
    print("All checks passed ✓")
    hr("═")
