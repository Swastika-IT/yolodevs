"""
train.py  –  YOLO26-ResNet50 Training Script
=============================================================
Two-stage training strategy:
  Stage 1 (epochs 0 → freeze_epochs):   backbone frozen, neck+head learn fast
  Stage 2 (epochs freeze_epochs → end): full fine-tune with lower LR

Usage
-----
  python train.py \\
      --data   data/rice.yaml \\
      --cfg    configs/models/yolo26_resnet50.yaml \\
      --epochs 100 \\
      --batch  16 \\
      --imgsz  640 \\
      --freeze-epochs 30 \\
      --pretrained          # load ImageNet weights into backbone
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# ── local imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.nn.yolo26_resnet50 import YOLO26ResNet50


# ═══════════════════════════════════════════════════════════════════════════
#  Minimal training harness
# ═══════════════════════════════════════════════════════════════════════════

def build_optimizer(model: YOLO26ResNet50,
                    lr: float,
                    weight_decay: float) -> optim.Optimizer:
    """
    Separate parameter groups so backbone can receive a smaller LR.
    During stage-1 the backbone is frozen so its group has grad=False;
    AdamW simply skips it.
    """
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    neck_head_params = [p for p in model.parameters()
                        if id(p) not in backbone_ids]
    backbone_params  = list(model.backbone.parameters())

    return optim.AdamW([
        {"params": neck_head_params, "lr": lr},
        {"params": backbone_params,  "lr": lr * 0.1},   # 10× smaller for backbone
    ], weight_decay=weight_decay)


def one_epoch(model, loader, optimizer, device, scaler, epoch_idx):
    """Single training epoch (placeholder loss – replace with your YOLO loss)."""
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        imgs   = batch["images"].to(device, non_blocking=True)   # [B,3,H,W]
        # targets = batch["labels"].to(device)   # your label tensors

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            preds = model(imgs)
            # ── Replace this placeholder with your actual YOLO loss ────────
            # loss, loss_items = criterion(preds, targets)
            loss = sum(box.mean() + cls.mean()
                       for box, cls in preds)                     # PLACEHOLDER
            # ──────────────────────────────────────────────────────────────

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  [E{epoch_idx:03d} B{batch_idx:04d}]  loss={loss.item():.4f}")

    elapsed = time.time() - t0
    return total_loss / max(len(loader), 1), elapsed


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = YOLO26ResNet50(nc=args.nc, pretrained=args.pretrained).to(device)
    info  = model.num_params()
    print(f"[train] params  total={info['total']:,}  trainable={info['trainable']:,}")

    # Stage-1: freeze backbone
    if args.freeze_epochs > 0:
        model.freeze_backbone()

    # ── Optimiser + Scheduler ─────────────────────────────────────────────
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=args.epochs - args.freeze_epochs,
                                  eta_min=args.lr * 0.01)

    # AMP scaler (skip on CPU)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Data (replace with your actual dataset) ───────────────────────────
    # Example:
    # from data.rice_dataset import RiceDataset
    # train_ds = RiceDataset(args.data, split="train", imgsz=args.imgsz)
    # train_loader = DataLoader(train_ds, batch_size=args.batch,
    #                           shuffle=True, num_workers=4, pin_memory=True)
    train_loader = []   # ← replace with real DataLoader

    # ── Checkpoint dir ────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(args.epochs):

        # Stage-2 transition
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            model.unfreeze_backbone()
            # Drop neck+head LR; backbone gets a fresh, even smaller LR
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = args.lr * (0.1 if i == 0 else 0.01)
            print(f"[train] Epoch {epoch}: backbone unfrozen, LR reduced.")

        if not train_loader:
            print("[train] ⚠  No DataLoader – skipping forward pass (stub mode)")
            break

        avg_loss, elapsed = one_epoch(model, train_loader,
                                      optimizer, device, scaler, epoch)
        print(f"[E{epoch:03d}] avg_loss={avg_loss:.4f}  "
              f"time={elapsed:.1f}s  lr={optimizer.param_groups[0]['lr']:.2e}")

        if epoch >= args.freeze_epochs:
            scheduler.step()

        # ── Save checkpoint ───────────────────────────────────────────────
        ckpt = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "loss":        avg_loss,
            "args":        vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  ✓ best checkpoint saved  (loss={best_loss:.4f})")

    print("[train] Done.")
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO26-ResNet50")
    p.add_argument("--data",          default="data/rice.yaml")
    p.add_argument("--cfg",           default="configs/models/yolo26_resnet50.yaml")
    p.add_argument("--nc",            type=int,   default=5)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch",         type=int,   default=16)
    p.add_argument("--imgsz",         type=int,   default=640)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--wd",            type=float, default=5e-4)
    p.add_argument("--freeze-epochs", type=int,   default=30,
                   dest="freeze_epochs",
                   help="Epochs to keep backbone frozen (stage-1)")
    p.add_argument("--pretrained",    action="store_true",
                   help="Load ImageNet weights into ResNet50 backbone")
    p.add_argument("--save-dir",      default="runs/train",
                   dest="save_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
