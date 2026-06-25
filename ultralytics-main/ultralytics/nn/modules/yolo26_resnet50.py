"""
YOLO26-ResNet50 Model
===============================================================
Wires the ResNet50Backbone → YOLO26 Neck (SPPF + C2PSA + FPN) → Detect Head.

Channel dimensions flow:
  Backbone out  →  p3:512, p4:1024, p5:2048
  After SPPF    →  p5_sppf: 1024        (halved by SPPF)
  After C2PSA   →  p5_attn: 1024
  After upsample+concat → p4_fused: 512
  After upsample+concat → p3_fused: 256
  Detect head receives [p3_fused:256, p4_fused:512, p5_attn:1024]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet50_backbone import ResNet50Backbone


# ═══════════════════════════════════════════════════════════
#  Neck building blocks
# ═══════════════════════════════════════════════════════════

class _ConvBnAct(nn.Module):
    """Conv → BN → SiLU (YOLO-style activation)."""
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1):
        super().__init__()
        p = p if p is not None else k // 2
        self.cv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.cv(x)))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling – Fast (YOLO26 variant).
    Three cascaded MaxPool with fixed kernel=5, then concat + 1×1 conv.
    Output channels = c_out  (default: c_in // 2).
    """
    def __init__(self, c_in: int, c_out: int, k: int = 5):
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = _ConvBnAct(c_in,      c_hidden, 1)
        self.cv2 = _ConvBnAct(c_hidden * 4, c_out, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class _Bottleneck(nn.Module):
    """Lightweight bottleneck used inside C3k2."""
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = _ConvBnAct(c, c, 3, p=1)
        self.cv2 = _ConvBnAct(c, c, 3, p=1)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2(nn.Module):
    """
    Cross Stage Partial with 2-conv bottleneck (YOLO26 style).
    Replaces C2f in newer YOLO variants.
    """
    def __init__(self, c_in: int, c_out: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        c_hidden = c_out // 2
        self.cv1 = _ConvBnAct(c_in, c_hidden, 1)
        self.cv2 = _ConvBnAct(c_in, c_hidden, 1)
        self.cv3 = _ConvBnAct(c_hidden * 2, c_out, 1)
        self.bottlenecks = nn.Sequential(
            *[_Bottleneck(c_hidden, shortcut) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat([self.bottlenecks(self.cv1(x)),
                                   self.cv2(x)], dim=1))


class C2PSA(nn.Module):
    """
    C2 with Parallel Self-Attention (YOLO26 global context module).
    Splits the channel into two paths:
      path-A → multi-head self-attention (global variety context)
      path-B → identity pass-through
    Both paths are concatenated and projected back.
    """
    def __init__(self, c: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert c % 2 == 0, "C2PSA requires even channel count"
        c_half = c // 2
        self.cv1  = _ConvBnAct(c, c, 1)       # pre-projection
        self.attn = nn.MultiheadAttention(
            embed_dim=c_half,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln   = nn.LayerNorm(c_half)
        self.cv2  = _ConvBnAct(c, c, 1)       # post-projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        B, C, H, W = x.shape
        c_half = C // 2
        x_a, x_b = x[:, :c_half], x[:, c_half:]   # split

        # flatten spatial dims for attention
        flat = x_a.flatten(2).permute(0, 2, 1)     # B, HW, c_half
        attn_out, _ = self.attn(flat, flat, flat)
        attn_out = self.ln(attn_out + flat)         # residual + LN
        attn_out = attn_out.permute(0, 2, 1).reshape(B, c_half, H, W)

        return self.cv2(torch.cat([attn_out, x_b], dim=1))


# ═══════════════════════════════════════════════════════════
#  Detection Head
# ═══════════════════════════════════════════════════════════

class DetectHead(nn.Module):
    """
    Anchor-free detection head (one-to-one at inference → no NMS needed).
    Predicts class logits + 4 box regressors per spatial location.

    For each scale:  [B, nc + 4, H, W]
    """
    def __init__(self, nc: int, ch: tuple):
        super().__init__()
        self.nc = nc
        self.reg_max = 16          # DFL regression bins
        c_reg = max(16, ch[0] // 4, self.reg_max * 4)
        c_cls = max(ch[0], nc)

        self.cv2 = nn.ModuleList(              # box branch
            nn.Sequential(
                _ConvBnAct(c, c_reg, 3, p=1),
                _ConvBnAct(c_reg, c_reg, 3, p=1),
                nn.Conv2d(c_reg, 4 * self.reg_max, 1),
            ) for c in ch
        )
        self.cv3 = nn.ModuleList(              # cls branch
            nn.Sequential(
                _ConvBnAct(c, c_cls, 3, p=1),
                _ConvBnAct(c_cls, c_cls, 3, p=1),
                nn.Conv2d(c_cls, nc, 1),
            ) for c in ch
        )

    def forward(self, features: list) -> list:
        """
        Args
            features : [p3_fused, p4_fused, p5_attn]
        Returns
            list of (box_raw, cls_logits) per scale
        """
        outputs = []
        for i, x in enumerate(features):
            box = self.cv2[i](x)   # [B, 4*reg_max, H, W]
            cls = self.cv3[i](x)   # [B, nc,        H, W]
            outputs.append((box, cls))
        return outputs


# ═══════════════════════════════════════════════════════════
#  Full YOLO26-ResNet50 Model
# ═══════════════════════════════════════════════════════════

class YOLO26ResNet50(nn.Module):
    """
    Complete YOLO26-ResNet50 detector.

    Parameters
    ----------
    nc          : number of output classes
    pretrained  : load ImageNet weights into the backbone
    """

    def __init__(self, nc: int = 5, pretrained: bool = False):
        super().__init__()
        self.nc = nc

        # ── Backbone ──────────────────────────────────────
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        # p3:512, p4:1024, p5:2048

        # ── Neck ──────────────────────────────────────────
        # Index 6: SPPF  2048 → 1024
        self.sppf   = SPPF(c_in=2048, c_out=1024, k=5)

        # Index 7: C2PSA global attention  1024 → 1024
        self.c2psa  = C2PSA(c=1024, num_heads=8)

        # P5→P4 path: upsample + concat(p4:1024) → C3k2 → 512
        self.up1    = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse1  = C3k2(c_in=1024 + 1024, c_out=512, n=3, shortcut=True)

        # P4→P3 path: upsample + concat(p3:512)  → C3k2 → 256
        self.up2    = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse2  = C3k2(c_in=512 + 512, c_out=256, n=3, shortcut=True)

        # ── Head ──────────────────────────────────────────
        self.head = DetectHead(nc=nc, ch=(256, 512, 1024))

    def forward(self, x: torch.Tensor):
        # ── backbone ──────────────────────────────────────
        p3, p4, p5 = self.backbone(x)
        # p3: [B,  512, H/8,  W/8 ]
        # p4: [B, 1024, H/16, W/16]
        # p5: [B, 2048, H/32, W/32]

        # ── neck ──────────────────────────────────────────
        p5 = self.c2psa(self.sppf(p5))          # [B,1024, H/32, W/32]

        p4_fused = self.fuse1(
            torch.cat([self.up1(p5), p4], dim=1) # [B,2048, H/16, W/16]
        )                                         # → [B,512, H/16, W/16]

        p3_fused = self.fuse2(
            torch.cat([self.up2(p4_fused), p3], dim=1)  # [B,1024, H/8,W/8]
        )                                                 # → [B,256, H/8, W/8]

        # ── head ──────────────────────────────────────────
        return self.head([p3_fused, p4_fused, p5])

    # ── convenience ───────────────────────────────────────
    def freeze_backbone(self):
        """Freeze all ResNet50 parameters (for stage-1 transfer learning)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[YOLO26ResNet50] Backbone frozen.")

    def unfreeze_backbone(self):
        """Unfreeze backbone (for stage-2 fine-tuning)."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("[YOLO26ResNet50] Backbone unfrozen for fine-tuning.")

    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": train}
