"""
ResNet50 Backbone for YOLO26
===============================================================
Drop-in replacement backbone that exposes the standard
P3 / P4 / P5 feature-map hierarchy expected by the YOLO26 neck.

Architecture (mirrors the PDF diagram exactly):
  Stage 1  : Conv 7x7  + BN + ReLU  → stride-2  (½)
             MaxPool 3x3             → stride-2  (¼)   [not a P-level]
  Stage 2  : 3 × Bottleneck (64→256)               P2 – 1/4  (unused by neck)
  Stage 3  : 4 × Bottleneck (128→512)              P3 – 1/8   → neck concat index 12
  Stage 4  : 6 × Bottleneck (256→1024)             P4 – 1/16  → neck concat index 9
  Stage 5  : 3 × Bottleneck (512→2048)             P5 – 1/32  → SPPF
"""

import torch
import torch.nn as nn
from typing import List


# ───────────────────────────────────────────────────────────
# 1.  Primitive helpers
# ───────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch: int, out_ch: int,
                  kernel: int = 1, stride: int = 1,
                  padding: int = 0, groups: int = 1) -> nn.Sequential:
    """Conv → BN → ReLU fused block (no bias, BN handles it)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ───────────────────────────────────────────────────────────
# 2.  ResNet Bottleneck  (the "ResNetBlock" referenced in YAML)
# ───────────────────────────────────────────────────────────

class ResNetBottleneck(nn.Module):
    """
    Standard ResNet-50 bottleneck:
        1×1 (squeeze) → 3×3 (process) → 1×1 (expand) + skip-connection

    expansion = 4  →  out_channels = planes × 4

    The 1×1 skip projection is added automatically when:
      • stride > 1  (spatial downsampling), or
      • in_channels ≠ out_channels
    """
    expansion: int = 4

    def __init__(self, in_ch: int, planes: int, stride: int = 1):
        super().__init__()
        out_ch = planes * self.expansion

        # ── main path ──────────────────────────────────────
        self.conv1 = _conv_bn_relu(in_ch, planes, kernel=1)           # 1×1
        self.conv2 = _conv_bn_relu(planes, planes,                     # 3×3
                                   kernel=3, stride=stride, padding=1)
        # final 1×1: NO ReLU yet (applied after residual add)
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

        # ── skip (projection) ──────────────────────────────
        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        return self.relu(out + identity)          # ← residual add → no vanishing grad


# ───────────────────────────────────────────────────────────
# 3.  One Stage = multiple bottlenecks
# ───────────────────────────────────────────────────────────

def _make_stage(in_ch: int, planes: int,
                blocks: int, stride: int = 1) -> nn.Sequential:
    """
    Build one ResNet stage (list of Bottleneck blocks).
    The FIRST block handles the stride + channel projection;
    subsequent blocks use stride=1 and in_ch = planes*4.
    """
    layers: List[nn.Module] = [ResNetBottleneck(in_ch, planes, stride)]
    in_ch = planes * ResNetBottleneck.expansion
    for _ in range(1, blocks):
        layers.append(ResNetBottleneck(in_ch, planes))
    return nn.Sequential(*layers)


# ───────────────────────────────────────────────────────────
# 4.  Full ResNet-50 Backbone (returns P3, P4, P5)
# ───────────────────────────────────────────────────────────

class ResNet50Backbone(nn.Module):
    """
    ResNet-50 feature extractor compatible with the YOLO26 neck.

    Returns
    -------
    (p3, p4, p5) : tuple of tensors
        p3 → 1/8   spatial resolution, 512  channels
        p4 → 1/16  spatial resolution, 1024 channels
        p5 → 1/32  spatial resolution, 2048 channels
    """

    # ResNet-50 block counts per stage
    _BLOCKS = (3, 4, 6, 3)

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # ── Stage 1: stem ──────────────────────────────────
        # 640×640 → 320×320  (stride 2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),           # 7×7 conv
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 320×320 → 160×160  (stride 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Stages 2-5 ─────────────────────────────────────
        # stage2: in=64,  planes=64,  out=256,  stride=1  → 1/4
        self.stage2 = _make_stage(64,  64,  self._BLOCKS[0], stride=1)
        # stage3: in=256, planes=128, out=512,  stride=2  → 1/8  (P3)
        self.stage3 = _make_stage(256, 128, self._BLOCKS[1], stride=2)
        # stage4: in=512, planes=256, out=1024, stride=2  → 1/16 (P4)
        self.stage4 = _make_stage(512, 256, self._BLOCKS[2], stride=2)
        # stage5: in=1024,planes=512, out=2048, stride=2  → 1/32 (P5)
        self.stage5 = _make_stage(1024,512, self._BLOCKS[3], stride=2)

        self._init_weights()
        if pretrained:
            self._load_imagenet_weights()

    # ── weight init ────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def _load_imagenet_weights(self):
        """Load torchvision ResNet50 ImageNet weights via state-dict mapping."""
        try:
            import torchvision.models as tvm
            src = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1).state_dict()
            # Build key map: torchvision → our naming
            mapping = {
                # stem
                "conv1.weight":    "stem.0.weight",
                "bn1.weight":      "stem.1.weight",
                "bn1.bias":        "stem.1.bias",
                "bn1.running_mean":"stem.1.running_mean",
                "bn1.running_var": "stem.1.running_var",
                # stages
                "layer1": "stage2",
                "layer2": "stage3",
                "layer3": "stage4",
                "layer4": "stage5",
            }
            dst = {}
            for k, v in src.items():
                new_k = k
                for old, new in mapping.items():
                    new_k = new_k.replace(old, new)
                dst[new_k] = v
            missing, unexpected = self.load_state_dict(dst, strict=False)
            print(f"[ResNet50Backbone] Pretrained weights loaded. "
                  f"Missing: {len(missing)}  Unexpected: {len(unexpected)}")
        except Exception as e:
            print(f"[ResNet50Backbone] Warning – could not load pretrained weights: {e}")

    # ── forward ────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        x = self.maxpool(self.stem(x))  # 1/4  (after stem + pool)
        x = self.stage2(x)              # 1/4   P2  (256 ch)
        p3 = self.stage3(x)             # 1/8   P3  (512 ch)
        p4 = self.stage4(p3)            # 1/16  P4  (1024 ch)
        p5 = self.stage5(p4)            # 1/32  P5  (2048 ch)
        return p3, p4, p5

    # ── channel info (used by YOLO26 neck builder) ─────────
    @property
    def out_channels(self):
        return {"p3": 512, "p4": 1024, "p5": 2048}
