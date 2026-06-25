"""
mnv4_modules.py
MobileNetV4 building blocks for YOLO26 backbone integration.
Add these to your rice_modules.py (or import from here into tasks.py).

All modules match the exact architectural descriptions from:
Qin et al. "MobileNetV4: Universal Models for the Mobile Ecosystem" (2024)

Modules:
    FusedIB     - Fused Inverted Bottleneck (stem layers)
    ExtraDW     - Extra DepthWise block (main backbone block)
    ConvNext    - ConvNext-like block (spatial mixing before expansion)
    FFN         - Feed-Forward Network (channel mixing only)
"""

import torch
import torch.nn as nn


# ============================================================
# Helper: standard Conv-BN-SiLU block
# ============================================================
class CBS(nn.Module):
    """Conv + BatchNorm + SiLU. The fundamental building unit."""
    def __init__(self, c1, c2, k=1, s=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ============================================================
# 1. FusedIB — Fused Inverted Bottleneck
# ============================================================
class FusedIB(nn.Module):
    """
    Fused Inverted Bottleneck (FusedIB).
    Used in MNv4 stem layers (early high-resolution stages).

    Standard Inverted Bottleneck does:
        PW-expand -> DW-spatial -> PW-project

    FusedIB fuses the expansion + spatial mixing into one Conv2D:
        kxk-Conv (expand+spatial) -> 1x1-Conv (project)

    Why this is better in early layers:
        At high resolution (320x320, 160x160), the combined Conv2D
        has higher operational intensity (MACs/byte) than a split
        PW+DW approach, making it faster on CPUs and accelerators.
        This is the insight from the MNv4 Roofline analysis (Section 3).

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k  (int): Kernel size for the fused spatial+expand conv (default 3)
        s  (int): Stride (2 for downsampling, 1 otherwise)
    """
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        expand = c2 * 4  # standard expansion ratio of 4

        # Fused: kxk Conv handles both spatial mixing and expansion
        self.fused_conv = CBS(c1, expand, k, s)

        # Project back to output channels
        self.proj = CBS(expand, c2, 1, 1)

        # Shortcut if input/output match (stride=1, same channels)
        self.use_shortcut = (s == 1 and c1 == c2)

    def forward(self, x):
        out = self.proj(self.fused_conv(x))
        if self.use_shortcut:
            out = out + x
        return out


# ============================================================
# 2. ExtraDW — Extra DepthWise Inverted Bottleneck
# ============================================================
class ExtraDW(nn.Module):
    """
    Extra DepthWise (ExtraDW) block.
    The PRIMARY building block of MobileNetV4's searchable stages.

    Structure (from MNv4 paper Figure 4, 'Extra DW' variant):
        Optional DW_K1 (before expansion)
        -> PW expand
        -> DW_K2 (between expand and project)
        -> PW project

    This is essentially an IB block with an ADDITIONAL depthwise conv
    inserted BEFORE the expansion. The two DW convolutions give:
        1. A larger effective receptive field (product of two kernel sizes)
        2. Extra spatial mixing capacity at lower parameter cost than
           a single large-kernel DW (e.g., 3x3+5x5 < 7x7 in params)
        3. The NAS procedure (TuNAS) selects K1 and K2 independently

    For rice quality: this is ideal for irregular grain shapes because
    the adaptive dual-kernel receptive field captures both local crack
    details (small K) and overall grain outline (large K).

    Args:
        c1   (int): Input channels
        c2   (int): Output channels
        k1   (int): Kernel size for the first (pre-expansion) DW conv
        k2   (int): Kernel size for the second (post-expansion) DW conv
        s    (int): Stride (2 for downsampling P levels, 1 otherwise)
        expand_ratio (int): Channel expansion factor (default 4 per MNv4)
    """
    def __init__(self, c1, c2, k1=3, k2=5, s=1, expand_ratio=4):
        super().__init__()
        mid = c1 * expand_ratio

        # Optional first DW conv (before expansion)
        # k1 provides initial spatial context before channel expansion
        self.dw1 = CBS(c1, c1, k1, 1, g=c1)   # groups=c1 -> depthwise

        # Pointwise expansion: c1 -> mid channels
        self.pw_expand = CBS(c1, mid, 1, 1)

        # Second DW conv (between expansion and projection)
        # k2 operates on the expanded feature space
        # stride=s here handles spatial downsampling
        self.dw2 = CBS(mid, mid, k2, s, g=mid)  # groups=mid -> depthwise

        # Pointwise projection: mid -> c2 channels
        # Note: no activation after final projection (linear bottleneck)
        self.pw_proj = nn.Sequential(
            nn.Conv2d(mid, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2)
        )

        # Residual shortcut only when resolution and channels are preserved
        self.use_shortcut = (s == 1 and c1 == c2)

    def forward(self, x):
        out = self.dw1(x)
        out = self.pw_expand(out)
        out = self.dw2(out)
        out = self.pw_proj(out)
        if self.use_shortcut:
            out = out + x
        return out


# ============================================================
# 3. ConvNext — ConvNext-Like Block
# ============================================================
class ConvNext(nn.Module):
    """
    ConvNext-Like block as used in MobileNetV4.
    Based on Liu et al. "A ConvNet for the 2020s" (ConvNext, 2022).

    Key difference from IB:
        IB: PW-expand -> DW-spatial -> PW-project
            (spatial mixing AFTER expansion = on expanded, wider features)
        ConvNext: DW-spatial -> PW-expand -> PW-project
            (spatial mixing BEFORE expansion = on narrower input features)

    This is cheaper because the large-kernel DW conv operates on
    fewer channels (c1 instead of c1*expand). The paper shows this
    is preferred in later network stages where "preceding layers have
    already conducted substantial spatial mixing" — exactly where
    ConvNext blocks appear in MNv4-Conv-Medium (layers 9, 11 at P4
    and layer 17, 22 at P5).

    For rice quality: the large depthwise kernel (7x7 default in original
    ConvNext, 3x5 in MNv4) captures long-range spatial patterns — useful
    for detecting elongated impurity structures like straw fragments.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels (usually c1 == c2 for in-place mixing)
        k  (int): Depthwise kernel size (3 or 5 in MNv4)
        s  (int): Stride (1 in all MNv4 ConvNext blocks)
    """
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        expand = c2 * 4

        # DW spatial mixing FIRST (on narrow c1 channels)
        self.dw = CBS(c1, c1, k, s, g=c1)

        # Expand channels
        self.pw1 = CBS(c1, expand, 1, 1)

        # Project back to c2
        self.pw2 = nn.Sequential(
            nn.Conv2d(expand, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.use_shortcut = (s == 1 and c1 == c2)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw1(out)
        out = self.pw2(out)
        if self.use_shortcut:
            out = out + x
        return out


# ============================================================
# 4. FFN — Feed-Forward Network
# ============================================================
class FFN(nn.Module):
    """
    Feed-Forward Network (FFN) block.
    Equivalent to the FFN block in Vision Transformers (ViTs),
    adapted for convolutional networks via 1x1 pointwise convolutions.

    Structure:
        1x1 Conv (expand) -> Activation -> 1x1 Conv (project)

    NO spatial mixing — purely operates on the channel dimension.
    This makes it:
        - Very accelerator-friendly (1x1 convs = dense matrix multiply)
        - Memory-efficient (no spatial sampling patterns)
        - Fast on both CPUs and specialized accelerators

    From MNv4 paper: "PW [pointwise] is very accelerator-friendly but
    works best with other blocks." FFN appears in later stages where
    spatial context has already been established by preceding ExtraDW
    and ConvNext blocks.

    For rice quality: at P4 (layer 10) and P5 (layers 16, 20, 21),
    the spatial structure of grain regions is already captured —
    FFN re-weights which channels (feature types) are most important
    for distinguishing grain quality categories.

    Args:
        c (int): Both input and output channels (FFN preserves dimensions)
        expand_ratio (int): Expansion factor for hidden dimension (default 4)
    """
    def __init__(self, c, expand_ratio=2):
        super().__init__()
        mid = int(c * expand_ratio)

        self.pw1 = CBS(c, mid, 1, 1)

        self.pw2 = nn.Sequential(
            nn.Conv2d(mid, c, 1, 1, bias=False),
            nn.BatchNorm2d(c)
        )

    def forward(self, x):
        # Always has a residual connection (same in/out dims)
        return x + self.pw2(self.pw1(x))


# ============================================================
# Registration helper
# ============================================================
# Add this import to ultralytics/nn/tasks.py:
#
#   from ultralytics.nn.modules.mnv4_modules import FusedIB, ExtraDW, ConvNext, FFN
#
# Then in the parse_model() function, add these modules to the
# channel-handling elif block (alongside AKConv, VoV_GSCSP, etc.):
#
#   elif m in (FusedIB, ExtraDW, ConvNext):
#       c1, c2 = ch[f], args[0]
#       args = [c1, c2, *args[1:]]
#
#   elif m is FFN:
#       c1 = ch[f]
#       args = [c1]
#
# ============================================================
