"""
Depthwise Separable Convolutional Neural Network (DS-CNN) for Keyword Spotting.

Architecture based on:
  Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers" (2018)

DS-CNN is the preferred architecture for on-device KWS because:
  1. Depthwise separable convolutions reduce parameters by ~8-9x vs standard convs
  2. Small memory footprint (< 200KB after INT8 quantization)
  3. Low latency (< 20ms inference on Raspberry Pi)
  4. Competitive accuracy with much larger models

Architecture:
  Input: (batch, 1, n_mfcc, time_steps) = (batch, 1, 40, 101)
  -> Standard Conv2D (captures initial spectro-temporal patterns)
  -> BatchNorm -> ReLU
  -> N x DepthwiseSeparable blocks
  -> Global Average Pooling
  -> FC -> Softmax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block.

    Factorizes a standard convolution into:
      1. Depthwise conv: one filter per input channel (spatial filtering)
      2. Pointwise conv: 1x1 conv to mix channels

    This reduces computation from O(K*K*C_in*C_out) to O(K*K*C_in + C_in*C_out).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0.0):
        super().__init__()
        # Depthwise: convolve each channel independently
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Key: groups=in_channels makes it depthwise
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise: 1x1 conv to combine channel information
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        x = self.dropout(x)
        return x


class DSCNN(nn.Module):
    """
    DS-CNN model for keyword spotting.

    Default configuration targets < 100K parameters (~200KB after quantization).
    """

    def __init__(
        self,
        n_classes=5,
        n_mfcc=40,
        # First standard conv
        first_filters=64,
        first_kernel=(4, 10),
        first_stride=(2, 2),
        # DS conv blocks
        ds_filters=(64, 64, 64, 64),
        ds_kernels=((3, 3), (3, 3), (3, 3), (3, 3)),
        dropout=0.2,
    ):
        super().__init__()

        # Initial standard convolution to capture broad spectro-temporal patterns
        # Padding calculated to maintain spatial dimensions after stride
        pad_h = (first_kernel[0] - 1) // 2
        pad_w = (first_kernel[1] - 1) // 2

        self.first_conv = nn.Sequential(
            nn.Conv2d(1, first_filters, kernel_size=first_kernel,
                      stride=first_stride, padding=(pad_h, pad_w), bias=False),
            nn.BatchNorm2d(first_filters),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Stack of depthwise separable conv blocks
        ds_layers = []
        in_ch = first_filters
        for filters, kernel in zip(ds_filters, ds_kernels):
            pad = (kernel[0] // 2, kernel[1] // 2)
            ds_layers.append(
                DepthwiseSeparableConv(
                    in_ch, filters,
                    kernel_size=kernel,
                    stride=1,
                    padding=pad,
                    dropout=dropout,
                )
            )
            in_ch = filters

        self.ds_blocks = nn.Sequential(*ds_layers)

        # Global average pooling + classifier
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, n_classes)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.
        Input: (batch, 1, n_mfcc, time_steps) e.g. (64, 1, 40, 101)
        Output: (batch, n_classes) logits
        """
        x = self.first_conv(x)     # (B, 64, ~20, ~50)
        x = self.ds_blocks(x)      # (B, 64, ~20, ~50)
        x = self.avg_pool(x)       # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        x = self.classifier(x)     # (B, n_classes)
        return x

    def predict_proba(self, x):
        """Get class probabilities (for inference)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def count_parameters(model: nn.Module) -> dict:
    """Count and report model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size
    # FP32: 4 bytes/param, INT8: 1 byte/param
    size_fp32_kb = total * 4 / 1024
    size_int8_kb = total * 1 / 1024

    info = {
        "total_params": total,
        "trainable_params": trainable,
        "size_fp32_kb": size_fp32_kb,
        "size_int8_kb": size_int8_kb,
    }

    print(f"Model Parameters:")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Size (FP32):  {size_fp32_kb:.1f} KB")
    print(f"  Size (INT8):  {size_int8_kb:.1f} KB")

    return info


if __name__ == "__main__":
    # Quick test: verify model architecture and size
    model = DSCNN(n_classes=5)
    info = count_parameters(model)

    # Test forward pass
    dummy_input = torch.randn(1, 1, 40, 101)
    output = model(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {F.softmax(output, dim=-1).detach().numpy()}")

    assert output.shape == (1, 5), f"Expected (1, 5), got {output.shape}"
    assert info["size_int8_kb"] < 200, f"Model too large: {info['size_int8_kb']:.1f} KB"
    print("\nAll checks passed!")
