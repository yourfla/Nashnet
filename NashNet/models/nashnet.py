import torch
from torch import nn

from .segmentation import UNet
from .reconstruction import VGG16Net
from .evidential_attention import SoftAtt
from .nash_equilibrium import NashEquilibrium


class NashNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.segmentation = UNet(resnet='resnet34', num_classes=2, pretrained=False)
        self.reconstruction = VGG16Net(num_classes=2)
        self.attention = SoftAtt()
        self.nash = NashEquilibrium(n_experts=6)

    def forward(self, images, weights=None):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError('Expected images with shape [B, 3, 256, 256].')
        if weights is None:
            weights = self.nash.compute_nash_equilibrium(
                batch_size=images.shape[0], device=images.device,
                temperature=1.0,
            )
            weights = weights.clamp(min=0.05, max=0.5)
            weights = weights / weights.sum(dim=1, keepdim=True)
        elif weights.shape != (images.shape[0], 6):
            raise ValueError('Expected weights with shape [B, 6].')
        elif not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Weights must be finite and nonnegative.')
        elif not torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0])):
            raise ValueError('Weights must sum to one for each image.')
        weights = weights.to(device=images.device, dtype=images.dtype)
        coarse, features = self.segmentation(images, weights)
        rater_logits = self.reconstruction(images, coarse, weights)
        logits, _ = self.attention(rater_logits, features, coarse, nash_weights=weights)
        return logits
