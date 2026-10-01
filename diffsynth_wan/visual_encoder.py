"""DAPE visual features for InfVSR conditioning."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from .swin import SwinTransformer


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = SwinTransformer(
            img_size=384,
            patch_size=4,
            in_chans=3,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
        )
        self.image_proj = nn.Linear(1536, 512)

    def forward(self, image):
        return self.image_proj(self.visual_encoder(image))


def load_visual_encoder(ram_path, dape_path, device, dtype):
    model = VisualEncoder()
    source = torch.load(ram_path, map_location="cpu", weights_only=True, mmap=True)[
        "model"
    ]
    weights = {
        key: value
        for key, value in source.items()
        if key.startswith(("visual_encoder.", "image_proj."))
        and not key.endswith(("relative_position_index", "attn_mask"))
    }
    adapter = torch.load(dape_path, map_location="cpu", weights_only=True)
    for name, module in model.named_modules():
        if not name.endswith(".attn.qkv"):
            continue
        a = adapter[f"{name}.lora_A"].float()
        b = adapter[f"{name}.lora_B"].float()
        # DAPE adapts Q and V; the middle K projection is unchanged.
        delta = F.conv1d(a.unsqueeze(0), b.unsqueeze(-1), groups=2).squeeze(0)
        dim = module.in_features
        weight = weights[f"{name}.weight"].float().clone()
        weight[:dim] += delta[:dim] / 8
        weight[2 * dim :] += delta[dim:] / 8
        weights[f"{name}.weight"] = weight
    result = model.load_state_dict(weights, strict=False)
    missing = [
        key
        for key in result.missing_keys
        if not key.endswith(("relative_position_index", "attn_mask"))
    ]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Incompatible visual encoder weights: {missing}, {result.unexpected_keys}"
        )
    return model.eval().requires_grad_(False).to(device=device, dtype=dtype)


def prepare_reference(frame):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (384, 384), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(frame)
