"""Bounded temporal execution of the InfVSR causal VAE."""

import torch

from .wan_video_vae import count_conv3d


class VAESequence:
    """Own encoder and decoder histories for one video and its spatial tiles."""

    def __init__(self, vae, height, width, device, tile_size=128, tile_stride=110):
        self.vae = vae
        self.device = device
        self.height, self.width = height, width
        self.tiled = height > 1024
        self.tile_size, self.tile_stride = tile_size, tile_stride
        self.enc_history = {}
        self.dec_history = {}
        self.first_input = True
        self.first_output = True
        self.enc_slots = count_conv3d(vae.model.encoder)
        self.dec_slots = count_conv3d(vae.model.decoder)

    def _regions(self):
        height, width = self.height // 8, self.width // 8
        if not self.tiled:
            yield 0, height, 0, width
            return
        size, stride = self.tile_size, self.tile_stride
        for top in range(0, height, stride):
            if top >= stride and top - stride + size >= height:
                continue
            for left in range(0, width, stride):
                if left >= stride and left - stride + size >= width:
                    continue
                yield top, min(top + size, height), left, min(left + size, width)

    def _history(self, store, region, slots):
        state = store.setdefault(region, [None] * slots)
        if self.tiled:
            state = [v.to(self.device) if torch.is_tensor(v) else v for v in state]
        return state

    def _remember(self, store, region, state):
        # Inactive tiles keep their convolution histories in host memory.
        store[region] = (
            [v.cpu() if torch.is_tensor(v) else v for v in state]
            if self.tiled
            else state
        )

    def _encode_region(self, pixels, history):
        features = []
        cursor = 0
        if self.first_input:
            features.append(
                self.vae.model.encoder(
                    pixels[:, :, :1], feat_cache=history, feat_idx=[0]
                )
            )
            cursor = 1
        for index in range(cursor, pixels.shape[2], 4):
            features.append(
                self.vae.model.encoder(
                    pixels[:, :, index : index + 4], feat_cache=history, feat_idx=[0]
                )
            )
        mean = self.vae.model.conv1(torch.cat(features, dim=2)).chunk(2, dim=1)[0]
        offset, inverse_std = [
            v.to(mean.device, mean.dtype).view(1, -1, 1, 1, 1) for v in self.vae.scale
        ]
        return (mean - offset) * inverse_std

    def _decode_region(self, latents, history):
        offset, inverse_std = [
            v.to(latents.device, latents.dtype).view(1, -1, 1, 1, 1)
            for v in self.vae.scale
        ]
        features = self.vae.model.conv2(latents / inverse_std + offset)
        frames = [
            self.vae.model.decoder(
                features[:, :, index : index + 1], feat_cache=history, feat_idx=[0]
            )
            for index in range(features.shape[2])
        ]
        return torch.cat(frames, dim=2)

    def _apply(self, value, decode):
        # Preserve the original wrapper's per-video layout for BF16 kernels.
        value = value[0].unsqueeze(0)
        history_store = self.dec_history if decode else self.enc_history
        slots = self.dec_slots if decode else self.enc_slots
        operation = self._decode_region if decode else self._encode_region
        if not self.tiled:
            region = next(self._regions())
            history = self._history(history_store, region, slots)
            result = operation(value.to(self.device), history)
            self._remember(history_store, region, history)
            return result.clamp_(-1, 1).contiguous() if decode else result.contiguous()

        length = (
            value.shape[2] * 4 - (3 if self.first_output else 0)
            if decode
            else (value.shape[2] + 3) // 4
        )
        factor = 8 if decode else 1
        channels = 3 if decode else self.vae.model.z_dim
        values = torch.zeros(
            (1, channels, length, self.height // 8 * factor, self.width // 8 * factor),
            dtype=value.dtype,
            device="cpu",
        )
        weights = torch.zeros(
            (1, 1, 1, *values.shape[-2:]), dtype=value.dtype, device="cpu"
        )
        for region in self._regions():
            top, bottom, left, right = region
            input_factor = 1 if decode else 8
            patch = value[
                :,
                :,
                :,
                top * input_factor : bottom * input_factor,
                left * input_factor : right * input_factor,
            ]
            history = self._history(history_store, region, slots)
            result = operation(patch.to(self.device), history).cpu()
            self._remember(history_store, region, history)
            overlap = (self.tile_size - self.tile_stride) * factor
            mask = self.vae.build_mask(
                result,
                (
                    top == 0,
                    bottom == self.height // 8,
                    left == 0,
                    right == self.width // 8,
                ),
                (overlap, overlap),
            ).to(value.dtype)
            row = slice(top * factor, bottom * factor)
            col = slice(left * factor, right * factor)
            values[:, :, :, row, col] += result * mask
            weights[:, :, :, row, col] += mask
        values /= weights
        return values.clamp_(-1, 1) if decode else values

    def encode(self, pixels):
        expected = 1 if self.first_input else 0
        if pixels.shape[2] % 4 != expected:
            raise ValueError("Pixel chunk is not aligned with the causal VAE timeline")
        result = self._apply(pixels, decode=False)
        self.first_input = False
        return result

    def decode(self, latents):
        result = self._apply(latents, decode=True)
        self.first_output = False
        return result
