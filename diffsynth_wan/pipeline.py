"""InfVSR inference pipeline."""

import json

import torch

from .video import pad_chunk, resize_video
from .vae_sequence import VAESequence
from .scenes import VideoChunk
from .visual_encoder import load_visual_encoder, prepare_reference
from .wan_video_dit import WanModel
from .wan_video_vae import WanVideoVAE


class InfVSR:
    def __init__(
        self, checkpoint, config, vae_path, ram_path, dape_path, device="cuda"
    ):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        with open(config) as handle:
            model_config = json.load(handle)["model_config"]
        self.dit = WanModel(**model_config)
        self.dit.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        )
        self.dit.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        self.vae = WanVideoVAE()
        vae_state = torch.load(vae_path, map_location="cpu", weights_only=True)
        self.vae.model.load_state_dict(vae_state.get("model_state", vae_state))
        self.vae.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        self.visual_encoder = load_visual_encoder(
            ram_path, dape_path, self.device, self.dtype
        )
        self.timestep = torch.tensor([399], device=self.device, dtype=self.dtype)
        sigmas = torch.linspace(1, 0, 1001)[:-1]
        sigmas = 5 * sigmas / (1 + 4 * sigmas)
        index = torch.argmin((sigmas * 1000 - self.timestep.cpu()).abs())
        self.sigma = sigmas[index].to(self.device)

    @torch.inference_mode()
    def __call__(self, chunks, scale=4):
        sequence = None
        cache = None
        position = 0
        input_size = None
        for chunk in chunks:
            reset = isinstance(chunk, VideoChunk) and chunk.new_scene
            update = isinstance(chunk, VideoChunk) and chunk.update_reference
            video = chunk.frames if isinstance(chunk, VideoChunk) else chunk
            if reset:
                sequence, cache, position = None, None, 0
            if video.ndim != 5 or video.shape[:2] != (1, 3):
                raise ValueError("Expected a [1, 3, T, H, W] video chunk")
            if input_size is None:
                input_size = video.shape[-2:]
            elif video.shape[-2:] != input_size:
                raise ValueError("Video resolution changed within a sequence")
            if sequence is None or update:
                reference = (video[0, :, 0] / 2 + 0.5).unsqueeze(0).to(self.device)
                context = self.visual_encoder(
                    prepare_reference(reference).to(self.dtype)
                )
            video = resize_video(video, scale).to(device=self.device, dtype=self.dtype)
            video, (length, height, width) = pad_chunk(video, first=sequence is None)
            if sequence is None:
                sequence = VAESequence(self.vae, *video.shape[-2:], self.device)
            latents = sequence.encode(video).to(self.device, self.dtype)
            del video
            prediction, cache = self.dit(
                latents,
                self.timestep,
                context,
                past_key_values=cache,
                use_cache=True,
                start_frame=position,
            )
            restored = latents + prediction * (-self.sigma)
            position += latents.shape[2]
            frames = sequence.decode(restored)
            yield frames[:, :, :length, :height, :width].cpu()
