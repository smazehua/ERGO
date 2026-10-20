"""Zero123-XL SDS: nearest reference for coarse, per-reference maps for TAO."""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import DDIMScheduler
from zero123 import Zero123Pipeline
from ergo.texture import sds_surrogate_map
from cameras import ELEVATIONS, AZIMUTHS


class Zero123:
    def __init__(self, model, references):
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.pipe = Zero123Pipeline.from_pretrained(
            str(model),
            torch_dtype=self.dtype,
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device)
        for module in (
            self.pipe.image_encoder,
            self.pipe.vae,
            self.pipe.unet,
            self.pipe.clip_camera_projection,
        ):
            module.eval().requires_grad_(False)
        self.vae = self.pipe.vae
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.embeddings = []
        with torch.no_grad():
            for image in references:
                image = F.interpolate(
                    image, (256, 256), mode="bilinear", align_corners=False
                )
                clip = self.pipe.feature_extractor(
                    images=[TF.to_pil_image(i) for i in image], return_tensors="pt"
                ).pixel_values.to(device=self.device, dtype=self.dtype)
                c = self.pipe.image_encoder(clip).image_embeds
                v = self.encode(image.to(self.dtype)) / self.vae.config.scaling_factor
                self.embeddings.append((c, v))

    def encode(self, image):
        return (
            self.vae.encode(2 * image - 1).latent_dist.sample()
            * self.vae.config.scaling_factor
        )

    def loss_maps(self, image, elevation, azimuth, step_ratio, indices):
        latent = self.encode(
            F.interpolate(image, (256, 256), mode="bilinear", align_corners=False).to(
                self.dtype
            )
        )
        count = self.scheduler.config.num_train_timesteps
        value = np.round((1 - step_ratio) * count).clip(
            int(0.02 * count), int(0.98 * count)
        )
        timestep = torch.full((1,), value, dtype=torch.long, device=self.device)
        gradients = []
        with torch.no_grad():
            noise = torch.randn_like(latent)
            noisy = self.scheduler.add_noise(latent, noise, timestep)
            for k in indices:
                delta_a = np.deg2rad(azimuth - AZIMUTHS[k])
                camera = torch.tensor(
                    [
                        [
                            [
                                np.deg2rad(elevation - ELEVATIONS[k]),
                                np.sin(delta_a),
                                np.cos(delta_a),
                                0.0,
                            ]
                        ]
                    ],
                    device=self.device,
                    dtype=self.dtype,
                )
                clip, vae = self.embeddings[k]
                condition = self.pipe.clip_camera_projection(
                    torch.cat([clip.repeat(1, 1, 1), camera], dim=-1)
                )
                condition = torch.cat([condition, torch.zeros_like(condition)])
                vae = torch.cat([vae, torch.zeros_like(vae)])
                prediction = self.pipe.unet(
                    torch.cat([torch.cat([noisy] * 2), vae], dim=1),
                    torch.cat([timestep] * 2).to(self.dtype),
                    encoder_hidden_states=condition,
                ).sample
                conditional, unconditional = prediction.chunk(2)
                guided = unconditional + 5 * (conditional - unconditional)
                gradients.append(
                    torch.nan_to_num(
                        (1 - self.alphas[timestep]).view(1, 1, 1, 1) * (guided - noise)
                    )
                )
        return [sds_surrogate_map(latent, gradient) for gradient in gradients]
