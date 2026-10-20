# Copyright 2023 Hillbot
# SPDX-License-Identifier: Apache-2.0
# Adapted from Zero123++; modified for paired DDIM inversion and refinement.
# See THIRD_PARTY_NOTICES.md and LICENSES/Zero123Plus-Apache-2.0.txt.

from diffusers.utils.torch_utils import randn_tensor
from typing import Any, Optional
import diffusers
import numpy
import torch
import transformers
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.schedulers import KarrasDiffusionSchedulers
from zero123plus_refine.generator_utils import normalize_generator
from zero123plus_refine.ddim_inversion import make_ddim_plan, invert_latents


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == "RGB":
        return maybe_rgba
    elif maybe_rgba.mode == "RGBA":
        rgba = maybe_rgba
        img = numpy.random.randint(
            127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8
        )
        img = Image.fromarray(img, "RGB")
        img.paste(rgba, mask=rgba.getchannel("A"))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


class ReferenceOnlyAttnProc(torch.nn.Module):

    def __init__(self, chained_proc, enabled=False, name=None) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        mode="w",
        ref_dict: dict = None,
        is_cfg_guidance=False,
    ) -> Any:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled and is_cfg_guidance:
            res0 = self.chained_proc(
                attn, hidden_states[:1], encoder_hidden_states[:1], attention_mask
            )
            hidden_states = hidden_states[1:]
            encoder_hidden_states = encoder_hidden_states[1:]
        if self.enabled:
            if mode == "w":
                ref_dict[self.name] = encoder_hidden_states
            elif mode == "r":
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, ref_dict.pop(self.name)], dim=1
                )
            else:
                assert False, mode
        res = self.chained_proc(
            attn, hidden_states, encoder_hidden_states, attention_mask
        )
        if self.enabled and is_cfg_guidance:
            res = torch.cat([res0, res])
        return res


class RefOnlyNoisedUNet(torch.nn.Module):

    def __init__(self, unet: UNet2DConditionModel, val_sched) -> None:
        super().__init__()
        self.unet = unet
        self.val_sched = val_sched
        unet_lora_attn_procs = dict()
        for name, _ in unet.attn_processors.items():
            default_attn_proc = AttnProcessor2_0()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=name.endswith("attn1.processor"), name=name
            )
        unet.set_attn_processor(unet_lora_attn_procs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(
        self,
        noisy_cond_lat,
        timestep,
        encoder_hidden_states,
        class_labels,
        ref_dict,
        is_cfg_guidance,
        **kwargs
    ):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]
        self.unet(
            noisy_cond_lat,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        )

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels=None,
        *args,
        cross_attention_kwargs,
        down_block_res_samples=None,
        mid_block_res_sample=None,
        **kwargs
    ):
        cond_lat = cross_attention_kwargs["cond_lat"]
        generator = cross_attention_kwargs.get("generator")
        is_cfg_guidance = cross_attention_kwargs.get("is_cfg_guidance", False)
        noise = cross_attention_kwargs.get("condition_noise")
        if noise is None:
            noise = (
                torch.randn_like(cond_lat)
                if generator is None
                else randn_tensor(
                    cond_lat.shape,
                    generator=generator,
                    device=cond_lat.device,
                    dtype=cond_lat.dtype,
                    layout=cond_lat.layout,
                )
            )
        noisy_cond_lat = self.val_sched.add_noise(cond_lat, noise, timestep.reshape(-1))
        noisy_cond_lat = self.val_sched.scale_model_input(
            noisy_cond_lat, timestep.reshape(-1)
        )
        ref_dict = {}
        self.forward_cond(
            noisy_cond_lat,
            timestep,
            encoder_hidden_states,
            class_labels,
            ref_dict,
            is_cfg_guidance,
            **kwargs
        )
        weight_dtype = self.unet.dtype
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            *args,
            class_labels=class_labels,
            cross_attention_kwargs=dict(
                mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance
            ),
            down_block_additional_residuals=(
                [sample.to(dtype=weight_dtype) for sample in down_block_res_samples]
                if down_block_res_samples is not None
                else None
            ),
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None
                else None
            ),
            **kwargs
        )


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


class Zero123PlusPipeline(diffusers.StableDiffusionPipeline):
    tokenizer: transformers.CLIPTokenizer
    text_encoder: transformers.CLIPTextModel
    vision_encoder: transformers.CLIPVisionModelWithProjection
    feature_extractor_clip: transformers.CLIPImageProcessor
    unet: UNet2DConditionModel
    scheduler: diffusers.schedulers.KarrasDiffusionSchedulers
    vae: AutoencoderKL
    feature_extractor_vae: transformers.CLIPImageProcessor

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vision_encoder: transformers.CLIPVisionModelWithProjection,
        feature_extractor_clip: CLIPImageProcessor,
        feature_extractor_vae: CLIPImageProcessor,
        ramping_coefficients: Optional[list] = None,
        safety_checker=None,
    ):
        DiffusionPipeline.__init__(self)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae,
        )
        self.register_to_config(ramping_coefficients=ramping_coefficients)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.set_progress_bar_config(disable=True)

    def prepare(self):
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = RefOnlyNoisedUNet(self.unet, self.scheduler).eval()

    def encode_condition_image(
        self, image: torch.Tensor, generator: Optional[torch.Generator] = None
    ):
        image = self.vae.encode(image).latent_dist.sample(generator=generator)
        return image

    def guided_embeddings(
        self, encoder_hidden_states, guidance_scale, num_images_per_prompt
    ):
        if guidance_scale <= 1:
            return encoder_hidden_states
        negative = self.encode_prompt("", self.device, num_images_per_prompt, False)[0]
        return torch.cat([negative, encoder_hidden_states])

    def predict_guided_noise(
        self, latents, timestep, embeddings, cross_attention_kwargs, guidance_scale
    ):
        model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
        model_input = self.scheduler.scale_model_input(model_input, timestep)
        prediction = self.unet(
            model_input.to(dtype=embeddings.dtype),
            timestep,
            encoder_hidden_states=embeddings,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        if guidance_scale > 1:
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + guidance_scale * (conditional - unconditional)
        return prediction.to(dtype=latents.dtype)

    def denoise_latents(
        self,
        latents,
        timesteps,
        encoder_hidden_states,
        cross_attention_kwargs,
        guidance_scale,
        generator,
        num_images_per_prompt,
    ):
        embeddings = self.guided_embeddings(
            encoder_hidden_states, guidance_scale, num_images_per_prompt
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta=0.0)
        for timestep in timesteps:
            prediction = self.predict_guided_noise(
                latents, timestep, embeddings, cross_attention_kwargs, guidance_scale
            )
            latents = self.scheduler.step(
                prediction, timestep, latents, **extra_step_kwargs, return_dict=False
            )[0]
            if not torch.isfinite(latents).all():
                raise FloatingPointError("nonfinite refinement latent")
        return latents

    def condition(self, image, generator, guidance_scale):
        image = to_rgb_image(image)
        vae_image = self.feature_extractor_vae(
            images=image, return_tensors="pt"
        ).pixel_values.to(device=self.vae.device, dtype=self.vae.dtype)
        clip_image = self.feature_extractor_clip(
            images=image, return_tensors="pt"
        ).pixel_values.to(device=self.vae.device, dtype=self.vae.dtype)
        cond_lat = self.encode_condition_image(vae_image, generator)
        if guidance_scale > 1:
            negative = self.encode_condition_image(
                torch.zeros_like(vae_image), generator
            )
            cond_lat = torch.cat([negative, cond_lat])
        global_embeds = self.vision_encoder(
            clip_image, output_hidden_states=False
        ).image_embeds.unsqueeze(-2)
        embeddings = self.encode_prompt("", self.device, 1, False)[0]
        ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
        return (cond_lat, embeddings + global_embeds * ramp)

    def decode_grid(self, latents):
        latents = unscale_latents(latents)
        image = unscale_image(
            self.vae.decode(
                (latents / self.vae.config.scaling_factor).to(self.vae.dtype),
                return_dict=False,
            )[0]
        )
        return self.image_processor.postprocess(image, output_type="pil")[0]

    @torch.no_grad()
    def generate(self, image, steps, generator):
        """Generate six 320px views with Euler ancestral sampling."""
        self.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.scheduler.config, timestep_spacing="trailing"
        )
        self.prepare()
        self.unet.val_sched = self.scheduler
        cond_lat, embeddings = self.condition(image, None, 4.0)
        latents = diffusers.StableDiffusionPipeline.__call__(
            self,
            None,
            cross_attention_kwargs={"cond_lat": cond_lat},
            guidance_scale=4.0,
            num_images_per_prompt=1,
            prompt_embeds=embeddings,
            num_inference_steps=steps,
            output_type="latent",
            width=640,
            height=960,
            generator=generator,
        ).images
        return self.decode_grid(latents)

    @torch.no_grad()
    def refine(self, image, coarse_grid, config, generator):
        """Partial DDIM inversion followed by matched deterministic denoising."""
        plan = make_ddim_plan(
            self.scheduler.config, config.steps, config.strength, self.device
        )
        self.scheduler = plan.forward
        self.prepare()
        self.unet.val_sched = self.scheduler
        generator = normalize_generator(generator, self.device)
        cond_lat, embeddings = self.condition(image, generator, config.guidance_scale)
        condition_noise = randn_tensor(
            cond_lat.shape,
            generator=generator,
            device=cond_lat.device,
            dtype=cond_lat.dtype,
        )
        kwargs = dict(
            cond_lat=cond_lat, generator=generator, condition_noise=condition_noise
        )
        coarse_grid = coarse_grid.convert("RGB")
        if coarse_grid.size != (640, 960):
            raise ValueError("Expected a native 640x960 coarse render grid")
        image_tensor = self.image_processor.preprocess(coarse_grid)
        clean = self.vae.encode(
            scale_image(image_tensor).to(device=self.vae.device, dtype=self.vae.dtype)
        ).latent_dist.mode()
        clean = scale_latents(clean.float() * self.vae.config.scaling_factor)
        inverse_kwargs = dict(
            kwargs, cond_lat=cond_lat[-1:], condition_noise=condition_noise[-1:]
        )
        latents = invert_latents(
            clean,
            plan,
            lambda value, time: self.predict_guided_noise(
                value, time, embeddings, inverse_kwargs, config.inversion_guidance_scale
            ),
        )
        latents = self.denoise_latents(
            latents,
            plan.denoising_timesteps,
            embeddings,
            kwargs,
            config.guidance_scale,
            generator,
            1,
        )
        return self.decode_grid(latents)
