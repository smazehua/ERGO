# Zero123 component container, derived from the Apache-2.0 Hugging Face pipeline.
# Copyright 2023 The HuggingFace Team. Licensed under Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# Modified for ERGO component registration and compatibility handling.
import torch
from diffusers.configuration_utils import ConfigMixin, FrozenDict, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from packaging import version
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class CLIPCameraProjection(ModelMixin, ConfigMixin):
    """Project concatenated CLIP and camera features from embedding_dim + additional_embeddings to embedding_dim."""

    @register_to_config
    def __init__(self, embedding_dim: int = 768, additional_embeddings: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.additional_embeddings = additional_embeddings
        self.input_dim = self.embedding_dim + self.additional_embeddings
        self.output_dim = self.embedding_dim
        self.proj = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, embedding: torch.FloatTensor):
        proj_embedding = self.proj(embedding)
        return proj_embedding


class Zero123Pipeline(DiffusionPipeline):
    _optional_components = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        clip_camera_projection: CLIPCameraProjection,
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        is_unet_version_less_0_9_0 = hasattr(
            unet.config, "_diffusers_version"
        ) and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse(
            "0.9.0.dev0"
        )
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            clip_camera_projection=clip_camera_projection,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
