from typing import Dict, Optional, Tuple, List, Union, Any, Literal, NamedTuple, Optional

from functools import cached_property, partial
from collections import UserList

from pydantic import (
    BaseModel,
    Extra,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    confloat,
    conint,
    validator,
)

import gradio as gr

from modules.processing import StableDiffusionProcessing

class ADetailerUnitSchema(BaseModel):
    ad_enabled: bool = False
    ad_skip_img2img: bool = False
    ad_model: str = "None"
    ad_model_classes: str = ""
    ad_prompt: str = ""
    ad_negative_prompt: str = ""
    ad_detection_confidence_threshold: confloat(ge=0.0, le=1.0) = 0.3
    ad_mask_min_ratio: confloat(ge=0.0, le=1.0) = 0.0
    ad_mask_max_ratio: confloat(ge=0.0, le=1.0) = 1.0
    ad_mask_k_largest: NonNegativeInt = 0
    ad_mask_x_offset: int = 0
    ad_mask_y_offset: int = 0
    ad_mask_erosion_dilation: int = 4
    ad_mask_merge_mode: Literal["None", "Merge", "Merge and Invert"] = "None"
    ad_inpaint_mask_blur: NonNegativeInt = 4
    ad_inpaint_mask_denoising: confloat(ge=0.0, le=1.0) = 0.4
    ad_inpaint_only_masked: bool = True
    ad_inpaint_only_masked_padding: NonNegativeInt = 32
    ad_use_separate_width_height: bool = False
    ad_inpaint_width: PositiveInt = 512
    ad_inpaint_height: PositiveInt = 512
    ad_use_separate_steps: bool = False
    ad_steps: PositiveInt = 28
    ad_use_separate_cfg_scale: bool = False
    ad_cfg_scale: NonNegativeFloat = 7.0
    ad_use_separate_checkpoint: bool = False
    ad_checkpoint: Optional[str] = None
    ad_use_separate_vae: bool = False
    ad_vae: Optional[str] = None
    ad_use_separate_sampler: bool = False
    ad_sampler: str = "DPM++ 2M Karras"
    ad_use_separate_scheduler: bool = False
    ad_scheduler: str = "Normal"
    ad_use_separate_noise_multiplier: bool = False
    ad_noise_multiplier: confloat(ge=0.5, le=1.5) = 1.0
    ad_use_separate_clip_skip: bool = False
    ad_clip_skip: conint(ge=1, le=12) = 1
    ad_use_restore_face_after_adetailer: bool = False
    ad_controlnet_model: str = "None"
    ad_controlnet_module: str = "None"
    ad_controlnet_guidance_start: confloat(ge=0.0, le=1.0) = 0
    ad_controlnet_guidance_end: confloat(ge=0.0, le=1.0) = 1.0
    ad_controlnet_weight: confloat(ge=0.0, le=1.0) = 1.0
    is_api: bool = True

class ADetailerUnit:
    def __init__(
        self,
        enabled=False,
        skip_img2img=False,
        model="None",
        model_classes="",
        prompt='',
        negative_prompt='',
        detection_confidence_threshold=0.3,
        mask_min_ratio=0,
        mask_max_ratio=1,
        mask_k_largest=0,
        mask_x_offset=0,
        mask_y_offset=0,
        mask_erosion_dilation=4,
        mask_merge_mode="None",
        inpaint_mask_blur=4,
        inpaint_mask_denoising=0.4,
        inpaint_only_masked=True,
        inpaint_only_masked_padding=32,
        use_separate_width_height=False,
        inpaint_width=512,
        inpaint_height=512,
        use_separate_steps=False,
        steps=20,
        use_separate_cfg_scale=False,
        cfg_scale=7,
        use_separate_checkpoint=False,
        checkpoint="None",
        use_separate_vae=False,
        vae="None",
        use_separate_sampler=False,
        sampler="Eular",
        use_separate_scheduler=False,
        scheduler="Normal",
        use_separate_noise_multiplier=False,
        noise_multiplier=1,
        use_separate_clip_skip=False,
        clip_skip=1,
        use_restore_face_after_adetailer=False,
        controlnet_model="None",
        controlnet_module="None",
        controlnet_guidance_start=0,
        controlnet_guidance_end=1,
        controlnet_weight=1
    ):
        self.ad_model = model
        self.ad_enabled = enabled
        self.ad_model_classes = model_classes
        self.ad_skip_img2img = skip_img2img
        self.ad_prompt = prompt
        self.ad_negative_prompt = negative_prompt
        self.ad_detection_confidence_threshold = detection_confidence_threshold
        self.ad_mask_min_ratio = mask_min_ratio
        self.ad_mask_max_ratio = mask_max_ratio
        self.ad_mask_k_largest = mask_k_largest
        self.ad_mask_x_offset = mask_x_offset
        self.ad_mask_y_offset = mask_y_offset
        self.ad_mask_erosion_dilation = mask_erosion_dilation
        self.ad_mask_merge_mode = mask_merge_mode
        self.ad_inpaint_mask_blur = inpaint_mask_blur
        self.ad_inpaint_mask_denoising = inpaint_mask_denoising
        self.ad_inpaint_only_masked = inpaint_only_masked
        self.ad_inpaint_only_masked_padding = inpaint_only_masked_padding
        self.ad_use_separate_width_height = use_separate_width_height
        self.ad_inpaint_width = inpaint_width
        self.ad_inpaint_height = inpaint_height
        self.ad_use_separate_steps = use_separate_steps
        self.ad_steps = steps
        self.ad_use_separate_cfg_scale = use_separate_cfg_scale
        self.ad_cfg_scale = cfg_scale
        self.ad_use_separate_checkpoint = use_separate_checkpoint
        self.ad_checkpoint = checkpoint
        self.ad_use_separate_vae = use_separate_vae
        self.ad_vae = vae
        self.ad_use_separate_sampler = use_separate_sampler
        self.ad_sampler = sampler
        self.ad_use_separate_scheduler = use_separate_scheduler
        self.ad_scheduler = scheduler
        self.ad_use_separate_noise_multiplier = use_separate_noise_multiplier
        self.ad_noise_multiplier = noise_multiplier
        self.ad_use_separate_clip_skip = use_separate_clip_skip
        self.ad_clip_skip = clip_skip
        self.ad_use_restore_face_after_adetailer = use_restore_face_after_adetailer
        self.ad_controlnet_model = controlnet_model
        self.ad_controlnet_module = controlnet_module
        self.ad_controlnet_guidance_start = controlnet_guidance_start
        self.ad_controlnet_guidance_end = controlnet_guidance_end
        self.ad_controlnet_weight = controlnet_weight

    def extra_params(self, suffix: str = "") -> dict[str, Any]:
        if self.ad_model == "None":
            return {}

        p = {name: getattr(self, attr) for attr, name in ALL_ARGS}
        ppop = partial(self.ppop, p)

        ppop("ADetailer model classes")
        ppop("ADetailer prompt")
        ppop("ADetailer negative prompt")
        ppop("ADetailer mask only top k largest", cond=0)
        ppop("ADetailer mask min ratio", cond=0.0)
        ppop("ADetailer mask max ratio", cond=1.0)
        ppop("ADetailer x offset", cond=0)
        ppop("ADetailer y offset", cond=0)
        ppop("ADetailer mask merge invert", cond="None")
        ppop("ADetailer inpaint only masked", ["ADetailer inpaint padding"])
        ppop(
            "ADetailer use inpaint width height",
            [
                "ADetailer use inpaint width height",
                "ADetailer inpaint width",
                "ADetailer inpaint height",
            ],
        )
        ppop(
            "ADetailer use separate steps",
            ["ADetailer use separate steps", "ADetailer steps"],
        )
        ppop(
            "ADetailer use separate CFG scale",
            ["ADetailer use separate CFG scale", "ADetailer CFG scale"],
        )
        ppop(
            "ADetailer use separate checkpoint",
            ["ADetailer use separate checkpoint", "ADetailer checkpoint"],
        )
        ppop(
            "ADetailer use separate VAE",
            ["ADetailer use separate VAE", "ADetailer VAE"],
        )
        ppop(
            "ADetailer use separate sampler",
            ["ADetailer use separate sampler", "ADetailer sampler"],
        )
        ppop(
            "ADetailer use separate scheduler",
            ["ADetailer use separate scheduler", "ADetailer scheduler"],
        )
        ppop(
            "ADetailer use separate noise multiplier",
            ["ADetailer use separate noise multiplier", "ADetailer noise multiplier"],
        )

        ppop(
            "ADetailer use separate CLIP skip",
            ["ADetailer use separate CLIP skip", "ADetailer CLIP skip"],
        )

        ppop("ADetailer restore face")
        ppop(
            "ADetailer ControlNet model",
            [
                "ADetailer ControlNet model",
                "ADetailer ControlNet module",
                "ADetailer ControlNet weight",
                "ADetailer ControlNet guidance start",
                "ADetailer ControlNet guidance end",
            ],
            cond="None",
        )
        ppop("ADetailer ControlNet module", cond="None")
        ppop("ADetailer ControlNet weight", cond=1.0)
        ppop("ADetailer ControlNet guidance start", cond=0.0)
        ppop("ADetailer ControlNet guidance end", cond=1.0)

        if suffix:
            p = {k + suffix: v for k, v in p.items()}

        return p

    @staticmethod
    def ppop(
        p: dict[str, Any],
        key: str,
        pops: list[str] | None = None,
        cond: Any = None,
    ) -> None:
        if pops is None:
            pops = [key]
        if key not in p:
            return
        value = p[key]
        cond = (not bool(value)) if cond is None else value == cond

        if cond:
            for k in pops:
                p.pop(k, None)

    @staticmethod
    def from_dict(d: Dict) -> "ADetailerUnit":
        return ADetailerUnit(
            **{k: v for k, v in d.items() if k in vars(ADetailerUnit)}
        )

    @staticmethod
    def infotext_fields():
        return (
            "ad_enabled",
            "ad_skip_img2img",
            "ad_model",
            "ad_model_classes",
            "ad_prompt",
            "ad_negative_prompt",
            "ad_detection_confidence_threshold",
            "ad_mask_min_ratio",
            "ad_mask_max_ratio",
            "ad_mask_k_largest",
            "ad_mask_x_offset",
            "ad_mask_y_offset",
            "ad_mask_erosion_dilation",
            "ad_mask_merge_mode",
            "ad_inpaint_mask_blur",
            "ad_inpaint_mask_denoising",
            "ad_inpaint_only_masked",
            "ad_inpaint_only_masked_padding",
            "ad_use_separate_width_height",
            "ad_inpaint_width",
            "ad_inpaint_height",
            "ad_use_separate_steps",
            "ad_steps",
            "ad_use_separate_cfg_scale",
            "ad_cfg_scale",
            "ad_use_separate_checkpoint",
            "ad_checkpoint",
            "ad_use_separate_vae",
            "ad_vae",
            "ad_use_separate_sampler",
            "ad_sampler",
            "ad_use_separate_scheduler",
            "ad_scheduler",
            "ad_use_separate_noise_multiplier",
            "ad_noise_multiplier",
            "ad_use_separate_clip_skip",
            "ad_clip_skip",
            "ad_use_restore_face_after_adetailer",
            "ad_controlnet_model",
            "ad_controlnet_module",
            "ad_controlnet_guidance_start",
            "ad_controlnet_guidance_end",
            "ad_controlnet_weight"
        )

class Arg(NamedTuple):
    attr: str
    name: str

class ArgsList(UserList):
    @cached_property
    def attrs(self) -> tuple[str, ...]:
        return tuple(attr for attr, _ in self)

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(name for _, name in self)

class PromptSR(NamedTuple):
    s: str
    r: str

class WebuiInfo:
    ad_model_list: list[str]
    sampler_names: list[str]
    scheduler_names: list[str]
    t2i_button: gr.Button
    i2i_button: gr.Button
    checkpoints_list: list[str]
    vae_list: list[str]
    controlnet_model_list: list[str]
    preprocessors_list: list[str]

_all_args = [
    ("ad_model", "ADetailer model"),
    ("ad_model_classes", "ADetailer model classes"),
    ("ad_prompt", "ADetailer prompt"),
    ("ad_negative_prompt", "ADetailer negative prompt"),
    ("ad_detection_confidence_threshold", "ADetailer confidence"),
    ("ad_mask_k_largest", "ADetailer mask only top k largest"),
    ("ad_mask_min_ratio", "ADetailer mask min ratio"),
    ("ad_mask_max_ratio", "ADetailer mask max ratio"),
    ("ad_mask_x_offset", "ADetailer x offset"),
    ("ad_mask_y_offset", "ADetailer y offset"),
    ("ad_mask_erosion_dilation", "ADetailer dilate erode"),
    ("ad_mask_merge_mode", "ADetailer mask merge invert"),
    ("ad_inpaint_mask_blur", "ADetailer mask blur"),
    ("ad_inpaint_mask_denoising", "ADetailer denoising strength"),
    ("ad_inpaint_only_masked", "ADetailer inpaint only masked"),
    ("ad_inpaint_only_masked_padding", "ADetailer inpaint padding"),
    ("ad_use_separate_width_height", "ADetailer use inpaint width height"),
    ("ad_inpaint_width", "ADetailer inpaint width"),
    ("ad_inpaint_height", "ADetailer inpaint height"),
    ("ad_use_separate_steps", "ADetailer use separate steps"),
    ("ad_steps", "ADetailer steps"),
    ("ad_use_separate_cfg_scale", "ADetailer use separate CFG scale"),
    ("ad_cfg_scale", "ADetailer CFG scale"),
    ("ad_use_separate_checkpoint", "ADetailer use separate checkpoint"),
    ("ad_checkpoint", "ADetailer checkpoint"),
    ("ad_use_separate_vae", "ADetailer use separate VAE"),
    ("ad_vae", "ADetailer VAE"),
    ("ad_use_separate_sampler", "ADetailer use separate sampler"),
    ("ad_sampler", "ADetailer sampler"),
    ("ad_use_separate_scheduler", "ADetailer use separate scheduler"),
    ("ad_scheduler", "ADetailer scheduler"),
    ("ad_use_separate_noise_multiplier", "ADetailer use separate noise multiplier"),
    ("ad_noise_multiplier", "ADetailer noise multiplier"),
    ("ad_use_separate_clip_skip", "ADetailer use separate CLIP skip"),
    ("ad_clip_skip", "ADetailer CLIP skip"),
    ("ad_use_restore_face_after_adetailer", "ADetailer restore face"),
    ("ad_controlnet_model", "ADetailer ControlNet model"),
    ("ad_controlnet_module", "ADetailer ControlNet module"),
    ("ad_controlnet_weight", "ADetailer ControlNet weight"),
    ("ad_controlnet_guidance_start", "ADetailer ControlNet guidance start"),
    ("ad_controlnet_guidance_end", "ADetailer ControlNet guidance end"),
]

_args = [Arg(*args) for args in _all_args]
ALL_ARGS = ArgsList(_args)

BBOX_SORTBY = [
    "None",
    "Position (left to right)",
    "Position (center to edge)",
    "Area (large to small)",
]

MASK_MERGE_INVERT = ["None", "Merge", "Merge and Invert"]

SCRIPT_DEFAULT = "dynamic_prompting,dynamic_thresholding,wildcard_recursive,wildcards,lora_block_weight,negpip,soft_inpainting"