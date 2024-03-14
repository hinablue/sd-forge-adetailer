from typing import List, Any
from functools import partial

import os
import cv2
import subprocess
import re
import gradio as gr

from fastapi import FastAPI

from modules import shared, sd_models, shared_items, paths, scripts
from modules.launch_utils import git
from modules.shared import cmd_opts
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img
from modules.sd_samplers import all_samplers

from lib_controlnet import external_code, global_state

from lib_adetailer import (
    AFTER_DETAILER,
    __version__,
    get_models,
    mediapipe_predict,
    ultralytics_predict,
)

from lib_adetailer.args import ADetailerUnit, ADetailerUnitSchema, WebuiInfo, PromptSR
from lib_adetailer.process import (
    get_model_mapping,
    get_controlnet_models
)

from .logger import logger_adetailer as logger

class ADetailerUiGroup(object):
    txt2img_submit_button = None
    img2img_submit_button = None

    def __init__(
        self,
        is_img2img: bool,
        default_unit: ADetailerUnit,
    ):
        self.is_img2img = is_img2img
        self.default_unit = default_unit

        self.ad_enabled = False
        self.ad_skip_img2img = None
        self.ad_model = None
        self.ad_model_classes = ""
        self.ad_prompt = None
        self.ad_negative_prompt = None
        self.ad_detection_confidence_threshold = None
        self.ad_mask_min_ratio = None
        self.ad_mask_max_ratio = None
        self.ad_mask_k_largest = None
        self.ad_mask_x_offset = None
        self.ad_mask_y_offset = None
        self.ad_mask_erosion_dilation = None
        self.ad_mask_merge_mode = None
        self.ad_inpaint_mask_blur = None
        self.ad_inpaint_mask_denoising = None
        self.ad_inpaint_only_masked = None
        self.ad_inpaint_only_masked_padding = None
        self.ad_use_separate_width_height = None
        self.ad_inpaint_width = None
        self.ad_inpaint_height = None
        self.ad_use_separate_steps = None
        self.ad_steps = None
        self.ad_use_separate_cfg_scale = False
        self.ad_cfg_scale = None
        self.ad_use_separate_checkpoint = False
        self.ad_checkpoint = None
        self.ad_use_separate_vae = False
        self.ad_vae = None
        self.ad_use_separate_sampler = False
        self.ad_sampler = None
        self.ad_use_separate_noise_multiplier = False
        self.ad_noise_multiplier = None
        self.ad_use_separate_clip_skip = False
        self.ad_clip_skip = None
        self.ad_use_restore_face_after_adetailer = False
        self.ad_controlnet_model = None
        self.ad_controlnet_module = None
        self.ad_controlnet_guidance_start = None
        self.ad_controlnet_guidance_end = None
        self.ad_controlnet_weight = None

    def gr_interactive(value: bool = True):
        return gr.update(interactive=value)

    def on_ad_model_update(self, model: str):
        if "-world" in model:
            return gr.update(
                visible=True,
                placeholder="Comma separated class names to detect, ex: 'person,cat'. default: COCO 80 classes",
            )
        return gr.update(visible=False, placeholder="")

    def on_cn_model_update(self, cn_model_name: str):
        cn_model_name = cn_model_name.replace("inpaint_depth", "depth")

        preprocessors_list = {
            "inpaint": list(global_state.get_filtered_preprocessors("Inpaint")),
            "lineart": list(global_state.get_filtered_preprocessors("Lineart")),
            "openpose": list(global_state.get_filtered_preprocessors("OpenPose")),
            "tile": list(global_state.get_filtered_preprocessors("Tile")),
            "scribble": list(global_state.get_filtered_preprocessors("Scribble")),
            "depth": list(global_state.get_filtered_preprocessors("Depth")),
        }

        for t in preprocessors_list:
            if t in cn_model_name:
                choices = preprocessors_list[t]
                return gr.update(visible=True, choices=choices, value=choices[0])
        return gr.update(visible=False, choices=["None"], value="None")

    def render(self, n: int, is_img2img: bool, webui_info: WebuiInfo) -> None:
        elemid_prefix = "adetailer-"

        with gr.Group():
            with gr.Row():
                with gr.Column(scale=6):
                    self.ad_enabled = gr.Checkbox(
                        value=self.default_unit.ad_enabled,
                        label="Enable ADetailer",
                        elem_id=f"{elemid_prefix}enabled-{n}",
                        elem_classes=["adetailer-unit-enabled"],
                    )

                with gr.Column(scale=6):
                    self.ad_skip_img2img = gr.Checkbox(
                        label="Skip img2img",
                        value=self.default_unit.ad_skip_img2img,
                        visible=is_img2img,
                        elem_id=f"{elemid_prefix}-skip-img2img-{n}"
                    )

        with gr.Group():
            with gr.Row():
                model_choices = (
                    [*webui_info.ad_model_list, "None"]
                    if n == 0
                    else ["None", *webui_info.ad_model_list]
                )

                self.ad_model = gr.Dropdown(
                    label="ADetailer model",
                    choices=model_choices,
                    value=model_choices[0],
                    elem_id=f"{elemid_prefix}model-{n}",
                )

            with gr.Row():
                self.ad_model_classes = gr.Textbox(
                    label="ADetailer model classes",
                    value=self.default_unit.ad_model_classes,
                    visible=False,
                    elem_id=f"{elemid_prefix}model_classes-{n}",
                )

                self.ad_model.change(
                    self.on_ad_model_update,
                    inputs=self.ad_model,
                    outputs=self.ad_model_classes,
                    queue=False,
                )

        gr.HTML("<br>")

        with gr.Group():
            with gr.Row(elem_id=f"{elemid_prefix}prompt-{n}"):
                self.ad_prompt = gr.Textbox(
                    label=f"ADetailer prompt",
                    show_label=False,
                    value=self.default_unit.ad_prompt,
                    lines=3,
                    placeholder=f"ADetailer prompt {(n + 1)}\nIf blank, the main prompt is used.",
                    elem_id=f"{elemid_prefix}prompt-{n}",
                )

            with gr.Row(elem_id=f"{elemid_prefix}negative-prompt-{n}"):
                self.ad_negative_prompt = gr.Textbox(
                    label=f"ADetailer negative prompt",
                    show_label=False,
                    value=self.default_unit.ad_negative_prompt,
                    lines=2,
                    placeholder=f"ADetailer negative prompt {(n + 1)}\nIf blank, the main negative prompt is used.",
                    elem_id=f"{elemid_prefix}negative_prompt-{n}",
                )

        with gr.Group():
            with gr.Accordion(
                "Detection",
                open=False,
                elem_id=f"{elemid_prefix}detection-accordion-{n}",
            ):
                with gr.Row():
                    with gr.Column(variant="compact"):
                        self.ad_detection_confidence_threshold = gr.Slider(
                            label="Detection model confidence threshold",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=self.default_unit.ad_detection_confidence_threshold,
                            visible=True,
                            elem_id=f"{elemid_prefix}detection_confidence_threshold-{n}",
                        )
                        self.ad_mask_k_largest = gr.Slider(
                            label="Mask only the top k largest (0 to disable)",
                            minimum=0,
                            maximum=10,
                            step=1,
                            value=self.default_unit.ad_mask_k_largest,
                            visible=True,
                            elem_id=f"{elemid_prefix}mask_k_largest-{n}",
                        )
                    with gr.Column(variant="compact"):
                        self.ad_mask_min_ratio = gr.Slider(
                            label="Mask min area ratio",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.001,
                            value=self.default_unit.ad_mask_min_ratio,
                            visible=True,
                            elem_id=f"{elemid_prefix}mask_min_ratio-{n}",
                        )
                        self.ad_mask_max_ratio = gr.Slider(
                            label="Mask max area ratio",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.001,
                            value=self.default_unit.ad_mask_max_ratio,
                            visible=True,
                            elem_id=f"{elemid_prefix}mask_max_ratio-{n}",
                        )

            with gr.Accordion(
                "Mask Preprocessing",
                open=False,
                elem_id=f"{elemid_prefix}mask-preprocessing-accordion-{n}",
            ):
                with gr.Group():
                    with gr.Row():
                        with gr.Column(variant="compact"):
                            self.ad_mask_x_offset = gr.Slider(
                                label="Mask x(→) offset",
                                minimum=-200,
                                maximum=200,
                                step=1,
                                value=self.default_unit.ad_mask_x_offset,
                                visible=True,
                                elem_id=f"{elemid_prefix}mask_x_offset-{n}",
                            )
                            self.ad_mask_y_offset = gr.Slider(
                                label="Mask y(↑) offset",
                                minimum=-200,
                                maximum=200,
                                step=1,
                                value=self.default_unit.ad_mask_y_offset,
                                visible=True,
                                elem_id=f"{elemid_prefix}mask_y_offset-{n}",
                            )

                        with gr.Column(variant="compact"):
                            self.ad_mask_erosion_dilation = gr.Slider(
                                label="Mask erosion (-) / dilation (+)",
                                minimum=-128,
                                maximum=128,
                                step=4,
                                value=self.default_unit.ad_mask_erosion_dilation,
                                visible=True,
                                elem_id=f"{elemid_prefix}mask_erosion_dilation-{n}",
                            )

                    with gr.Row():
                        self.ad_mask_merge_mode = gr.Radio(
                            label="Mask merge mode",
                            choices=["None", "Merge", "Merge and Invert"],
                            value=self.default_unit.ad_mask_merge_mode,
                            elem_id=f"{elemid_prefix}mask_merge_mode-{n}",
                        )

            with gr.Accordion(
                "Inpainting",
                open=False,
                elem_id=f"{elemid_prefix}inpainting-accordion-{n}",
            ):
                with gr.Group():
                    with gr.Row():
                        self.ad_inpaint_mask_blur = gr.Slider(
                            label="Inpaint mask blur",
                            minimum=0,
                            maximum=64,
                            step=1,
                            value=self.default_unit.ad_inpaint_mask_blur,
                            visible=True,
                            elem_id=f"{elemid_prefix}inpaint_mask_blur-{n}",
                        )

                        self.ad_inpaint_mask_denoising = gr.Slider(
                            label="Inpaint denoising strength",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=self.default_unit.ad_inpaint_mask_denoising,
                            visible=True,
                            elem_id=f"{elemid_prefix}inpaint_mask_denoising-{n}",
                        )

                    with gr.Row():
                        with gr.Column(variant="compact"):
                            self.ad_inpaint_only_masked = gr.Checkbox(
                                label="Inpaint only masked",
                                value=self.default_unit.ad_inpaint_only_masked,
                                visible=True,
                                elem_id=f"{elemid_prefix}inpaint_only_masked-{n}",
                            )
                            self.ad_inpaint_only_masked_padding = gr.Slider(
                                label="Inpaint only masked padding, pixels",
                                minimum=0,
                                maximum=256,
                                step=4,
                                value=self.default_unit.ad_inpaint_only_masked_padding,
                                visible=True,
                                elem_id=f"{elemid_prefix}inpaint_only_masked_padding-{n}",
                            )
                            self.ad_inpaint_only_masked.change(
                                self.gr_interactive,
                                inputs=self.ad_inpaint_only_masked,
                                outputs=self.ad_inpaint_only_masked_padding,
                                queue=False,
                            )

                        with gr.Column(variant="compact"):
                            self.ad_use_separate_width_height = gr.Checkbox(
                                label="Use separate width/height",
                                value=self.default_unit.ad_use_separate_width_height,
                                visible=True,
                                elem_id=f"{elemid_prefix}use_separate_width_height-{n}",
                            )

                            self.ad_inpaint_width = gr.Slider(
                                label="inpaint width",
                                minimum=64,
                                maximum=2048,
                                step=4,
                                value=self.default_unit.ad_inpaint_width,
                                visible=True,
                                elem_id=f"{elemid_prefix}inpaint_width-{n}",
                            )

                            self.ad_inpaint_height = gr.Slider(
                                label="inpaint height",
                                minimum=64,
                                maximum=2048,
                                step=4,
                                value=self.default_unit.ad_inpaint_height,
                                visible=True,
                                elem_id=f"{elemid_prefix}inpaint_height-{n}",
                            )

                            self.ad_use_separate_width_height.change(
                                lambda value: (self.gr_interactive(value)),
                                inputs=self.ad_use_separate_width_height,
                                outputs=[self.ad_inpaint_width, self.ad_inpaint_height],
                                queue=False,
                            )

                    with gr.Row():
                        with gr.Column(variant="compact"):
                            self.ad_use_separate_steps = gr.Checkbox(
                                label="Use separate steps",
                                value=self.default_unit.ad_use_separate_steps,
                                visible=True,
                                elem_id=f"{elemid_prefix}use_separate_steps-{n}",
                            )

                            self.ad_steps = gr.Slider(
                                label="ADetailer steps",
                                minimum=1,
                                maximum=150,
                                step=1,
                                value=self.default_unit.ad_steps,
                                visible=True,
                                elem_id=f"{elemid_prefix}steps-{n}",
                            )

                            self.ad_use_separate_steps.change(
                                self.gr_interactive,
                                inputs=self.ad_use_separate_steps,
                                outputs=self.ad_steps,
                                queue=False,
                            )

                        with gr.Column(variant="compact"):
                            self.ad_use_separate_cfg_scale = gr.Checkbox(
                                label="Use separate CFG scale",
                                value=self.default_unit.ad_use_separate_cfg_scale,
                                visible=True,
                                elem_id=f"{elemid_prefix}use_separate_cfg_scale-{n}",
                            )

                            self.ad_cfg_scale = gr.Slider(
                                label="ADetailer CFG scale",
                                minimum=0.0,
                                maximum=30.0,
                                step=0.5,
                                value=self.default_unit.ad_cfg_scale,
                                visible=True,
                                elem_id=f"{elemid_prefix}cfg_scale-{n}",
                            )

                            self.ad_use_separate_cfg_scale.change(
                                self.gr_interactive,
                                inputs=self.ad_use_separate_cfg_scale,
                                outputs=self.ad_cfg_scale,
                                queue=False,
                            )

                    with gr.Row():
                        with gr.Column(variant="compact"):
                            self.ad_use_separate_checkpoint = gr.Checkbox(
                                label="Use separate checkpoint",
                                value=self.default_unit.ad_use_separate_checkpoint,
                                visible=True,
                                elem_id=f"{elemid_prefix}use_separate_checkpoint-{n}",
                            )

                            ckpts = ["Use same checkpoint", *webui_info.checkpoints_list]

                            self.ad_checkpoint = gr.Dropdown(
                                label="ADetailer checkpoint",
                                choices=ckpts,
                                value=ckpts[0],
                                visible=True,
                                elem_id=f"{elemid_prefix}checkpoint-{n}",
                            )

                        with gr.Column(variant="compact"):
                            self.ad_use_separate_vae = gr.Checkbox(
                                label="Use separate VAE",
                                value=self.default_unit.ad_use_separate_vae,
                                visible=True,
                                elem_id=f"{elemid_prefix}use_separate_vae-{n}",
                            )

                            vaes = ["Use same VAE", *webui_info.vae_list]

                            self.ad_vae = gr.Dropdown(
                                label="ADetailer VAE",
                                choices=vaes,
                                value=vaes[0],
                                visible=True,
                                elem_id=f"{elemid_prefix}vae-{n}",
                            )

                    with gr.Row(), gr.Column(variant="compact"):
                        self.ad_use_separate_sampler = gr.Checkbox(
                            label="Use separate sampler",
                            value=self.default_unit.ad_use_separate_sampler,
                            visible=True,
                            elem_id=f"{elemid_prefix}use_separate_sampler-{n}",
                        )

                        self.ad_sampler = gr.Dropdown(
                            label="ADetailer sampler",
                            choices=webui_info.sampler_names,
                            value=webui_info.sampler_names[0],
                            visible=True,
                            elem_id=f"{elemid_prefix}sampler-{n}",
                        )

                        self.ad_use_separate_sampler.change(
                            self.gr_interactive,
                            inputs=self.ad_use_separate_sampler,
                            outputs=self.ad_sampler,
                            queue=False,
                        )

                    with gr.Row():
                        with gr.Column(variant="compact"):
                            self.ad_use_separate_noise_multiplier = gr.Checkbox(
                                label="Use separate noise multiplier",
                                value=self.default_unit.ad_use_separate_noise_multiplier,
                                visible=True,
                                elem_id=f"{elemid_prefix}use_separate_noise_multiplier-{n}",
                            )

                            self.ad_noise_multiplier = gr.Slider(
                                label="Noise multiplier for img2img",
                                minimum=0.5,
                                maximum=1.5,
                                step=0.01,
                                value=self.default_unit.ad_noise_multiplier,
                                visible=True,
                                elem_id=f"{elemid_prefix}noise_multiplier-{n}",
                            )

                            self.ad_use_separate_noise_multiplier.change(
                                self.gr_interactive,
                                inputs=self.ad_use_separate_noise_multiplier,
                                outputs=self.ad_noise_multiplier,
                                queue=False,
                            )

                        with gr.Column(variant="compact"):
                            self.ad_use_separate_clip_skip = gr.Checkbox(
                                label="Use separate CLIP skip",
                                value=self.default_unit.ad_use_separate_clip_skip,
                                visible=True,
                                elem_id=f"{elemid_prefix}use_separate_clip_skip-{n}",
                            )

                            self.ad_clip_skip = gr.Slider(
                                label="ADetailer CLIP skip",
                                minimum=1,
                                maximum=12,
                                step=1,
                                value=self.default_unit.ad_clip_skip,
                                visible=True,
                                elem_id=f"{elemid_prefix}clip_skip-{n}",
                            )

                            self.ad_use_separate_clip_skip.change(
                                self.gr_interactive,
                                inputs=self.ad_use_separate_clip_skip,
                                outputs=self.ad_clip_skip,
                                queue=False,
                            )

                    with gr.Row(), gr.Column(variant="compact"):
                        self.ad_use_restore_face_after_adetailer = gr.Checkbox(
                            label="Restore faces after ADetailer",
                            value=self.default_unit.ad_use_restore_face_after_adetailer,
                            elem_id=f"{elemid_prefix}use_restore_face_after_adetailer-{n}",
                        )

            with gr.Group():
                with gr.Row(variant="panel"):
                    with gr.Column(variant="compact"):
                        self.ad_controlnet_model = gr.Dropdown(
                            label="ControlNet model",
                            choices=["None", *webui_info.controlnet_model_list],
                            value=self.default_unit.ad_controlnet_model,
                            visible=True,
                            type="value",
                            elem_id=f"{elemid_prefix}controlnet_model-{n}",
                        )

                        self.ad_controlnet_module = gr.Dropdown(
                            label="ControlNet module",
                            choices=["None"],
                            value=self.default_unit.ad_controlnet_module,
                            visible=False,
                            type="value",
                            elem_id=f"{elemid_prefix}controlnet_module-{n}",
                        )

                        self.ad_controlnet_weight = gr.Slider(
                            label="ControlNet weight",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=self.default_unit.ad_controlnet_weight,
                            visible=True,
                            elem_id=f"{elemid_prefix}controlnet_weight-{n}",
                        )

                        self.ad_controlnet_model.change(
                            self.on_cn_model_update,
                            inputs=self.ad_controlnet_model,
                            outputs=self.ad_controlnet_module,
                            queue=False,
                        )

                    with gr.Column(variant="compact"):
                        self.ad_controlnet_guidance_start = gr.Slider(
                            label="ControlNet guidance start",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=self.default_unit.ad_controlnet_guidance_start,
                            visible=True,
                            elem_id=f"{elemid_prefix}controlnet_guidance_start-{n}",
                        )

                        self.ad_controlnet_guidance_end = gr.Slider(
                            label="ControlNet guidance end",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=self.default_unit.ad_controlnet_guidance_end,
                            visible=True,
                            elem_id=f"{elemid_prefix}controlnet_guidance_end-{n}",
                        )

        unit_args = (
            self.ad_enabled,
            self.ad_skip_img2img,
            self.ad_model,
            self.ad_model_classes,
            self.ad_prompt,
            self.ad_negative_prompt,
            self.ad_detection_confidence_threshold,
            self.ad_mask_min_ratio,
            self.ad_mask_max_ratio,
            self.ad_mask_k_largest,
            self.ad_mask_x_offset,
            self.ad_mask_y_offset,
            self.ad_mask_erosion_dilation,
            self.ad_mask_merge_mode,
            self.ad_inpaint_mask_blur,
            self.ad_inpaint_mask_denoising,
            self.ad_inpaint_only_masked,
            self.ad_inpaint_only_masked_padding,
            self.ad_use_separate_width_height,
            self.ad_inpaint_width,
            self.ad_inpaint_height,
            self.ad_use_separate_steps,
            self.ad_steps,
            self.ad_use_separate_cfg_scale,
            self.ad_cfg_scale,
            self.ad_use_separate_checkpoint,
            self.ad_checkpoint,
            self.ad_use_separate_vae,
            self.ad_vae,
            self.ad_use_separate_sampler,
            self.ad_sampler,
            self.ad_use_separate_noise_multiplier,
            self.ad_noise_multiplier,
            self.ad_use_separate_clip_skip,
            self.ad_clip_skip,
            self.ad_use_restore_face_after_adetailer,
            self.ad_controlnet_model,
            self.ad_controlnet_module,
            self.ad_controlnet_guidance_start,
            self.ad_controlnet_guidance_end,
            self.ad_controlnet_weight,
        )

        unit = gr.State(self.default_unit)
        (
            ADetailerUiGroup.img2img_submit_button
            if self.is_img2img
            else ADetailerUiGroup.txt2img_submit_button
        ).click(
            fn=ADetailerUnit,
            inputs=list(unit_args),
            outputs=unit,
            queue=False,
        )

        return unit


def on_after_component(component, **_kwargs):
    elem_id = getattr(component, "elem_id", None)

    if elem_id == "txt2img_generate":
        ADetailerUiGroup.txt2img_submit_button = component
        return

    if elem_id == "img2img_generate":
        ADetailerUiGroup.img2img_submit_button = component
        return

# XYZ Plot

def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_ad_xyz"):
        p._ad_xyz = {}
    p._ad_xyz[field] = x

def search_and_replace_prompt(p, x: Any, xs: Any, replace_in_main_prompt: bool):
    if replace_in_main_prompt:
        p.prompt = p.prompt.replace(xs[0], x)
        p.negative_prompt = p.negative_prompt.replace(xs[0], x)

    if not hasattr(p, "_ad_xyz_prompt_sr"):
        p._ad_xyz_prompt_sr = []
    p._ad_xyz_prompt_sr.append(PromptSR(s=xs[0], r=x))

def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    model_list = ["None", *get_model_mapping().keys()]
    samplers = [sampler.name for sampler in all_samplers]

    cn_models, _ = get_controlnet_models()

    axis = [
        xyz_grid.AxisOption(
            "[ADetailer] Unit 1 model",
            str,
            partial(set_value, field="ad_model"),
            choices=lambda: model_list,
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Unit 1 prompt",
            str,
            partial(set_value, field="ad_prompt"),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Unit 1 negative prompt",
            str,
            partial(set_value, field="ad_negative_prompt"),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Unit 1 Prompt S/R",
            str,
            partial(search_and_replace_prompt, replace_in_main_prompt=False),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Unit 1 and Main Prompt, Prompt S/R",
            str,
            partial(search_and_replace_prompt, replace_in_main_prompt=True),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Unit 1 Mask detection confidence threshold",
            int,
            partial(set_value, field="ad_detection_confidence_threshold"),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Unit 1 Inpaint denoising strength",
            float,
            partial(set_value, field="ad_denoising_strength"),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Unit 1 Inpaint only masked",
            str,
            partial(set_value, field="ad_inpaint_only_masked"),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Unit 1 Inpaint only masked padding",
            int,
            partial(set_value, field="ad_inpaint_only_masked_padding"),
        ),
        xyz_grid.AxisOption(
            "[ADetailer] Unit 1 sampler",
            str,
            partial(set_value, field="ad_sampler"),
            choices=lambda: samplers,
        ),
        # xyz_grid.AxisOption(
        #     "[ADetailer] Unit 1 ControlNet model",
        #     str,
        #     partial(set_value, field="ad_controlnet_model"),
        #     choices=lambda: ["None", "Passthrough", cn_models],
        # ),
    ]

    if not any(x.label.startswith("[ADetailer]") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)

def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception as e:
        logger.error(f"xyz_grid error:\n{e}")

def add_api_endpoints(_: gr.Blocks, app: FastAPI):
    @app.get("/adetailer/v1/version")
    async def version():
        return {"version": __version__}

    @app.get("/adetailer/v1/schema")
    async def schema():
        return ADetailerUnitSchema.schema()

    @app.get("/adetailer/v1/ad_model")
    async def ad_model():
        return {"ad_model": list(get_model_mapping().keys())}
