import os
import platform
from typing import Dict, Optional, Tuple, List, Union, Any, NamedTuple

import copy
import cv2
import torch

import re
import gradio as gr
from pathlib import Path
from PIL import Image

import numpy as np

from torchvision.transforms.functional import to_pil_image

from contextlib import contextmanager, suppress

from modules import devices, images, safe, shared, paths, scripts, sd_models, shared_items, sd_schedulers
from modules.shared import cmd_opts, opts, state
from modules.sd_samplers import all_samplers

from modules.scripts import PostprocessImageArgs
from modules.processing import (
    Processed,
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
    create_infotext,
    process_images,
)

from modules.devices import NansException

from lib_adetailer.common import PredictOutput
from lib_adetailer.logger import logger_adetailer as logger
from lib_adetailer.args import ADetailerUnit, WebuiInfo, PromptSR, SCRIPT_DEFAULT, BBOX_SORTBY

from lib_adetailer.mask import (
    filter_by_ratio,
    filter_k_largest,
    mask_preprocess,
    sort_bboxes,
)

from lib_controlnet import external_code, global_state
from lib_controlnet.external_code import ControlNetUnit

from lib_adetailer import (
    AFTER_DETAILER,
    __version__,
    get_models,
    mediapipe_predict,
    ultralytics_predict,
)

@contextmanager
def change_torch_load():
    orig = torch.load
    try:
        torch.load = safe.unsafe_torch_load
        yield
    finally:
        torch.load = orig

@contextmanager
def pause_total_tqdm():
    orig = opts.data.get("multiple_tqdm", True)
    try:
        opts.data["multiple_tqdm"] = False
        yield
    finally:
        opts.data["multiple_tqdm"] = orig

@contextmanager
def preseve_prompts(p):
    all_pt = copy.copy(p.all_prompts)
    all_ng = copy.copy(p.all_negative_prompts)
    try:
        yield
    finally:
        p.all_prompts = all_pt
        p.all_negative_prompts = all_ng

@torch.no_grad()
def afterdetailer_process_image(n: int, unit: ADetailerUnit, p, pp, *args):
    if state.interrupted or state.skipped:
        return False

    kwargs = {}
    if unit.ad_model.lower().startswith("mediapipe"):
        predictor = mediapipe_predict
        ad_model = unit.ad_model
    else:
        model_mapping = get_model_mapping()
        predictor = ultralytics_predict
        ad_model = model_mapping[unit.ad_model]
        kwargs["device"] = get_ultralytics_device()
        kwargs["classes"] = unit.ad_model_classes

    detection_confidence_threshold = getattr(unit, "ad_detection_confidence_threshold")
    if detection_confidence_threshold is None:
        detection_confidence_threshold = unit.ad_detection_confidence_threshold

    with change_torch_load():
        pred = predictor(ad_model, pp.image, detection_confidence_threshold, **kwargs)

    masks = pred_preprocessing(pred, unit)
    state.assign_current_image(pred.preview)

    if not masks:
        logger.warn(f"No masks found for {unit.ad_model!r}")
        return False

    save_image(
        p,
        pred.preview,
        condition="ad_save_previews",
        suffix=f"-ad-preview-{n}",
    )

    faces = len(masks)
    processed = None
    state.job_count += faces

    logger.info(f"{faces} masks found for {unit.ad_model!r}")

    noise_multiplier = get_initial_noise_multiplier(p, unit)
    seed, subseed = get_seed(p)
    width, height = get_width_height(p, unit)
    sampler_name = get_sampler(p, unit)
    scheduler_name = get_scheduler(p, unit)
    steps = get_steps(p, unit)
    cfg_scale = get_cfg_scale(p, unit)
    override_settings = get_override_settings(p, unit)

    denoising_strength = get_ad_xyz(p, 'ad_inpaint_mask_denoising')
    if denoising_strength is None:
        denoising_strength = unit.ad_inpaint_mask_denoising

    inpaint_only_masked = get_ad_xyz(p, 'ad_inpaint_only_masked')
    if inpaint_only_masked is None:
        inpaint_only_masked = unit.ad_inpaint_only_masked

    inpaint_only_masked_padding = get_ad_xyz(p, 'ad_inpaint_only_masked_padding')
    if inpaint_only_masked_padding is None:
        inpaint_only_masked_padding = unit.ad_inpaint_only_masked_padding

    # Remove Hires prompt and Hires negative prompt params for i2i
    if 'Hires prompt' in p.extra_generation_params:
        del p.extra_generation_params['Hires prompt']
    if 'Hires negative prompt' in p.extra_generation_params:
        del p.extra_generation_params['Hires negative prompt']

    i2i = StableDiffusionProcessingImg2Img()

    i2i.cached_c = [None, None, None]
    i2i.cached_uc = [None, None, None]
    i2i.scripts, i2i.script_args = script_filter(p, unit)

    # Override parameters AGAIN.
    i2i.init_images = [pp.image]
    i2i.resize_mode = 0
    i2i.denoising_strength = denoising_strength
    i2i.mask = None
    i2i.mask_blur = unit.ad_inpaint_mask_blur
    i2i.inpainting_fill = 1
    i2i.inpaint_full_res = inpaint_only_masked
    i2i.inpaint_full_res_padding = inpaint_only_masked_padding
    i2i.inpainting_mask_invert = 0
    i2i.sd_model = p.sd_model
    i2i.outpath_samples = p.outpath_samples
    i2i.outpath_grids = p.outpath_grids
    i2i.prompt = ""
    i2i.negative_prompt = ""
    i2i.styles = p.styles
    i2i.seed = seed
    i2i.subseed = subseed
    i2i.subseed_strength = p.subseed_strength
    i2i.seed_resize_from_h = p.seed_resize_from_h
    i2i.seed_resize_from_w = p.seed_resize_from_w
    i2i.sampler_name = sampler_name
    i2i.scheduler = scheduler_name
    i2i.batch_size = 1
    i2i.n_iter = 1
    i2i.steps = steps
    i2i.cfg_scale = cfg_scale
    i2i.width = width
    i2i.height = height
    i2i.restore_faces = unit.ad_use_restore_face_after_adetailer
    i2i.tiling = p.tiling
    i2i.extra_generation_params = p.extra_generation_params
    i2i.do_not_save_samples = True
    i2i.do_not_save_grid = True
    i2i.override_settings = override_settings

    # TODO: Clean up this, need more test.
    i2i.scripts.alwayson_scripts = []
    i2i.script_args_value = []

    # TODO: Need more test.
    # script_args_value = []
    # if type(i2i.script_args_value) is tuple:
    #     script_args_value.append(i2i.script_args_value)
    # else:
    #     script_args_value = i2i.script_args_value
    #
    # i2i.script_args_value = script_args_value
    # del script_args_value

    ad_prompts, ad_negatives = get_prompt(p, unit)

    i2i.prompts = ad_prompts
    i2i.negative_prompts = ad_negatives
    i2i.setup_conds()

    if unit.ad_controlnet_model not in ["None", "Passthrough"]:
        image = np.asarray(pp.image)
        mask = np.full_like(image, fill_value=255)
        cnet_image = {"image": image, "mask": mask}

        pres = external_code.pixel_perfect_resolution(
            image,
            target_H=height,
            target_W=width,
            resize_mode=external_code.resize_mode_from_value(i2i.resize_mode),
        )

        cn = ControlNetUnit()
        cn.enabled = True
        cn.image = cnet_image
        cn.model = unit.ad_controlnet_model
        cn.module = unit.ad_controlnet_module
        cn.weight = unit.ad_controlnet_weight
        cn.guidance_start = unit.ad_controlnet_guidance_start
        cn.guidance_end = unit.ad_controlnet_guidance_end
        cn.processor_res = pres

        add_forge_script_to_adetailer_run(
            i2i,
            "ControlNet",
            [cn],
        )

    elif unit.ad_controlnet_model == "None":
        i2i.control_net_enabled = False

    p2 = copy.copy(i2i)

    for j in range(faces):
        p2.image_mask = masks[j]
        p2.init_images[0] = ensure_rgb_image(p2.init_images[0])
        i2i_prompts_replace(p2, ad_prompts, ad_negatives, j)

        if re.match(r"^\s*\[SKIP\]\s*$", p2.prompt):
            continue

        p2.seed = get_each_tap_seed(seed, j)
        p2.subseed = get_each_tap_seed(subseed, j)
        p2.prompts = [p2.prompt]
        p2.negative_prompts = [p2.negative_prompt]
        p2.setup_conds()

        try:
            processed = process_images(p2)
        except NansException as e:
            logger.error(f"'NansException' occurred with {n} unit.\n{e}")

            continue
        finally:
            p2.close()

        compare_prompt(p2, processed, n=n)

        p2 = copy.copy(i2i)
        p2.init_images = [processed.images[0]]

    if processed is not None:
        pp.image = processed.images[0]
        state.assign_current_image(pp.image)

        del(p2)
        devices.torch_gc()

        return True

    return False

def pred_preprocessing(pred: PredictOutput, unit: ADetailerUnit):
    pred = filter_by_ratio(
        pred, low=unit.ad_mask_min_ratio, high=unit.ad_mask_max_ratio
    )
    pred = filter_k_largest(pred, k=unit.ad_mask_k_largest)
    sortby = opts.data.get("ad_bbox_sortby", BBOX_SORTBY[0])
    sortby_idx = BBOX_SORTBY.index(sortby)
    pred = sort_bboxes(pred, sortby_idx)

    return mask_preprocess(
        pred.masks,
        kernel=unit.ad_mask_erosion_dilation,
        x_offset=unit.ad_mask_x_offset,
        y_offset=unit.ad_mask_y_offset,
        merge_invert=unit.ad_mask_merge_mode,
    )

def script_filter(p, unit: ADetailerUnit):
    script_runner = copy.copy(p.scripts)
    script_args = script_args_copy(p.script_args)

    return script_runner, script_args

    # TODO: Need more test.
    # ad_only_seleted_scripts = opts.data.get("ad_only_seleted_scripts", True)
    # if not ad_only_seleted_scripts:
    #     return script_runner, script_args

    # ad_script_names = opts.data.get("ad_script_names", SCRIPT_DEFAULT)
    # script_names_set = {
    #     name
    #     for script_name in ad_script_names.split(",")
    #     for name in (script_name, script_name.strip())
    # }

    # filtered_alwayson = []
    # for script_object in script_runner.alwayson_scripts:
    #     filepath = script_object.filename
    #     filename = Path(filepath).stem
    #     if filename in script_names_set:
    #         filtered_alwayson.append(script_object)

    # script_runner.alwayson_scripts = filtered_alwayson
    # return script_runner, script_args

def save_image(p, image, *, condition: str, suffix: str) -> None:
    i = get_i(p)

    if p.all_prompts:
        i %= len(p.all_prompts)
        save_prompt = p.all_prompts[i]
    else:
        save_prompt = p.prompt

    if opts.data.get(condition, False):
        images.save_image(
            image=image,
            path=p.outpath_samples,
            basename="",
            seed=p.seed,
            prompt=save_prompt,
            extension=opts.samples_format,
            p=p,
            suffix=suffix,
        )

def _get_prompt(
    ad_prompt: str,
    all_prompts: list[str],
    i: int,
    default: str,
    replacements: list[PromptSR],
) -> list[str]:
    prompts = re.split(r"\s*\[SEP\]\s*", ad_prompt)
    blank_replacement = prompt_blank_replacement(all_prompts, i, default)

    for n in range(len(prompts)):
        if not prompts[n]:
            prompts[n] = blank_replacement
        elif "[PROMPT]" in prompts[n]:
            prompts[n] = prompts[n].replace("[PROMPT]", f" {blank_replacement} ")

        for pair in replacements:
            prompts[n] = prompts[n].replace(pair.s, pair.r)

    return prompts

def get_prompt(p, unit: ADetailerUnit) -> tuple[list[str], list[str]]:
    i = get_i(p)
    prompt_sr = p._ad_xyz_prompt_sr if hasattr(p, "_ad_xyz_prompt_sr") else []

    ad_prompt = get_ad_xyz(p, "ad_prompt")
    if ad_prompt is None:
        ad_prompt = unit.ad_prompt

    ad_negative_prompt = get_ad_xyz(p, "ad_negative_prompt")
    if ad_negative_prompt is None:
        ad_negative_prompt = unit.ad_negative_prompt

    prompt = _get_prompt(ad_prompt, p.all_prompts, i, p.prompt, prompt_sr)

    negative_prompt = _get_prompt(
        ad_negative_prompt,
        p.all_negative_prompts,
        i,
        p.negative_prompt,
        prompt_sr,
    )

    return prompt, negative_prompt

def get_seed(p) -> tuple[int, int]:
    i = get_i(p)

    if not p.all_seeds:
        seed = p.seed
    elif i < len(p.all_seeds):
        seed = p.all_seeds[i]
    else:
        j = i % len(p.all_seeds)
        seed = p.all_seeds[j]

    if not p.all_subseeds:
        subseed = p.subseed
    elif i < len(p.all_subseeds):
        subseed = p.all_subseeds[i]
    else:
        j = i % len(p.all_subseeds)
        subseed = p.all_subseeds[j]

    return seed, subseed

def add_forge_script_to_adetailer_run(
    p: StableDiffusionProcessing,
    script_title: str,
    script_args: list
):
    script = next((s for s in p.scripts.scripts if s.title() == script_title), None)
    if not script:
        raise Exception(f"Script not found: {script_title!r}")

    script = copy.copy(script)
    script.args_from = len(p.script_args_value)
    script.args_to = len(p.script_args_value) + len(script_args)
    p.scripts.alwayson_scripts.append(script)
    p.script_args_value.extend(script_args)

def write_params_txt(content: str) -> None:
    params_txt = Path(paths.data_path, "params.txt")
    with suppress(Exception):
        params_txt.write_text(content, encoding="utf-8")

def prompt_blank_replacement(
    all_prompts: list[str], i: int, default: str
) -> str:
    if not all_prompts:
        return default
    if i < len(all_prompts):
        return all_prompts[i]
    j = i % len(all_prompts)
    return all_prompts[j]

def get_i(p) -> int:
    return p.iteration * p.batch_size + p.batch_index

def get_width_height(p, unit: ADetailerUnit) -> tuple[int, int]:
    if unit.ad_use_separate_width_height:
        width = unit.ad_inpaint_width
        height = unit.ad_inpaint_height
    else:
        width = p.width
        height = p.height

    return width, height

def get_steps(p, unit: ADetailerUnit) -> int:
    if unit.ad_use_separate_steps:
        return unit.ad_steps
    return p.steps

def get_cfg_scale(p, unit: ADetailerUnit) -> float:
    return unit.ad_cfg_scale if unit.ad_use_separate_cfg_scale else p.cfg_scale

def get_sampler(p, unit: ADetailerUnit) -> str:
    if hasattr(p, "_ad_xyz") and "ad_sampler" in p._ad_xyz.keys():
        return p._ad_xyz.get('ad_sampler')
    if unit.ad_use_separate_sampler:
        return unit.ad_sampler
    return p.sampler_name

def get_scheduler(p, unit: ADetailerUnit) -> str:
    if hasattr(p, "_ad_xyz") and "ad_scheduler" in p._ad_xyz.keys():
        return p._ad_xyz.get('ad_scheduler')
    if unit.ad_use_separate_scheduler:
        return unit.ad_scheduler
    return p.scheduler

def get_ad_xyz(p, key: str):
    if hasattr(p, "_ad_xyz") and key in p._ad_xyz.keys():
        return p._ad_xyz.get(key)
    return None

def get_override_settings(p, unit: ADetailerUnit) -> dict[str, Any]:
    d = {}

    if unit.ad_use_separate_clip_skip:
        d["CLIP_stop_at_last_layers"] = unit.ad_clip_skip

    if (
        unit.ad_use_separate_checkpoint
        and unit.ad_checkpoint
        and unit.ad_checkpoint not in ("None", "Use same checkpoint")
    ):
        d["sd_model_checkpoint"] = unit.ad_checkpoint

    if (
        unit.ad_use_separate_vae
        and unit.ad_vae
        and unit.ad_vae not in ("None", "Use same VAE")
    ):
        d["sd_vae"] = unit.ad_vae
    return d

def get_initial_noise_multiplier(p, unit: ADetailerUnit) -> float | None:
    return unit.ad_noise_multiplier if unit.ad_use_separate_noise_multiplier else None

def get_model_mapping():
    no_huggingface = getattr(cmd_opts, "ad_no_huggingface", False)
    adetailer_dir = os.path.join(paths.models_path, "adetailer")
    extra_models_dir = opts.data.get("ad_extra_models_dir", "")

    if (
        not os.path.exists(adetailer_dir)
        and os.path.exists(paths.models_path)
        and os.access(paths.models_path, os.W_OK)
    ):
        os.makedirs(adetailer_dir, mode=0o755)

    return get_models(
        adetailer_dir,
        extra_dir=extra_models_dir,
        huggingface=not no_huggingface
    )

def get_enabled_units(units):
    units = [
        ADetailerUnit.from_dict(unit) if isinstance(unit, dict) else unit
        for unit in units
    ]
    assert all(isinstance(unit, ADetailerUnit) for unit in units)
    enabled_units = [x for x in units if x.ad_enabled]

    return enabled_units

def script_args_copy(script_args):
    type_: type[list] | type[tuple] = type(script_args)
    result = []
    for arg in script_args:
        try:
            a = copy.copy(arg)
        except TypeError:
            a = arg
        result.append(a)
    return type_(result)

def compare_prompt(p, processed, n: int = 0):
    if p.prompt != processed.all_prompts[0]:
        logger.info(f"Applied {(n + 1)} prompt: {processed.all_prompts[0]!r}")

    if p.negative_prompt != processed.all_negative_prompts[0]:
        logger.info(f"Applied {(n + 1)} negative_prompt: {processed.all_negative_prompts[0]!r}")

def get_each_tap_seed(seed: int, i: int):
    use_same_seed = opts.data.get("ad_same_seed_for_each_tap", False)
    return seed if use_same_seed else seed + i

def ensure_rgb_image(image: Any):
    if not isinstance(image, Image.Image):
        image = to_pil_image(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def i2i_prompts_replace(
    i2i, prompts: list[str], negative_prompts: list[str], j: int
) -> None:
    i1 = min(j, len(prompts) - 1)
    i2 = min(j, len(negative_prompts) - 1)
    prompt = prompts[i1]
    negative_prompt = negative_prompts[i2]
    i2i.prompt = prompt
    i2i.negative_prompt = negative_prompt

def get_ultralytics_device() -> str:
    if "adetailer" in cmd_opts.use_cpu:
        return "cpu"

    if platform.system() == "Darwin":
        return ""

    vram_args = ["lowvram", "medvram", "medvram_sdxl"]
    if any(getattr(cmd_opts, vram, False) for vram in vram_args):
        return "cpu"

    return ""

def get_controlnet_models():
    cn_model_module = {
        "inpaint": "inpaint_global_harmonious",
        "scribble": "t2ia_sketch_pidi",
        "lineart": "lineart_coarse",
        "openpose": "openpose_full",
        "tile": "tile_resample",
        "depth": "depth_midas",
    }

    cn_model_regex = re.compile("|".join(cn_model_module.keys()))
    models = global_state.get_all_controlnet_names()
    controlnet_model_list = [m for m in models if cn_model_regex.search(m)]

    preprocessors_list = {
        "inpaint": list(global_state.get_filtered_preprocessors("Inpaint")),
        "lineart": list(global_state.get_filtered_preprocessors("Lineart")),
        "openpose": list(global_state.get_filtered_preprocessors("OpenPose")),
        "tile": list(global_state.get_filtered_preprocessors("Tile")),
        "scribble": list(global_state.get_filtered_preprocessors("Scribble")),
        "depth": list(global_state.get_filtered_preprocessors("Depth")),
    }

    return controlnet_model_list, preprocessors_list

def get_webui_info():
    model_mapping = get_model_mapping()

    ad_model_list = list(model_mapping.keys())
    sampler_names = [sampler.name for sampler in all_samplers]
    scheduler_names = [scheduler.name for scheduler in sd_schedulers.schedulers]

    controlnet_model_list, preprocessors_list = get_controlnet_models()

    try:
        checkpoint_list = sd_models.checkpoint_tiles(use_shorts=True)
    except TypeError:
        checkpoint_list = sd_models.checkpoint_tiles()

    vae_list = shared_items.sd_vae_items()

    webui_info = WebuiInfo()
    webui_info.ad_model_list = ad_model_list
    webui_info.sampler_names = sampler_names
    webui_info.scheduler_names = scheduler_names
    webui_info.t2i_button = None
    webui_info.i2i_button = None
    webui_info.checkpoints_list = checkpoint_list
    webui_info.vae_list = vae_list
    webui_info.controlnet_model_list = controlnet_model_list
    webui_info.preprocessors_list = preprocessors_list

    return webui_info