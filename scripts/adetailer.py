
import os

import copy
import torch

import re
import gradio as gr

from typing import Tuple, List
from pathlib import Path

from modules import script_callbacks, safe, paths, scripts, sd_models, shared_items
from modules.shared import cmd_opts, opts
from modules.scripts import PostprocessImageArgs
from modules.processing import StableDiffusionProcessing

from lib_adetailer.settings import on_ui_settings
from lib_adetailer.infotext import Infotext
from lib_adetailer.ui import ADetailerUiGroup, on_after_component, on_before_ui, add_api_endpoints

from lib_adetailer.logger import logger_adetailer as logger
from lib_adetailer.args import ADetailerUnit

from lib_controlnet import global_state

from lib_adetailer.mask import (
    filter_by_ratio,
    filter_k_largest,
    mask_preprocess,
    sort_bboxes,
)

from lib_adetailer import (
    AFTER_DETAILER,
    __version__,
)

from lib_adetailer.process import (
    afterdetailer_process_image,
    ensure_rgb_image,
    save_image,
    pause_total_tqdm,
    get_webui_info,
    get_enabled_units,
    write_params_txt,
)

class ADetailerScript(scripts.Script):

    def __init__(self):
        self.infotext_fields: List[Tuple[gr.components.IOComponent, str]] = []
        self.paste_field_names: List[str] = []

    def title(self):
        return AFTER_DETAILER

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        num_models = opts.data.get("ad_max_models", 2)

        infotext = Infotext()
        webui_info = get_webui_info()

        default_unit = ADetailerUnit(enabled=False, model="None")
        ui_groups = []
        adetailers = []

        with gr.Group(elem_id="adetailer-group"):
            with gr.Accordion(AFTER_DETAILER, open=False):
                with gr.Group(), gr.Tabs():
                    for n in range(num_models):
                        with gr.Tab(f"Unit {(n + 1)}"):
                            group = ADetailerUiGroup(is_img2img, default_unit)
                            ui_groups.append(group)
                            adetailers.append(group.render(n, is_img2img, webui_info))

        for i, ui_group in enumerate(ui_groups):
            infotext.register_unit(i, ui_group)

        self.infotext_fields = infotext.infotext_fields
        self.paste_field_names = infotext.paste_field_names

        return tuple(adetailers)

    @torch.no_grad()
    def postprocess_image(self, p: StableDiffusionProcessing, pp: PostprocessImageArgs, *args):
        enabled_units = get_enabled_units(args)

        if not enabled_units or len(enabled_units) == 0:
            return

        if hasattr(p, "image_mask") and bool(p.image_mask):
            logger.info(f"img2img inpainting detected. ADetailer disabled.")
            return

        logger.info(f"Start processing.")

        # For torch >= 2.6, we need to set the environment variable
        # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

        _weight_only_load = os.getenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        if _weight_only_load is None:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        params_txt_content = Path(paths.data_path, "params.txt").read_text("utf-8")

        pp.image = ensure_rgb_image(pp.image)
        init_image = copy.copy(pp.image)

        processed = False
        with pause_total_tqdm():
            for i, unit in enumerate(enabled_units):
                processed |= afterdetailer_process_image(i, unit, p, pp, args)

        if processed and not getattr(unit, "ad_skip_img2img", False):
            save_image(
                p, init_image, condition="ad_save_images_before", suffix="-ad-before"
            )

        Infotext.write_infotext(enabled_units, p)
        Infotext.write_params_txt(params_txt_content)

        if _weight_only_load is None:
            del os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = _weight_only_load

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(on_after_component)
script_callbacks.on_infotext_pasted(Infotext.on_infotext_pasted)
script_callbacks.on_app_started(add_api_endpoints)
script_callbacks.on_before_ui(on_before_ui)