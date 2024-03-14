import gradio as gr

from modules import shared
from lib_adetailer import (
    AFTER_DETAILER,
    __version__,
)

from lib_adetailer.args import BBOX_SORTBY, SCRIPT_DEFAULT

def on_ui_settings():
    section = ("ADetailer", AFTER_DETAILER)
    shared.opts.add_option(
        "ad_max_models",
        shared.OptionInfo(
            default=2,
            label="Max models",
            component=gr.Slider,
            component_args={"minimum": 1, "maximum": 10, "step": 1},
            section=section,
        ),
    )

    shared.opts.add_option(
        "ad_extra_models_dir",
        shared.OptionInfo(
            default="",
            label="Extra path to scan adetailer models",
            component=gr.Textbox,
            section=section,
        ),
    )

    shared.opts.add_option(
        "ad_save_previews",
        shared.OptionInfo(False, "Save mask previews", section=section),
    )

    shared.opts.add_option(
        "ad_save_images_before",
        shared.OptionInfo(False, "Save images before ADetailer", section=section),
    )

    shared.opts.add_option(
        "ad_only_seleted_scripts",
        shared.OptionInfo(
            True, "Apply only selected scripts to ADetailer", section=section
        ),
    )

    textbox_args = {
        "placeholder": "comma-separated list of script names",
        "interactive": True,
    }

    shared.opts.add_option(
        "ad_script_names",
        shared.OptionInfo(
            default=SCRIPT_DEFAULT,
            label="Script names to apply to ADetailer (separated by comma)",
            component=gr.Textbox,
            component_args=textbox_args,
            section=section,
        ),
    )

    shared.opts.add_option(
        "ad_bbox_sortby",
        shared.OptionInfo(
            default="None",
            label="Sort bounding boxes by",
            component=gr.Radio,
            component_args={"choices": BBOX_SORTBY},
            section=section,
        ),
    )

    shared.opts.add_option(
        "ad_same_seed_for_each_tap",
        shared.OptionInfo(
            False, "Use same seed for each tab in adetailer", section=section
        ),
    )
