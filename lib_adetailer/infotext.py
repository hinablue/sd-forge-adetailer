from typing import List, Tuple, Union

import os
import gradio as gr

from modules import paths
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img

from lib_adetailer.logger import logger_adetailer as logger

from lib_adetailer.args import ADetailerUnit


def field_to_displaytext(fieldname: str) -> str:
    return " ".join([word.capitalize() for word in fieldname.split("_")])

def displaytext_to_field(text: str) -> str:
    return "_".join([word.lower() for word in text.strip().split(" ")]).replace('ad_', '')

def parse_value(value: str) -> Union[str, float, int, bool]:
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # Plain string.


def serialize_unit(unit: ADetailerUnit) -> str:
    log_value = {
        field_to_displaytext(field): getattr(unit, field)
        for field in ADetailerUnit.infotext_fields()
        if getattr(unit, field) != -1
        # Note: exclude hidden slider values.
    }

    return ", ".join(f"{field}: {value}" for field, value in log_value.items())


def parse_unit(text: str) -> ADetailerUnit:
    return ADetailerUnit(
        enabled=True,
        **{
            displaytext_to_field(key): parse_value(value)
            for item in text.split(",")
            for (key, value) in (item.strip().split(":"),)
        },
    )

class Infotext(object):
    def __init__(self) -> None:
        self.infotext_fields: List[Tuple[gr.components.IOComponent, str]] = []
        self.paste_field_names: List[str] = []

    @staticmethod
    def unit_prefix(unit_index: int) -> str:
        return f"ADetailer {unit_index}"

    def register_unit(self, unit_index: int, uigroup) -> None:
        unit_prefix = Infotext.unit_prefix(unit_index)
        for field in ADetailerUnit.infotext_fields():
            io_component = getattr(uigroup, field)
            component_locator = f"{unit_prefix} {field}"
            self.infotext_fields.append((io_component, component_locator))
            self.paste_field_names.append(component_locator)

    @staticmethod
    def write_infotext(
        units: List[ADetailerUnit], p: StableDiffusionProcessing
    ):
        """Write infotext to `p`."""
        p.extra_generation_params.update(
            {
                Infotext.unit_prefix(i): serialize_unit(unit)
                for i, unit in enumerate(units)
                if unit.ad_enabled
            }
        )

    @staticmethod
    def write_params_txt(info: str):
        with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
            file.write(info)

    @staticmethod
    def on_infotext_pasted(infotext, results):
        updates = {}
        for k, v in results.items():
            if not k.startswith("ADetailer"):
                continue

            assert isinstance(v, str), f"Expected string but got {v}."
            try:
                for field, value in vars(parse_unit(v)).items():
                    component_locator = f"{k} {field}"
                    updates[component_locator] = value
            except Exception as e:
                logger.warn(f"Failed to parse infotext value:\n{v}")
                logger.warn(f"Exception: {e}")
            break
        results.update(updates)
