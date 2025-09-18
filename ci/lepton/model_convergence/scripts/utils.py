from pathlib import Path
import textwrap, json
from omegaconf import OmegaConf

TEMPLATE_PATH = Path(__file__).parent / "wrap_template.sh"

def render_wrapper_string(script: str, all_config_json: str) -> str:
    tpl = TEMPLATE_PATH.read_text(encoding="utf-8")
    script_indented = textwrap.indent(script.rstrip("\n"), "  ")
    return (tpl
        .replace("__SCRIPT__", script_indented)
        .replace("__ALL_CONFIG_JSON__", all_config_json))