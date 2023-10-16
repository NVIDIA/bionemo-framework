import os

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


NEMO_APPEND_SEARCH_PATH: str = os.getenv(
    "NEMO_APPEND_SEARCH_PATH", "/workspace/nemo/examples/nlp/language_modeling/conf"
)


class NeMoConfigSearchPathConfig(SearchPathPlugin):
    """
    Hydra SearchPathPlugin that appends location of the default NeMo configs to the list of paths using env variable
    NEMO_APPEND_SEARCH_PATH that hydra searches in order to find non-primary configs.
    Consequently, the searchpath with NeMo configs is automatically used in every hydra search and there is no need to
    attach this path to yaml configs.

    When a config is requested, the first matching config in the search path is selected, hence a user needs to be
    careful to name configs uniquely especially regarding file names in NEMO_APPEND_SEARCH_PATH.

    The files in thedefault NeMo (as well as BioNeMo) container under /workspace/nemo/examples/ correspond to files
    in /opt/nvidia/nemo/examples if no NEMO_PATH is mounted during ./launch.sh dev.
    See NeMo's Dockerfile: https://github.com/NVIDIA/NeMo/blob/main/Dockerfile

    See hydra.searchpath details: https://hydra.cc/docs/advanced/search_path/

    *** USAGE ****
    NEMO_APPEND_SEARCH_PATH=<PATH> python ....

    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="nemo-config-searchpath-plugin", path=f"file://{NEMO_APPEND_SEARCH_PATH}")
