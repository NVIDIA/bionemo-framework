from typing import Optional

import dllogger as dll
from dllogger import DLLLoggerAlreadyInitialized, JSONStreamBackend, StdOutBackend, Verbosity
from nemo.utils import logging
from pytorch_lightning.utilities import rank_zero_only


class DLLogger:
    """
    DLLogger class for logging.
    See more details under: https://github.com/NVIDIA/dllogger
    """

    def __init__(
        self, json_file: Optional[str] = None, append_to_json: bool = True, use_existing_dllogger: bool = False
    ):
        """
        Args:
            json_file: path to the json file where logs of dllogger are saved
            append_to_json: append (True) or write (False) to json file
            use_existing_dllogger: in case a dllogger has been already initialized should it be used for logging or
                              a seperate object should be created?
        """

        super().__init__()
        self.logger = None
        self._initialize_dllogger(json_file, append_to_json, use_existing_dllogger)

    @rank_zero_only
    def _initialize_dllogger(self, json_file, append_to_json: bool = True, use_existing_dllogger: bool = False):
        backends = [
            StdOutBackend(Verbosity.VERBOSE),
        ]
        if json_file is not None:
            backends.append(JSONStreamBackend(Verbosity.VERBOSE, json_file, append=append_to_json))
        try:
            dll.init(backends=backends)
            self.logger = dll
        except DLLLoggerAlreadyInitialized:
            msg = "DLLogger already initialized..."
            if not use_existing_dllogger:
                self.logger = dll.Logger(backends=backends)
                msg += f"Initialized seperate dllogger that logs to: {json_file}"
            else:
                msg += "Appending logging to the existing dllogger"
                self.logger = dll
            logging.warning(msg)

    @rank_zero_only
    def log_metrics(self, metrics: dict, step: Optional[tuple] = None) -> None:
        if step is None:
            step = ()
        self.logger.log(step=step, data=metrics)

    @rank_zero_only
    def log_metadata(self, metric: str, metadata: dict) -> None:
        self.logger.metadata(metric, metadata)

    @rank_zero_only
    def flush(self) -> None:
        self.logger.flush()
