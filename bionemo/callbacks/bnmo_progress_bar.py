# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
from time import time
from typing import Any, Dict

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar, convert_inf
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm as _tqdm
from typing_extensions import override


class BnmoTqdm(Tqdm):
    def __init__(self, *args: Any, warmup_n: int = 0, **kwargs: Any):
        super(BnmoTqdm, self).__init__(*args, **kwargs)
        self.warmup_n = warmup_n
        self.warmup_t = None

    @property
    def format_dict(self) -> Dict:
        """In tqdm, this is where the current time is obtained."""
        # time_now = self._time()
        time_now = time()
        formatted_dict = super(Tqdm, self).format_dict

        elapsed = time_now - self.start_t if hasattr(self, "start_t") else 0  # same as tqdm
        formatted_dict["elapsed"] = elapsed
        if self.warmup_t is not None:
            elapsed_since_warmup = time_now - self.warmup_t
            formatted_dict["rate"] = (self.n - self.warmup_n) / elapsed_since_warmup
        else:
            formatted_dict["rate"] = 10**-6

        return formatted_dict


class BnmoTQDMProgressBar(TQDMProgressBar):
    """A slight modifification of pytorch lighting's TQDMProgressBar, to improve
    the accuracy of the rate estimate by including a warmup phase for the
    rate estimate in the training progress bar.  The early training steps are
    excluded from the rate estimate.

    Empirically it's found that the early steps of a training run can take 3+ times
    as much wall-clock time as later training steps.  So for training time
    measurement, it can be helpful to remove these data points.

    Attributes:
        BAR_FORMAT: All instance of this class will have this format for the progress bar,
            with placeholder variables interpreted by an embedded/contained
            instance of the tqdm class.
    """

    BAR_FORMAT: str = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}{postfix}]"

    def __init__(self, refresh_rate: int = 1, process_position: int = 0, warmup_n: int = 0):
        """Initializes an instance configured by arguments.

        Args:
            refresh_rate: The number of steps between the updating-and-printing of
                the progress bar.  A value of 0 disables the printing of the
                progress bar.
            process_position: An offset to the first character where the progress
                bar is printed.  A values less-than-or-equal to zero results in
                no offset.
            warmup_n: The rate estimate appearing in the training progress bar is
                invalid for the first warmup_n training steps. Subsequently,
                the training progress bar rate estimate is

                (number-of-training-steps-so-far - warmup_n) /
                (time-at-end-of-the-current-training-step - time-at-end-of-warmup_nth-training-step)

                The value should be a non-negative integer, and is passed to an
                instnace of BnmoTqdm.

        For reference, see
        https://github.com/Lightning-AI/pytorch-lightning/blob/76b691d80c6c5203c66365272ce246ac86e418f0/src/lightning/pytorch/callbacks/progress/tqdm_progress.py#L65 # noqa
        """
        super().__init__(refresh_rate, process_position)
        self.pre_prefix = type(self).__name__ + ","

        # check and store subclass parameters
        assert warmup_n >= 0
        self.warmup_n = warmup_n

    @override
    def on_train_start(self, *_: Any) -> None:
        self.train_progress_bar = self.init_train_tqdm()

    @override
    def init_train_tqdm(self) -> BnmoTqdm:
        """Create the tqdm object used for training progress.

        TQDMProgressBar make very limited usage of the tqdm object:
            (1) tqdm.update(..) is not called by TQDMProgressBar
            (2) tqdm.__iter__ is not called by TQDMProgressBar

        As a result:
            (a) The callable EMA instance for dt, and the callabel EMA
            instance for dn, are initialized with self.last=0,
            but only called with default arg x=None, so are never updated,
            and always return self.last = 0

            (b) The rate returned by tqdm.format_dict is None, and so is
            computed in tqdm.format_meter
                if rate is None and elapsed:
                    rate = (n - initial) / elapsed
                    inv_rate = 1 / rate if rate else None

            (c) self.initial is returned in the return value from format_dict,
            and used in format_meter
                - if refresh is called in tqdm.__init__, then initial must
                be set in TQDM.__init__, otherwise the value is not relevant
                at the stage of TQDM.__init__

            (d) The tqdm class is effectively used for
                - file object io
                - standardized bar format
                - time elapsed, and total count of batches processed


        With smoothing=0, the exponential moving averages in the computation
        of the rate, would be the full-term averages

        The arguments to Tqdm, are passed to the super class tqdm,
        initial is the intial value for n and last_print_n.

        The value for delay is a number of seconds, in tqdm.__init__, determines
        whether or not refresh is called.  This is the only impact for
        TQDMProgressBar

        See
        https://github.com/Lightning-AI/pytorch-lightning/blob/438f29f07a803309b2c83801cda1625d3bc8eb64/src/lightning/pytorch/callbacks/progress/tqdm_progress.py#L39  # noqa
        https://github.com/tqdm/tqdm/blob/cc372d09dcd5a5eabdc6ed4cf365bdb0be004d44/tqdm/std.py#L245  # noqa
        """
        return BnmoTqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
            initial=-1,  # with delay>0, this value irrelevent for tqdm
            delay=10**-3,  # choose value to skip tqdm.refresh() in tqdm.__init__
            warmup_n=self.warmup_n,
        )

    @override
    def on_train_epoch_start(self, trainer: Trainer, *_: Any) -> None:
        """Reset the state of relevant tqdm objects, e.g.
            tqdm.start_t
            tqdm.n

        self.total_train_batches is a property implemented by superclass
        ProgressBar, and returns the value of trainer.num_training_batches see
        https://github.com/Lightning-AI/pytorch-lightning/blob/438f29f07a803309b2c83801cda1625d3bc8eb64/src/lightning/pytorch/callbacks/progress/progress_bar.py#L81

        """

        # tqdm.reset:
        #   self.n = 0
        #   if total is not None:
        #       self.total = total
        #   if self.disable:
        #       return
        #   self.last_print_n = 0
        #   self.last_print_t = self.start_t = self._time()
        #   self._ema_dn = EMA(self.smoothing)
        #   self._ema_dt = EMA(self.smoothing)
        #   self._ema_miniters = EMA(self.smoothing)
        #   self.refresh()
        #
        #  https://github.com/tqdm/tqdm/blob/cc372d09dcd5a5eabdc6ed4cf365bdb0be004d44/tqdm/std.py#L1360
        #
        #  tqdm.initial is used in format_dict and format_meter for rate
        #
        self.train_progress_bar.initial = 0
        self.train_progress_bar.reset(total=convert_inf(self.total_train_batches))

        if self.warmup_n == 0:
            _bnmo_do_at_warmup_n(self.train_progress_bar)
        self.train_progress_bar.set_description(
            desc=f"{self.pre_prefix} Training Epoch {trainer.current_epoch}", refresh=False
        )

    @override
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _bnmo_update_n(self.train_progress_bar, n)

            if self.warmup_n > 0 and self.train_progress_bar.n == self.warmup_n:
                _bnmo_do_at_warmup_n(self.train_progress_bar)

            self.train_progress_bar.set_postfix(ordered_dict=self.get_metrics(trainer, pl_module))


def _bnmo_do_at_warmup_n(bar: _tqdm) -> None:
    bar.warmup_t = time()


def _bnmo_update_n(bar: _tqdm, value: int) -> None:
    if not bar.disable:
        bar.n = value
