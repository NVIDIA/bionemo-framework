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

import functools
from pathlib import Path
from typing import Tuple

import pytest
import pytorch_lightning as pl
import torch
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm as nllm
from nemo.lightning import io, resume
from nemo.lightning.nemo_logger import NeMoLogger
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset

from bionemo import esm2
from bionemo.core.data.resamplers import PRNGResampleDataset
from bionemo.esm2.api import ESM2Config, ESM2GenericConfig
from bionemo.esm2.data import tokenizer
from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.model.finetune_token_classifier import ESM2FineTuneSeqLenBioBertConfig, Label2IDTokenizer
from bionemo.llm.data import collate
from bionemo.llm.lightning import LossLoggingCallback
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.utils.datamodule_utils import infer_num_samples
from bionemo.testing import megatron_parallel_state_utils
from bionemo.testing.callbacks import MetricTracker
from bionemo.testing.data.load import load


bionemo2_root: Path = (
    # esm2 module's path is the most dependable --> don't expect this to change!
    Path(esm2.__file__)
    # This gets us from 'sub-packages/bionemo-esm2/src/bionemo/esm2/__init__.py' to 'sub-packages/bionemo-esm2'
    .parent.parent.parent.parent
    # From here, we want to get to the root of the repository: _before_ sub-packages/
    .parent.parent
).absolute()
assert bionemo2_root != Path("/")
nemo1_checkpoint_path: Path = load("esm2/nv_650m:1.0")


@pytest.fixture(scope="module")
def esm2_config() -> ESM2Config:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        yield ESM2Config()


@pytest.fixture
def pretrain_data_module(dummy_protein_dataset, dummy_parquet_train_val_inputs):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs
    data_module = ESMDataModule(
        train_cluster_path=train_cluster_path,
        train_database_path=dummy_protein_dataset,
        valid_cluster_path=valid_cluster_path,
        valid_database_path=dummy_protein_dataset,
        global_batch_size=8,
        micro_batch_size=4,
        min_seq_length=None,
        max_seq_length=1024,
    )
    yield data_module


class PerTokenValueDataset(Dataset):
    def __init__(self, tokenizer):
        self.data = [
            (
                "6o41-O",
                "MTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEHHHHH",
                "CCCCCEEEEEEEECCCEEEEEEEEECCHHHHHHHHHHHHHHCCCCCEEEEECCCCEEEEECCCCCC",
            ),
            (
                "6o43-A",
                "MNDQEKIDKFTHSYINDDFGLTIDQLVPKVKGYGRFNVWLGGNESKIRQVLKAVKEIGVSPTLFAVYEKNEGFSSGLGWLNHTSARGDYLTDAKFIARKLVSQSKQAGQPSWYDAGNIVHFVPQDVQRKGNADFAKNMKAGTIGRAYIPLTAAATWAAYYPLGLKASYNKVQNYGNPFLDGANTILAWGGKLDGKGGSPS",
                "CCHHHHHHHHHCCECCCCCCCCHHHHHHHHCCCHHHHHCCCCCHHHHHHHHHHHHHHCCCHHHHHHHHHHHCCCCCECHHHCCECCCCHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCCCHHHHHHHHHHHHHCCCCCEHHHHHHHHCHHHHHHHHCHHHHCHHHHCCCCCCCHHHHHHHHHHHCCCCCCCCCCCCC",
            ),
            (
                "6o5k-B",
                "GPGFMRDSGSKASSDSQDANQCCTSCEDNAPATSYCVECSEPLCETCVEAHQRVKYTKDHTVRSTGPAKTRD",
                "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCEEECCCCEEECHHHHHHHHHCCCCCCCCEEECCCCCCCC",
            ),
            (
                "6o6d-A",
                "SNALSRNEVLLNGDINFKEVRCVGDNGEVYGIISSKEALKIAQNLGLDLVLISASAKPPVCKVMDYNKFRYQNEKKIKEA",
                "CCCCCCCCCCEHHHCCCCCEEEEECCCCEEEEECHHHHHHHHHHHCCEEEEEECCCCCCEEEEECHHHHHHHCCCCCCCC",
            ),
            (
                "6o6j-A",
                "MAKGKSEVVEQNHTLILGWSDKLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVICRSGSPLILADLKKVSVSKARTIIVLAEDGNADQSDARALRTVLSLTGVKEGLRGHIVVEMSDLDNEVLVKLVGGDLVETVVAHDVIGRLMIQCARQPGLAQIWEDILGFENCEFYIKRWPQLDGMLFEDVLISFPAAIPCGIKVASYGGKIILNPDDSYVLQEGDEVLVIAEDDDTYAPAPLPMVRRGSLPKDFVYPKSPERILFCGWRRDMEDMITVLDASLAPDSELWMFNDVPEKEREKKLIDGGLDISRLENISLVNREGNAVIRRHLESLPLESFDSILILADESVEDSAIQADSRSLATLLLIRDIQARRLPYVAMASQTQGGNFSKGSWIGEMKQASDKTVIISEILDPRTKNLLSMSKISDYVLSNELVSMALAMVAEDRQINDVLEELFAEEGNEMHIRQADIYLREGEEMSFYEIMLRARQRREILIGYRLANAERAVINPPAKTGRRKWSLKDVFVVITEKEGSRSHHHHHH",
                "CCCCCCCCCCCCCEEEECCCCCHHHHHHHHHHHHHHHCCCEEEEEECCCHHHHHHHHHCCCCCCCCCEEEEEECCCCCHHHHHHCCHHHCCEEEECCCCCCHHHHHHHHHHHHHHHHCCCCCCCCEEEEEECCHHHHHHHHHHHHHHEEEEEHHHHHHHHHHHHHHCCCHHHHHHHHHCCCCCEEEEECCHHHCCCEHHHHCCCCCCEEEEEEEEHHHCCEEEECCCCCCECCCCCEEEEEECCCCCCCCCCCCCCCCCCCCCCCCCCCCCCEEEEECCCCCHHHHHHHHHHHCCCCEEEEEEECCCHHHHHHHHHHCCCCHHHCCCEEEEEEECCCCCHHHHHCCCHHHCCEEEECCCCCCCCCHHHHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCEEEEEECCHHHHHHCCCCHHHHHEEECHHHHHHHHHHHHCCCHHHHHHHHHCCCCEEEEEEEHHHCCCCCCEEEHHHHHHHHHCCCEEEEEEECCCCCCCEECCCCCCCEEECCCCCEEEEEEECCCCCCCCCCCC",
            ),
            (
                "6o7b-A",
                "MGSSHHHHHHSQDPMEIDELTALGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLFSEFHHKGHMKNDELMIRRIKELSPERFGLTMEDVLNALWIVYEADNLASGEREEGQPQASRPLYSVFNPGKAYPWAELDFEKELPVPGDVFSIRSQDYRELVKRLWEELSKAKLRSDRLLPVLEKYLTFVSSVTSEGNIISLYDHMRMTSAIALAMLRAGCTAEDVRSGRCRKEKRFLLIEGDFSGIQDFIYRVSGKGTLKYLRARSAYLELIGWDVVLEILSRLGLTRANVVFNAGGHFMIIAQNTPDAVKELEEIRAKAVEWLYREFESDLYLAIEWEPVSGREFGREGGKNLFAEARKRLKHKLTVRKLKRFGEIKGLFEHGHTERLAECPVCGRELPEGKLEPSASDPETKVCPTCNRLVSLGGNLPKLLGFGRTAKNDAGVLVEGPFSGFVPYLQGGRPVGEQILVKNTLNPGEIPESAQFVPYFVADYFKKDPKGGVATFEELSMASTGTRRLGVMKGDVDRLGEFFSSMDSPSKLATASRFMDYFFKGYIGAIIEGKFGYIIGDVPSLRDWPEEPDIVVVYAGGDAFFIVGAWDQIFELAFRVRRAFNAYTGGKLTLSVGLGYFDERTPIYRMADVVSERLDTAKDEGRNRVFVVGRSRPLDGKHKLSYEWNHYEELWRTYAPRIYAGNGRLKGKLESKKGLLWKLLEIRELYVRDPNDVRWAYLTAYLLGRHGLSDLFPELVGIDTKAVERKEPQPVYWVDGVLKIVLMAVRR",
                "CCCCCCCCCCCCCCCCHHHHHHHHHHHCCCHHHHHCCCCCCCCHHHHHHHHHHHHHHCCCCCHHHHHHCCCCCCCCCCCCCCHHHHHHHHHCCHHHHCCCHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCECCCCCCCECCCCCCCCCCCCCCCECCCCCCCCHHHHHHHHHHHHHCCCCCCHHHHHHHHHHHHCCCECCCCCCCCCEHHHHHHHHHHHHHHHHHCCCCCCCCCCCHHHHCCCEEEEEEEEECHHHHHHCCCCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHCCCHHHEEEEECCEEEEEEECCHHHHHHHHHHHHHHHHHHHHHHCCCCEEEEEEEEECHHHHCCCCCCCHHHHHHHHHHHHHHHHHCCCCCCCHHHHCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCEEEEECCCCCCCCEECCCCEEEEECCCCCCCCCEEEEECCCCCCCCCCCCEEEEECCCCCCCECCCCCECCHHHHHHCCCCCCCEEEEEEEECCHHHHHHCCCCHHHHHHHHHHHHHHHHHHHHHHHHCCCHHHHCCCCCCCCCCCCCCEEEEEEECCEEEEEEEHHHHHHHHHHHHHHHHHHCCCCCCEEEEEEEECCCCCHHHHHHHHHHHHHHHHHCCCCEEECCCCCCCCCCCCCCCEEHHHHHHHHHHHHHHHEEECCEECHHHCCCCHHHHHHHHHHHHHHHCCCCCHHHHHHHHCCCCCCCCCCCHHHCCCCHHHHHCCCCCHHHCCHHHHHHHHHHHCC",
            ),
            (
                "6o7b-B",
                "MPKFIAVKLIPKGPFRDIPRADTLFGAIGNAISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKPLSVEPALEGILTGLDEEERYTTAKRLRKAKYLDLKNFELALRLRPFTIPEEIPYARVDVPRVVLDRVTQDSSIYFWEEIRFREKSGVYFLYSGPREVFDGYIAPAMRFLGDTGIGGKSTWGAGLFEVEFHEMKIDAPGSEYSVTLSNALPTKTPVLWRLLRKGGWSFGRRKPRMTFIAEGSIVKNDPGGMERLELGLSHEVYVYGLTFPLGVELPEGLE",
                "CCEEEEEEEEECCCECCCCCHHHHHHHHHHHHHHHHCHHHHHHHHHHHHCCCEECCCEEEECCEEEEECCHHHCCCCCCCCCCCCCCCHHHHHHHHCCCCEEEHHHHHHHHCCCCCCCCCCCCEEEEEEEEEECCCCCCCCEEEEEEEEEECCCEEEEEEEECCHHHHHHCHHHHHHHHHCCCCCCCCCCCCCCEEEEEEEEECCCCCCCCEEECCCECCCCCCCECCEEECCCCCCCCCCCCCEEECCCCEECCCCECCEEECCCCCCCEEECCECCEEECCCCCCCC",
            ),
        ]
        self._len = len(self.data)
        self.tokenizer = tokenizer
        label_tokenizer = Label2IDTokenizer()
        self.label_tokenizer = label_tokenizer.build_vocab("CHE")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        sequence = self.data[idx][1]
        tokenized_sequence = self._tokenize(sequence)
        # Overall mask for a token being masked in some capacity - either mask token, random token, or left as-is
        loss_mask = ~torch.isin(tokenized_sequence, torch.tensor(self.tokenizer.all_special_ids))
        labels = self._tokenize_labels(self.data[idx][2])

        return {
            "text": tokenized_sequence,
            "types": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
            "attention_mask": torch.ones_like(tokenized_sequence, dtype=torch.int64),
            "labels": labels,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
        }

    def _tokenize_labels(self, labels_sequence: str) -> torch.Tensor:
        label_ids = torch.tensor(self.label_tokenizer.text_to_ids(labels_sequence))

        # # for multi-label classification with BCEWithLogitsLoss
        # tokenized_labels = torch.nn.functional.one_hot(label_ids, num_classes=self.label_tokenizer.vocab_size)
        # cls_eos = torch.full((1, self.label_tokenizer.vocab_size), -1, dtype=tokenized_labels.dtype)

        # for multi-class (mutually exclusive) classification with CrossEntropyLoss
        tokenized_labels = label_ids
        cls_eos = torch.tensor([-1], dtype=tokenized_labels.dtype)

        # add cls / eos labels with padding value -1 to have the same shape as tokenized_sequence
        labels = torch.cat((cls_eos, tokenized_labels, cls_eos))
        return labels

    def _tokenize(self, sequence: str) -> torch.Tensor:
        """Tokenize a protein sequence.

        Args:
            sequence: The protein sequence.

        Returns:
            The tokenized sequence.
        """
        tensor = self.tokenizer.encode(sequence, add_special_tokens=True, return_tensors="pt")
        return tensor.flatten()  # type: ignore


class PerTokenValueDataModule(pl.LightningDataModule):  # noqa: D101
    def __init__(
        self,
        seed: int | None = 42,
        min_seq_length: int | None = None,
        max_seq_length: int = 1024,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        num_workers: int = 10,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        rampup_batch_size: list[int] | None = None,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        mask_random_prob: float = 0.1,
        tokenizer: tokenizer.BioNeMoAutoTokenizer = tokenizer.get_tokenizer(),
    ) -> None:
        super().__init__()
        self._seed = seed
        self._min_seq_length = min_seq_length
        self._max_seq_length = max_seq_length
        self._mask_prob = mask_prob
        self._mask_token_prob = mask_token_prob
        self._mask_random_prob = mask_random_prob
        self._tokenizer = tokenizer

        self._micro_batch_size = micro_batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory

        self.data_sampler = MegatronDataSampler(
            seq_len=max_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            dataloader_type="single",  # `MegatronPretrainingRandomSampler` from "cyclic" is failing.
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str) -> None:
        """Setup the ESMDataModule.

        Args:
            stage: Unused.

        Raises:
            RuntimeError: If the trainer is not attached, or if the trainer's max_steps is not set.
        """
        del stage  # Unused.

        if not hasattr(self, "trainer") or self.trainer is None:
            raise RuntimeError("Setup should be completed when trainer and config are attached.")

        if self.trainer.max_epochs is not None and self.trainer.max_epochs > 1:
            logging.warning(
                "Trainer is set to run for multiple epochs. This is not recommended due to the same shuffle being used "
                "in each. Instead set max_epochs to 1 and increase the number of max_steps."
            )

        max_train_steps = self.trainer.max_steps
        if max_train_steps <= 0:
            raise RuntimeError("Please specify trainer.max_steps")

        # Create training dataset
        _train_ds = PerTokenValueDataset(tokenizer=self._tokenizer)
        num_train_samples = int(max_train_steps * self.data_sampler.global_batch_size)

        self._train_ds = self._sample_and_shuffle_dataset(
            _train_ds, num_train_samples, "train"
        )  # shuffle manually without cyclic MegatronPretrainingRandomSampler

        # Create validation dataset
        _valid_ds = PerTokenValueDataset(tokenizer=self._tokenizer)
        num_val_samples = infer_num_samples(
            limit_batches=self.trainer.limit_val_batches,
            num_samples_in_dataset=len(_valid_ds),
            global_batch_size=self.data_sampler.global_batch_size,
            stage="val",
        )
        self._valid_ds = self._sample_and_shuffle_dataset(
            _valid_ds, num_val_samples, "val"
        )  # shuffle manually without cyclic MegatronPretrainingRandomSampler

        assert (
            hasattr(self, "trainer") and self.trainer is not None
        ), "Setup should be completed when trainer and config are attached."

    def _create_dataloader(self, dataset, **kwargs) -> torch.utils.data.DataLoader:
        assert self._tokenizer.pad_token_id is not None, "Tokenizer must have a pad token id."

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            collate_fn=functools.partial(
                collate.bert_padding_collate_fn,
                padding_value=self._tokenizer.pad_token_id,
                min_length=self._min_seq_length,
                max_length=self._max_seq_length,
            ),
            **kwargs,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Returns the dataloader for training data."""
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Returns the dataloader for validation data."""
        return self._create_dataloader(self._valid_ds)

    def _sample_and_shuffle_dataset(self, dataset: Dataset, num_samples: int, stage: str):  # noqa: D417
        """Sample the training dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from

        Returns:
            ResamplingMappedDataset: Resampled dataset

        """
        # This is where re-sampling occurs.
        return PRNGResampleDataset(
            dataset,
            num_samples=num_samples,
            seed=self._seed + len(stage),
        )


def _train_model(
    name: str,
    root_dir: Path,
    config: ESM2GenericConfig,
    data_module: pl.LightningDataModule,
    n_steps_train: int,
    tokenizer: tokenizer.BioNeMoAutoTokenizer,
) -> Tuple[Path, MetricTracker, nl.Trainer]:
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_best_model=False,
        save_last=True,
        save_on_train_epoch_end=True,
        monitor="reduced_train_loss",  # TODO find out how to get val_loss logged and use "val_loss",
        every_n_train_steps=n_steps_train // 2,
        enable_nemo_ckpt_io=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
    )

    # Setup the logger and train the model
    nemo_logger = NeMoLogger(
        dir=str(root_dir),
        name=name,
        tensorboard=TensorBoardLogger(save_dir=root_dir, name=name),
        ckpt=checkpoint_callback,
    )
    # Needed so that the trainer can find an output directory for the profiler
    # ckpt_path needs to be a string for SerDe
    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=5e-4,
            optimizer="adam",
            use_distributed_optimizer=True,
            fp16=config.fp16,
            bf16=config.bf16,
        )
    )
    module = BioBertLightningModule(config=config, tokenizer=tokenizer, optimizer=optimizer)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=True,
    )
    metric_tracker = MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"])
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        limit_val_batches=2,
        val_check_interval=n_steps_train // 2,
        max_steps=n_steps_train,
        num_nodes=1,
        log_every_n_steps=n_steps_train // 2,
        callbacks=[LossLoggingCallback(), metric_tracker],
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )
    nllm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            path=None,  # Overrides the path found by resume_if_exists when set.
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
    ckpt_path = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    return ckpt_path, metric_tracker, trainer


@pytest.mark.needs_gpu
def test_esm2_finetune_token_classifier(
    tmpdir, esm2_config, tokenizer, pretrain_data_module, n_steps_train: int = 50, seed: int = 42
):
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        ckpt_path, initial_metrics, _ = _train_model(
            name="test_experiment",
            root_dir=tmpdir / "pretrain",
            config=esm2_config,
            data_module=pretrain_data_module,
            n_steps_train=n_steps_train,
            tokenizer=tokenizer,
        )
        assert ckpt_path.exists()
        assert ckpt_path.is_dir()
        assert io.is_distributed_ckpt(ckpt_path)
        assert initial_metrics.collection_train["loss"][0] > initial_metrics.collection_train["loss"][-1]

    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        esm2_finetune_config = ESM2FineTuneSeqLenBioBertConfig(initial_ckpt_path=str(ckpt_path))
        finetune_data_module = PerTokenValueDataModule()
        simple_ft_checkpoint, simple_ft_metrics, _ = _train_model(
            name="finetune_new_head",
            root_dir=tmpdir / "finetune_new_head",  # new checkpoint will land in a subdir of this
            config=esm2_finetune_config,  # same config as before since we are just continuing training
            data_module=finetune_data_module,
            n_steps_train=n_steps_train,
            tokenizer=tokenizer,
        )
        assert simple_ft_checkpoint.exists()
        assert simple_ft_checkpoint.is_dir()
        assert io.is_distributed_ckpt(simple_ft_checkpoint)
        assert simple_ft_metrics.collection_train["loss"][0] > simple_ft_metrics.collection_train["loss"][-1]
