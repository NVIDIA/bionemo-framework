import os
import json
import numpy as np
import torch
import bisect
from bionemo.noodles.nvfaidx import NvFaidx
import lightning.pytorch as pl
import torch.distributed as dist
from nemo.utils import logging
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from torch.utils.data import Dataset, default_collate
from nemo.lightning.data import WrappedDataLoader
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils.import_utils import safe_import
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

_, HAVE_TE = safe_import("transformer_engine")

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class EdenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        fasta_file: str,
        seq_length: int = 8192,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 1,
        global_batch_size: int = 4,
        rampup_batch_size: Optional[List[int]] = None,
        train_val_test_split: List[float] = [0.8, 0.1, 0.1],
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        create_attention_mask: bool = False,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        rc_aug: bool = False,
        stride: int = 7992,  # 200bps overlap (8192 - 200)
        use_control_tags: bool = True,
        seed: int = 29,
        sequence_subset: Optional[List[str]] = None,
    ):
        super().__init__()
        self.fasta_file = fasta_file
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.create_attention_mask = create_attention_mask or not HAVE_TE
        self.rc_aug = rc_aug
        self.stride = stride if stride is not None else 7992
        self.use_control_tags = use_control_tags
        self.init_global_step = 0
        self.sequence_subset = sequence_subset
        self.seed = seed
        if tokenizer is None:
            self.tokenizer = get_nmt_tokenizer("byte-level")
        else:
            self.tokenizer = tokenizer

        # Megatron sampler
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def _compute_num_windows(self, seq_len: int) -> int:
        """Helper method to compute number of windows for a sequence."""
        if seq_len < self.seq_length:
            return 1
        else:
            return (seq_len - self.seq_length + self.stride) // self.stride

    def build(
        self,
        trainer_max_steps: int,
        trainer_val_check_interval: int,
        trainer_limit_val_batches: Union[int, float],
        trainer_limit_test_batches: Union[int, float],
    ):
        """
        Build the datasets using sequence data from FASTA file.

        Args:
            trainer_max_steps: The maximum number of training steps.
            trainer_val_check_interval: The validation check interval.
            trainer_limit_val_batches: The number of validation batches.
            trainer_limit_test_batches: The number of test batches.
        """
        # First, load the FASTA file to get the sequence names
        # sequences = Fasta(self.fasta_file)
        faidx_path = self.fasta_file + ".fai"
        assert os.path.exists(faidx_path), f"FAI file {faidx_path} does not exist"
        sequences = NvFaidx(
            self.fasta_file,
            faidx_path=faidx_path,
            ignore_existing_fai=False,
        )
        sequence_names = list(sequences.keys())

        # Shuffle the sequence names to ensure random distribution of sequences
        rng = np.random.RandomState(self.seed)
        rng.shuffle(sequence_names)

        # Group sequences by approximate length category
        short_seqs = []
        medium_seqs = []
        long_seqs = []

        for seq_name in sequence_names:
            seq_len = len(sequences[seq_name])

            # Categorize by length
            if seq_len < 10000:
                short_seqs.append(seq_name)
            elif seq_len < 100000:
                medium_seqs.append(seq_name)
            else:
                long_seqs.append(seq_name)

        # Calculate split sizes for each length category
        def split_category(category, split_ratios):
            total = len(category)
            train_size = int(total * split_ratios[0])
            val_size = int(total * split_ratios[1])
            return (
                category[:train_size],
                category[train_size : train_size + val_size],
                category[train_size + val_size :],
            )

        # Apply splits to each category
        train_val_test = self.train_val_test_split
        short_train, short_val, short_test = split_category(short_seqs, train_val_test)
        medium_train, medium_val, medium_test = split_category(
            medium_seqs, train_val_test
        )
        long_train, long_val, long_test = split_category(long_seqs, train_val_test)

        # Combine categories for final splits
        train_seq_names = short_train + medium_train + long_train
        val_seq_names = short_val + medium_val + long_val
        test_seq_names = short_test + medium_test + long_test

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"Sequence split: Train={len(train_seq_names)}, Val={len(val_seq_names)}, Test={len(test_seq_names)}"
            )

        # Create datasets for each split

        self._train_ds = EdenDataset(
            self.tokenizer,
            self.fasta_file,
            self.seq_length,
            self.create_attention_mask,
            stride=self.stride,
            rc_aug=self.rc_aug,
            use_control_tags=self.use_control_tags,
            sequence_subset=train_seq_names,
            faidx_path=faidx_path,
        )

        self._validation_ds = EdenDataset(
            self.tokenizer,
            self.fasta_file,
            self.seq_length,
            self.create_attention_mask,
            stride=self.stride,
            rc_aug=self.rc_aug,
            use_control_tags=self.use_control_tags,
            sequence_subset=val_seq_names,
            faidx_path=faidx_path,
        )

        self._test_ds = EdenDataset(
            self.tokenizer,
            self.fasta_file,
            self.seq_length,
            self.create_attention_mask,
            rc_aug=self.rc_aug,
            stride=self.stride,
            use_control_tags=self.use_control_tags,
            sequence_subset=test_seq_names,
            faidx_path=faidx_path,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"Dataset split: Train={len(self._train_ds)}, Val={len(self._validation_ds)}, Test={len(self._test_ds)}"
            )

            # Sanity check: ensure every window is in the dataset
            for split_name, ds, seq_names in [
                ("Train", self._train_ds, train_seq_names),
                ("Val", self._validation_ds, val_seq_names),
                ("Test", self._test_ds, test_seq_names),
            ]:
                total_windows = sum(
                    self._compute_num_windows(len(sequences[name]))
                    for name in seq_names
                )
                print(
                    f"{split_name}: computed windows = {total_windows}, "
                    f"dataset __len__ = {len(ds)}"
                )

    def setup(self, stage: str = "") -> None:
        """
        Setup the data module.
        """
        assert (
            hasattr(self, "trainer") and self.trainer is not None
        ), "Setup should be completed when trainer and config are attached."

        self.build(
            trainer_max_steps=self.trainer.max_steps,
            trainer_val_check_interval=self.trainer.val_check_interval,
            trainer_limit_val_batches=self.trainer.limit_val_batches,
            trainer_limit_test_batches=self.trainer.limit_test_batches,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Get the train dataloader.
        """
        return self._create_dataloader(self._train_ds, mode="train")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Get the validation dataloader.
        """
        return self._create_dataloader(self._validation_ds, mode="validation")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Get the test dataloader.
        """
        return self._create_dataloader(self._test_ds, mode="test")

    def _create_dataloader(self, dataset, mode, **kwargs) -> WrappedDataLoader:
        self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step
        dataloader = WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # Memory optimization, use it with multiple workers
            persistent_workers=self.persistent_workers,
            collate_fn=getattr(dataset, "collate_fn", default_collate),
            **kwargs,
        )
        return dataloader

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        consumed_samples = self.data_sampler.compute_consumed_samples(
            self.trainer.global_step - self.init_global_step
        )
        return {"consumed_samples": consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule stat

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """
        try:
            from megatron.core.num_microbatches_calculator import (
                update_num_microbatches,
            )

        except (ImportError, ModuleNotFoundError):
            logging.warning(
                "Megatron num_microbatches_calculator not found, using Apex version."
            )
            from apex.transformer.pipeline_parallel.utils import update_num_microbatches

        consumed_samples = state_dict["consumed_samples"]
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples

        update_num_microbatches(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        self.data_sampler.if_first_step = 1

    def reconfigure_limit_batches(self):
        """
        Reconfigure trainer.limit_train_batches and trainer.limit_val_batches in terms of num of microbatches.
        """
        # Override limit_train_batches in terms of num of microbatches
        self._reconfigure_limit_batches(
            self.trainer.limit_train_batches, self._train_ds, "train"
        )
        # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting
        #   in between a step
        self._reconfigure_limit_batches(
            self.trainer.limit_val_batches, self._validation_ds, "val"
        )

    def _reconfigure_limit_batches(self, limit_batches, dataloader, mode):
        """
        Reconfigure trainer.limit_val_batches for pretraining
        """
        # Override limit_batches in terms of num microbatches and so there are limit_batches//num_micro_batches
        #   num of global batches
        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning(
                "Megatron num_microbatches_calculator not found, using Apex version."
            )
            from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        if isinstance(limit_batches, int):
            limit_batches *= get_num_microbatches()
        else:
            assert isinstance(limit_batches, float)
            # Don't reconfigure if limit_batches is 0.0 or if there's no dataloader
            if limit_batches == 0.0 or dataloader is None:
                return
            # len(dataloader) returns len as num of microbatches
            dl_len_in_micro_batches = len(dataloader)
            if len(dataloader) != float("inf"):
                if limit_batches == 1.0:
                    limit_batches = dl_len_in_micro_batches
                else:
                    limit_micro_batches = int(dl_len_in_micro_batches * limit_batches)
                    if limit_micro_batches == 0 and limit_batches > 0.0:
                        min_percentage = 1.0 / len(dataloader)
                        raise ValueError(
                            f"You requested to check {limit_batches} of the val_dataloader but"
                            f" {limit_batches} * {len(dataloader)} < 1. Please increase the"
                            f" `limit_val_batches` argument. Try at least"
                            f" `limit_val_batches={min_percentage}`"
                        )
                    # Make sure trainer.limit_val_batches is a multiple of num of microbatches
                    if limit_micro_batches < get_num_microbatches():
                        limit_batches = get_num_microbatches()
                    else:
                        limit_batches = (
                            limit_batches - limit_batches % get_num_microbatches()
                        )

        if mode == "train":
            self.trainer.limit_train_batches = limit_batches
        else:
            self.trainer.limit_val_batches = limit_batches

        # Override num sanity steps to be a multiple of num of microbatches
        # self.trainer.num_sanity_val_steps *= get_num_microbatches()


class EdenDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        fasta_file: str,
        seq_length: int,
        create_attention_mask: bool = False,
        rc_aug: bool = False,
        stride: Optional[int] = 7992,
        use_control_tags: bool = False,
        sequence_subset: Optional[List[str]] = None,
        faidx_path: str = None,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.fasta_file = fasta_file
        self.create_attention_mask = create_attention_mask
        self.rc_aug = rc_aug
        self.stride = stride if stride is not None else 7992
        self.use_control_tags = use_control_tags
        self.sequence_subset = sequence_subset
        self.faidx_path = faidx_path

        self.sequences = NvFaidx(
            fasta_file,
            faidx_path=faidx_path,
            ignore_existing_fai=False,
        )

        # Summary of length distribution (optional)
        lengths = [len(self.sequences[name]) for name in self.sequences.keys()]
        num_total = len(lengths)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Total seqs: {num_total}")

        # Build sequence index and window metadata
        self.sequence_index = []  # list of tuples: (name, start_idx, end_idx, seq_len)
        total_windows = 0
        seq_names_to_process = (
            self.sequence_subset
            if self.sequence_subset is not None
            else self.sequences.keys()
        )

        for seq_name in seq_names_to_process:
            if seq_name not in self.sequences:
                continue
            L = len(self.sequences[seq_name])
            # Determine number of windows
            num_windows = self._compute_num_windows(L)
            self.sequence_index.append(
                (seq_name, total_windows, total_windows + num_windows, L)
            )
            total_windows += num_windows

        # Finalize index lists for bisect lookup
        self.length = total_windows
        self.starts = [start for (_, start, _, _) in self.sequence_index]

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"Created dataset with {self.length} sequence windows "
                f"from {len(self.sequence_index)} sequences"
            )

        # Prepare control-tag IDs
        if self.use_control_tags:
            self.ctrl_ids_map = {
                name: tokenizer.text_to_ids(f"<{name.lower()}>")
                for (name, _, _, _) in self.sequence_index
            }
        else:
            self.ctrl_ids_map = {}

        # Attention mask and position ids
        if create_attention_mask:
            self.attention_mask = (
                torch.tril(torch.ones((seq_length, seq_length))).unsqueeze(0) < 0.5
            )
        # Shared position_ids for memory efficiency
        if (
            not hasattr(EdenDataset, "_position_ids")
            or EdenDataset._position_ids.size(0) != seq_length
        ):
            EdenDataset._position_ids = torch.arange(seq_length, dtype=torch.int64)
        self.position_ids = EdenDataset._position_ids

        # Initialize consumption tracking
        self.access_counter = 0
        self.current_global_step = 0
        # self._init_consumption_tracking()  # Disabled to prevent checkpoint saving issues

    # Initialize CSV logging for data consumption tracking
    def _init_consumption_tracking(self):
        """Initialize CSV logging for data consumption tracking"""
        import csv
        import datetime

        # Try to get results path from environment or use current directory
        results_path = "/project-eden/datasets/"

        # Create separate log file for each rank to avoid conflicts
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()

        log_path = os.path.join(results_path, f"data_consumption_rank_{rank}.csv")

        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir:  # Only create if dirname is not empty
            os.makedirs(log_dir, exist_ok=True)

        self.consumption_log_path = log_path

        # Initialize CSV file with headers
        with open(self.consumption_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "access_order",
                    "dataset_idx",
                    "sequence_name",
                    "window_idx",
                    "start_pos",
                    "end_pos",
                    "sequence_length",
                    "timestamp",
                ]
            )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Data consumption logging initialized: {self.consumption_log_path}")

    def _log_data_access(self, idx: int):
        """Log each data access for tracking consumption"""
        import csv
        import datetime

        # Find sequence and window information
        i = bisect.bisect_right(self.starts, idx) - 1
        if i < 0 or idx >= self.sequence_index[i][2]:
            return  # Skip logging for invalid indices

        seq_name, start_idx, end_idx, seq_len = self.sequence_index[i]
        rel_idx = idx - start_idx
        start_pos = rel_idx * self.stride
        end_pos = min(start_pos + self.seq_length, seq_len)

        # Increment access counter
        self.access_counter += 1

        # Get current timestamp
        timestamp = datetime.datetime.now().isoformat()

        # Log to CSV
        try:
            with open(self.consumption_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.access_counter,
                        idx,
                        seq_name,
                        rel_idx,
                        start_pos,
                        end_pos,
                        seq_len,
                        timestamp,
                    ]
                )
        except Exception as e:
            # Don't crash training if logging fails
            print(f"Warning: Failed to log data access: {e}")

    def __len__(self) -> int:
        return self.length

    def _compute_num_windows(self, seq_len: int) -> int:
        """Helper method to compute number of windows for a sequence."""
        if seq_len < self.seq_length:
            return 1
        else:
            return (seq_len - self.seq_length + self.stride) // self.stride

    def reverse_complement(self, seq: str) -> str:
        cmap = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        return "".join(cmap.get(b, b) for b in reversed(seq))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Log data consumption
        # self._log_data_access(idx)  # Disabled to prevent checkpoint saving issues

        # Locate window via bisect
        i = bisect.bisect_right(self.starts, idx) - 1
        if i < 0 or idx >= self.sequence_index[i][2]:
            raise IndexError(f"Index {idx} out of bounds")

        seq_name, start_idx, end_idx, seq_len = self.sequence_index[i]
        rel_idx = idx - start_idx
        start_pos = rel_idx * self.stride

        # Build token window
        ctrl_ids = self.ctrl_ids_map.get(seq_name, [])
        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        sep_id = self.tokenizer._sep_id
        pad_id = self.tokenizer.pad_id

        header = [bos_id] + ctrl_ids + [sep_id]
        footer = [eos_id]
        special_tokens_count = len(header) + len(footer)
        eff_len = self.seq_length - special_tokens_count

        seq = str(self.sequences[seq_name][start_pos : start_pos + eff_len]).upper()
        if self.rc_aug and np.random.rand() > 0.5:
            seq = self.reverse_complement(seq)

        token_ids = header + self.tokenizer.text_to_ids(seq) + footer
        # Pad/trim
        if len(token_ids) < self.seq_length:
            token_ids += [pad_id] * (self.seq_length - len(token_ids))
        else:
            token_ids = token_ids[: self.seq_length]

        tokens = torch.tensor(token_ids, dtype=torch.int64)

        # Flatten ctrl_ids and create special_ids list
        flat_ctrl_ids = []
        if isinstance(ctrl_ids, list):
            for item in ctrl_ids:
                if isinstance(item, list):
                    flat_ctrl_ids.extend(item)
                else:
                    flat_ctrl_ids.append(item)

        special_ids_list = [bos_id, eos_id, sep_id, pad_id] + flat_ctrl_ids
        special_ids = torch.tensor(special_ids_list, dtype=torch.int64)


        # Create labels for next token prediction
        labels = tokens.clone()
        labels[:-1] = tokens[1:]
        labels[-1] = pad_id
        
        # Create loss mask
        loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        loss_mask[torch.isin(labels, special_ids)] = 0

        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": self.position_ids,
        }
        if self.create_attention_mask:
            batch["attention_mask"] = self.attention_mask
        return batch

    def collate_fn(self, batch):
        return default_collate(batch)
