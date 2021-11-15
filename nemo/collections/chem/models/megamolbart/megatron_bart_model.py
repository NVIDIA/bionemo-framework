
import os
import re
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import tempfile
import time

import numpy as np
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.utils.data as pt_data
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from apex.transformer import parallel_state

from nemo.core import optim
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import ChannelType, LossType, MaskType, NeuralType
from nemo.utils import logging, model_utils
from nemo.utils.app_state import AppState
from nemo.collections.common import metrics
from nemo.collections.common.data import ConcatDataset
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.losses import CrossEntropyLoss

from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    initialize_model_parallel_for_nemo,
    set_jit_fusion_options,
)

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)

from nemo.collections.chem.data import MoleculeDataset, MoleculeIterableDataset, ConcatIterableDataset, MoleculeEnumeration, expand_dataset_paths
from nemo.collections.chem.tokenizer import MolEncTokenizer, MolEncTokenizerFromVocabFileConfig
from nemo.collections.chem.decoder import DecodeSampler
from nemo.collections.chem.optimizer import TransformerLR, TransformerLRParams
from .megatron_bart_base import MegatronBART

__all__ = ["MegaMolBARTModel"]


class MegaMolBARTModel(NLPModel):   
    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer) # TODO BUG number of GPUS not correctly initialized?
        self.cfg = cfg

        # These handle irregular configuration settings upon restore from old checkpoints
        cfg_model = cfg.model if cfg.get('model', False) else cfg
        cfg_tokenizer = cfg.tokenizer if cfg.get('tokenizer', False) else OmegaConf.create(MolEncTokenizerFromVocabFileConfig()) # TODO: change when other tokenizers added

        if self.cfg.get('use_cpu_initialization', False) is False:
            torch.cuda.set_device(trainer.local_rank)

        logging.info('Model parallel setup beginning')
        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg_model.get('tensor_model_parallel_size', 1),
            seed=self.cfg.get('seed', 1234),
        )

        if not self.cfg.get('fused_bf16', False):
            set_jit_fusion_options()

        # Load tokenizer and model
        self.tokenizer = self.setup_tokenizer(cfg_tokenizer)
        self.sampler = self.setup_sampler(self.tokenizer, cfg_model)

        pad_token_idx = self.tokenizer.vocab[self.tokenizer.pad_token]        
        self._vocab_size = len(self.tokenizer)
        self._model_name = cfg_model.name
        self.max_seq_len = cfg_model.max_seq_len
        self.val_sampling_alg = 'greedy'
        self.d_model = cfg_model.d_model # for scheduler
        
        self.model = MegatronBART( 
                                self.sampler,
                                pad_token_idx,
                                self._vocab_size,
                                cfg_model.d_model,
                                cfg_model.num_layers,
                                cfg_model.num_heads,
                                cfg_model.d_feedforward,
                                cfg_model.max_seq_len,
                                cfg_model.dropout)

        self.num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        optim.lr_scheduler.register_scheduler('TransformerLR', TransformerLR, TransformerLRParams) # TODO scale LR for global_batch_size
        self.setup_optimization(cfg_model.optim)

        self.val_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)
        self.test_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)

        # TODO collate functions are not currently threadsafe and have to be declared in itit
        self.train_collate = MoleculeEnumeration(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, **cfg_model.train_ds)
        self.val_collate = MoleculeEnumeration(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, **cfg_model.validation_ds)
        self.test_collate = MoleculeEnumeration(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, **cfg_model.validation_ds) # TODO test_ds is not used

    @staticmethod
    def setup_tokenizer(cfg: DictConfig) -> MolEncTokenizer:
        if not os.path.exists(cfg.vocab_path):
            raise ValueError(f'Vocab file not found at {cfg.vocab_path}')
        tokenizer = MolEncTokenizer.from_vocab_file(**cfg)
        return tokenizer

    @staticmethod
    def setup_sampler(tokenizer: MolEncTokenizer, cfg: DictConfig) -> DecodeSampler:
        return DecodeSampler(tokenizer, cfg.max_seq_len)

    def compute_consumed_samples(self, global_step, micro_batch_size):
        app_state = AppState()
        consumed_samples = (
            global_step
            * app_state.data_parallel_size
            * micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)

    def setup(self, stage=None):
        if stage == 'predict':
            return
            
        # TODO: consider adding a ModelPT guard to check if model is being restored allowing restored models to optionally setup datasets
        logging.info('Setting up datasets and dataloaders')
        self.setup_training_data(self.cfg.model.train_ds)

        if self.cfg.model.validation_ds.get('filepath', False):
            self.setup_validation_data(self.cfg.model.validation_ds)

        if self.cfg.do_testing:
            self.setup_test_data(self.cfg.model.validation_ds)  # TODO BUG test_ds is not used

    def setup_training_data(self, cfg: DictConfig) -> None:
        logging.info('Loading training data')
        cfg = cfg.copy()
        self._train_ds = self._setup_dataset_from_config(cfg)

        resume_checkpoint_path = self.trainer.checkpoint_connector.resume_checkpoint_path
        if resume_checkpoint_path:
            consumed_samples = int(
                float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", resume_checkpoint_path)[0])
            )
        else:
            consumed_samples = 0
        logging.info(
            f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
        )

        consumed_samples = 0
        collate_fn = self.train_collate.collate_fn # TODO a multiprocessing error is thrown if collate fn not defined in init
        self._train_dl = self._setup_dataloader_from_config(self._train_ds, cfg, collate_fn, consumed_samples)

    def setup_validation_data(self, cfg: DictConfig):
        logging.info('Loading validation data')
        cfg = cfg.copy()
        self._validation_ds = self._setup_dataset_from_config(cfg)

        collate_fn = self.val_collate.collate_fn # TODO a multiprocessing error is thrown if collate fn not defined in init
        self._validation_dl = self._setup_dataloader_from_config(self._validation_ds, cfg, collate_fn)

    def setup_test_data(self, cfg: DictConfig):
        logging.info('Loading test data')
        cfg = cfg.copy()
        self._test_ds = self._setup_dataset_from_config(cfg)

        collate_fn = self.test_collate.collate_fn # TODO a multiprocessing error is thrown if collate fn not defined in init
        self._test_dl = self._setup_dataloader_from_config(self._test_ds, cfg, collate_fn)

    def _setup_dataset_from_config(self, cfg: DictConfig):
        # Setup config
        cfg = cfg.copy()

        OmegaConf.set_struct(cfg, False)
        filepath = cfg.pop('filepath', None)
        use_iterable = cfg.pop('use_iterable', False)
        cfg['tensor_model_parallel_size'] = self.cfg.model.tensor_model_parallel_size
        OmegaConf.set_struct(cfg, True)

        # Get datasets and load data
        dataset_paths = expand_dataset_paths(filepath)
        logging.info(f'Loading data from {dataset_paths}')
        dataset_list = []
        for path in dataset_paths:
            if use_iterable:
                data = MoleculeIterableDataset(filepath=path, cfg=cfg, trainer=self.trainer)
            else:
                data = MoleculeDataset(filepath=path, cfg=cfg, trainer=self.trainer)
            dataset_list.append(data)

        if len(dataset_list) == 1:
            dataset = dataset_list[0]
        else:
            dataset = ConcatIterableDataset(dataset_list) if use_iterable else pt_data.ConcatDataset(dataset_list)
        return dataset

    def _setup_dataloader_from_config(self, dataset, cfg: DictConfig, collate_fn: Callable, consumed_samples: int = 0):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        sampler_kwargs = dict(total_samples=len(dataset),
                              consumed_samples=consumed_samples,
                              micro_batch_size=cfg.micro_batch_size,
                              data_parallel_rank=parallel_state.get_data_parallel_rank(),
                              data_parallel_size=parallel_state.get_data_parallel_world_size())
        shuffle = cfg.get('shuffle', False)
        if shuffle:
            batch_sampler = MegatronPretrainingRandomSampler(**sampler_kwargs)
        else:
            batch_sampler = MegatronPretrainingSampler(**sampler_kwargs)

        dataloader = pt_data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, 
            num_workers=cfg.num_workers, pin_memory=cfg.get("pin_memory", True) 
        )
        return dataloader

    @typecheck()
    def forward(self, batch):
        app_state = AppState()
        if app_state.model_parallel_size is None:
            self.complete_lazy_init()

        outputs = self.model(batch)
        return outputs

    def training_step(self, batch: dict, batch_idx: int) -> Dict:
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`. 
        """
        start_time = time.monotonic()

        outputs = self.forward(batch)
        loss = self.model._calc_loss(batch, outputs)
        char_acc = self.model._calc_char_acc(batch, outputs)
        lr = self._optimizer.param_groups[0]["lr"]

        end_time = time.monotonic()
        duration = end_time - start_time
        global_step = self.trainer.global_step
        micro_batch_size = self.cfg.model.train_ds.micro_batch_size
        consumed_samples = self.compute_consumed_samples(global_step, micro_batch_size)

        self.log('lr', lr)
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_char_acc', char_acc, on_epoch=True, sync_dist=True)
        self.log('step_time', duration, on_step=True)
        self.log('global_step', global_step, prog_bar=True)
        self.log('consumed_samples', consumed_samples, prog_bar=True)

        tensorboard_logs = {'train/loss': loss.item(),
                            'train/char_acc': char_acc, 
                            'trainer/lr': lr,
                            'trainer/step_time': duration,
                            'trainer/global_step': global_step,
                            'trainer/consumed_samples': consumed_samples}

        return {'loss': loss, 
                'log': tensorboard_logs}

    def _eval_step(self, batch: dict, batch_idx: int, mode: str) -> Dict:
        self.model.eval()
        
        model_output = self.model.forward(batch)
        target_smiles = batch['target_smiles']

        loss = self.model._calc_loss(batch, model_output).item()
        perplexity = self.model._calc_perplexity(batch, model_output)
        token_acc = self.model._calc_char_acc(batch, model_output)
        (mol_strs, log_lhs) = self.model.sample_molecules(batch, sampling_alg=self.val_sampling_alg) 
        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        # Molecular_accuracy is entered twice so a version w/o slash exists bc it interferes with checkpoint name
        logs = {
            f'{mode}/step': self.global_step,
            f'{mode}/loss': loss,
            f'{mode}/perplexity': perplexity,
            f'{mode}/char_acc': token_acc,
            f'{mode}/molecular_accuracy': metrics['accuracy'],
            f'{mode}/invalid_smiles': metrics['invalid'],
            f'{mode}_molecular_accuracy': metrics['accuracy']}

        self.log_dict(logs, on_epoch=True, sync_dist=True)
        logs['log'] = logs.copy()
        return logs

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        return self._eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'test')

    def training_epoch_end(self, outputs):
        logging.info(f'Finishing training epoch {self.current_epoch}')
        return super().training_epoch_end(outputs)

    def _eval_epoch_end(self, outputs: List[Dict], mode: str, micro_batch_size: int) -> Dict:
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        logging.info(f'Starting final evaluation for {mode} step.')

        loss_label = f'{mode}/loss'
        eval_loss = torch.tensor([x[loss_label] for x in outputs]).mean().item()

        ppl_label = f'{mode}/perplexity'
        eval_ppl = torch.tensor([x[ppl_label] for x in outputs]).mean().item()

        token_label = f'{mode}/char_acc'
        eval_token_acc = torch.tensor([x[token_label] for x in outputs]).mean().item()

        mol_acc_label = f'{mode}/molecular_accuracy'
        eval_mol_acc = torch.tensor([x[mol_acc_label] for x in outputs]).mean().item()

        consumed_samples_label = f'{mode}/consumed_samples'
        consumed_samples = self.compute_consumed_samples(self.trainer.global_step, micro_batch_size)

        logs =  {f'{mode}/step_avg': self.global_step,
                 f'{loss_label}_avg': eval_loss, 
                 f'{ppl_label}_avg': eval_ppl,
                 f'{token_label}_avg': eval_token_acc,
                 f'{mol_acc_label}_avg': eval_mol_acc,
                 f'{consumed_samples_label}': consumed_samples}

        logging.info(f'Metrics from {mode} epoch end at step {self.global_step}: loss:{eval_loss}, perplexity:{eval_ppl}, token acc:{eval_token_acc}, molecular acc:{eval_mol_acc}')

        self.log_dict(logs, sync_dist=True)
        logs['log'] = logs.copy()

        logging.info(f'Finished final evaluation for {mode} step.')
        return logs

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        mode = 'val'
        micro_batch_size = self.cfg.model.validation_ds.micro_batch_size
        logs = self._eval_epoch_end(outputs, mode, micro_batch_size)
        return logs

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        mode = 'test'
        micro_batch_size = self.cfg.model.test_ds.micro_batch_size
        logs = self._eval_epoch_end(outputs, mode, micro_batch_size)
        return logs

    @rank_zero_only
    def log_param_stats(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.trainer.logger.experiment.add_histogram(name + '_hist', p, global_step=self.global_step)
                self.trainer.logger.experiment.add_scalars(
                    name,
                    {'mean': p.mean(), 'stddev': p.std(), 'max': p.max(), 'min': p.min()},
                    global_step=self.global_step,
                )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass


    # def _set_app_state(self, cfg, node_rank):
    #     app_state = AppState()
    #     if cfg.get('trainer'):
    #         app_state._world_size = cfg.trainer.num_nodes * cfg.trainer.gpus
    #         num_gpus = cfg.trainer.gpus
    #     else:
    #         app_state._world_size = 1
    #         num_gpus = 1

    #     app_state.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    #     # app_state.node_rank = node_rank # 'NODE_RANK' NOT USED BY SELENE
    #     app_state.global_rank = node_rank * num_gpus + app_state.local_rank
    #     app_state.model_parallel_size = None
    #     app_state.model_parallel_rank = None
    #     # app_state.device_id = None # TODO add these
    #     # app_state.model_parallel_group = None
    #     # app_state.data_parallel_size = None
    #     # app_state.data_parallel_rank = None
    #     # app_state.data_parallel_group = None
    #     self._app_state = app_state



    # @staticmethod
    # def _update_megatron_args(
    #     micro_batch_size: int = 1,
    #     tensor_model_parallel_size: int = 1,
    #     scaled_masked_softmax_fusion: bool = False,
    #     bias_gelu_fusion: bool = False,
    #     bias_dropout_fusion: bool = False):

    #     def extra_args_provider(parser):
    #         parser.set_defaults(micro_batch_size=micro_batch_size)
    #         parser.set_defaults(tensor_model_parallel_size=tensor_model_parallel_size)
    #         parser.set_defaults(scaled_masked_softmax_fusion=scaled_masked_softmax_fusion)
    #         parser.set_defaults(bias_gelu_fusion=bias_gelu_fusion)
    #         parser.set_defaults(bias_dropout_fusion=bias_dropout_fusion)
    #         return parser

    #     return extra_args_provider

    # @staticmethod
    # def _get_megatron_vocab_file() -> str:
    #     """Generate fake Megatron vocab file with required tokens"""
    #     fake_vocab_contents = '\n'.join(['[CLS]', '[SEP]', '[PAD]', '[MASK]'])
    #     with tempfile.NamedTemporaryFile(mode='w', delete=False) as fh:
    #         fh.write(fake_vocab_contents)
    #         vocab_file = fh.name
    #     return vocab_file

    # def complete_lazy_init(self) -> None:
    #     # Finish megatron-lm initialization
    #     if hasattr(self, "_lazy_init_fn") and self._lazy_init_fn is not None:
    #         logging.info('Completing lazy initialization of Megatron framework...')
    #         self._lazy_init_fn()
    #         self._lazy_init_fn = None

    #         LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
    #         RANK = int(os.environ.get('RANK', -1))
    #         app_state = AppState()
    #         logging.info(f'Env GPU rank setup: local rank {LOCAL_RANK}, global rank {RANK}')
    #         logging.info(f'App GPU rank setup: local rank {app_state.local_rank}, global rank {app_state.global_rank}')

    # def setup_megatron(self, cfg: DictConfig) -> dict:
    #     """Initialize Megatron"""
    #     app_state = AppState()
    #     model_parallel_size = app_state.model_parallel_size
    #     model_parallel_rank = app_state.model_parallel_rank

    #     # Configure globals
    #     set_pipeline_model_parallel_rank(0)  # Pipeline model parallelism not currently implemented in NeMo
    #     set_pipeline_model_parallel_world_size(1)  # Pipeline model parallelism not currently implemented in NeMo

    #     # megatron input arguments
    #     args = {'num_layers': cfg.num_layers,
    #             'hidden_size': cfg.d_model,
    #             'num_attention_heads': cfg.num_heads,
    #             'max_position_embeddings': cfg.max_seq_len,
    #             'onnx_safe': True,
    #             'lazy_mpu_init': True,
    #             'tokenizer_type': 'BertWordPieceCase',
    #             'vocab_file': self._get_megatron_vocab_file()}
    #             # TODO vocab size may need to be set

    #     # extra args provider
    #     if model_parallel_size is not None:
    #         app_state = AppState()
    #         self._app_state = app_state
    #         os.environ["WORLD_SIZE"] = str(app_state.world_size) # Must be set for model parallel megatron-lm
    #         os.environ["RANK"] = str(model_parallel_rank)
    #         extra_args_provider = self._update_megatron_args(tensor_model_parallel_size=model_parallel_size)
    #     else:
    #         extra_args_provider = self._update_megatron_args()

    #     # Initialize part of Megatron global state that is needed for its constructor.
    #     # We set 'lazy_mpu_init' flag on to make Megatron do only the initialization that does not depend
    #     # on ddp be initialized yet (and we don't want Megatron to initialize DDP itself either)
    #     # and to return a hook for us to call after PTL has torch.distributed initialized.
    #     # (or if no PTL in case of inference - then we'll initialize torch.distributed)
    #     # We call and clear this hook on first call to forward()
    #     self._lazy_init_fn = initialize_megatron(
    #         extra_args_provider=extra_args_provider, args_defaults=args, ignore_unknown_args=True
    #     )

    #     # Read Megatron arguments back
    #     args = get_args()
    #     logging.info(f'Megatron-lm argparse args: {args}')

    #     # This loads a fake model from megatron, mostly for the sake of ensuring compatible checkpoint dict
    #     _, self._language_model_key = get_language_model(
    #         attention_mask_func=bert_attention_mask_func, num_tokentypes=2, add_pooler=False
    #     )
    #     return args