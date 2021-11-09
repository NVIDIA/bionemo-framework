
import os
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
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

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

# from megatron import get_args, initialize_megatron
# from megatron.model.bert_model import bert_attention_mask_func
# from megatron.checkpointing import set_checkpoint_version
# from megatron.model import get_language_model
# from megatron.mpu import (
#     get_data_parallel_world_size,
#     get_data_parallel_rank,
#     get_model_parallel_group,
#     model_parallel_is_initialized,
#     set_pipeline_model_parallel_rank,
#     set_pipeline_model_parallel_world_size,
# )

from nemo.collections.chem.data import MoleculeDataset, MoleculeIterableDataset, ConcatIterableDataset, MoleculeEnumeration, expand_dataset_paths
from nemo.collections.chem.tokenizer import MolEncTokenizer, MolEncTokenizerFromVocabFileConfig
from nemo.collections.chem.decoder import DecodeSampler
from nemo.collections.chem.optimizer import TransformerLR, TransformerLRParams
from .megatron_bart_base import MegatronBART

__all__ = ["MegaMolBARTModel"]


class MegaMolBARTModel(NLPModel):   
    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None) -> None:

        # Handle irregular configuration settings
        cfg_model = cfg.model if cfg.get('model') else cfg

        if cfg.get('tokenizer'):
            cfg_tokenizer = cfg.tokenizer
        else:
            # TODO: must be changed when other tokenizers added
            cfg_tokenizer = OmegaConf.create(MolEncTokenizerFromVocabFileConfig())

        self.tokenizer = self.setup_tokenizer(cfg_tokenizer)
        self._vocab_size = len(self.tokenizer)
        self._model_name = cfg_model.name
        self.max_seq_len = cfg_model.max_seq_len
        self.val_sampling_alg = 'greedy'

        # TODO THIS WAS PROBLEMATIC IN THE FUNCTION AND HAS TO BE PLACED BEFORE INITIALIZATION
        self.train_collate = MoleculeEnumeration(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, **cfg_model.train_ds)
        self.val_collate = MoleculeEnumeration(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, **cfg_model.validation_ds)
        self.test_collate = MoleculeEnumeration(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, **cfg_model.validation_ds) # TODO test_ds is not used

        super().__init__(cfg=cfg_model, trainer=trainer)
        self.cfg = cfg

        if self.cfg.get('use_cpu_initialization', False) is False:
            torch.cuda.set_device(trainer.local_rank)

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            seed=self.cfg.get('seed', 1234),
        )

        if not self.cfg.get('fused_bf16'):
            set_jit_fusion_options()

        if trainer:
            self._set_ddp(trainer)

        # cfg = model_utils.convert_model_config_to_dict_config(cfg)
        # node_rank = trainer.node_rank if trainer else 0 # TODO fix this for model parallel checkpoint loading
        # self._set_app_state(cfg, node_rank)
        # cfg = model_utils.maybe_update_config_version(cfg)

        # # Handle irregular configuration settings
        # # cfg_model = cfg.model if cfg.get('model') else cfg
        # if cfg.get('tokenizer'):
        #     cfg_tokenizer = cfg.tokenizer
        # else:
        #     # TODO: must be changed when other tokenizers added
        #     cfg_tokenizer = OmegaConf.create(MolEncTokenizerFromVocabFileConfig())

        # self._model_name = cfg_model.name
        # self.max_seq_len = cfg_model.max_seq_len
        # self.tokenizer = self.setup_tokenizer(cfg_tokenizer)
        # self._vocab_size = len(self.tokenizer)
        # self.val_sampling_alg = 'greedy'

        # if trainer:
        #     self._set_ddp(trainer)

        # # Augmentation / collate functionality
        # self.train_collate = MoleculeEnumeration(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, **cfg_model.train_ds)
        # self.val_collate = MoleculeEnumeration(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, **cfg_model.validation_ds)
        # self.test_collate = MoleculeEnumeration(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, **cfg_model.validation_ds) # TODO test_ds is not used

        # _ = self.setup_megatron(cfg_model) # Megatron initialization -- must be done before superclass init and model loaded            
        # super().__init__(cfg=cfg_model, trainer=trainer)

        # Load model
        self.sampler = self.setup_sampler(self.tokenizer, cfg_model)
        pad_token_idx = self.tokenizer.vocab[self.tokenizer.pad_token]
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
        optim.lr_scheduler.register_scheduler('TransformerLR', TransformerLR, TransformerLRParams)
        self.setup_optimization(cfg_model.optim)

        self.val_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)
        self.test_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True) 

    def _set_ddp(self, trainer):
        # Sampler is replaced manually below because PTL seems to use global_rank instead of local_rank
        if trainer.accelerator_connector.replace_sampler_ddp & (trainer.accelerator_connector.distributed_backend == 'ddp'):
            self.replace_sampler_ddp = True
            trainer.accelerator_connector.replace_sampler_ddp = False
        else:
            self.replace_sampler_ddp = False

    @staticmethod
    def setup_tokenizer(cfg: DictConfig) -> MolEncTokenizer:
        if not os.path.exists(cfg.vocab_path):
            raise ValueError(f'Vocab file not found at {cfg.vocab_path}')
        tokenizer = MolEncTokenizer.from_vocab_file(**cfg)
        return tokenizer

    @staticmethod
    def setup_sampler(tokenizer: MolEncTokenizer, cfg: DictConfig) -> DecodeSampler:
        return DecodeSampler(tokenizer, cfg.max_seq_len)

    def setup(self, stage=None):
        if stage == 'predict':
            return
        # TODO: consider adding a ModelPT guard to check if model is being restored.
        # TODO BUG: handle train_valid_test_num_samples for model parallel and also consumed_samples on resume, can train_collate be setup here?
        # allowing restored models to optionally setup datasets
        self.setup_training_data(self.cfg.model.train_ds)
        self.setup_validation_data(self.cfg.model.validation_ds)
        self.setup_test_data(self.cfg.model.validation_ds)  # TODO test_ds is not used

    def setup_training_data(self, cfg: DictConfig) -> None:
        logging.info('Loading training data')
        collate_fn = self.train_collate.collate_fn
        self._train_dl = self.setup_dataloader_from_config(cfg, self.trainer, 'train', collate_fn)

    def setup_validation_data(self, cfg: DictConfig):
        
        if cfg.model.validation_ds.get('filepath'): # TODO skip if limit_val_batches=0.0
            logging.info('Loading validation data')
            collate_fn = self.val_collate.collate_fn
            self._validation_dl = self.setup_dataloader_from_config(cfg, self.trainer, 'validation', collate_fn)
        else:
            logging.info('Skipping validation data loading')

    def setup_test_data(self, cfg: DictConfig):
        logging.info('Loading test data')
        collate_fn = self.test_collate.collate_fn
        self._test_dl = self.setup_dataloader_from_config(cfg, self.trainer, 'test', collate_fn)

    def _setup_dataset_from_config(self, cfg: DictConfig):
        cfg = dict(cfg.copy())
        filepath = cfg.pop('filepath', None)
        use_iterable = cfg.pop('use_iterable', False)

        dataset_paths = expand_dataset_paths(filepath)
        logging.info(f'Loading data from {dataset_paths}')
        datasets = []
        for path in dataset_paths:
            if use_iterable:
                data = MoleculeIterableDataset(filepath=path, **cfg)
            else:
                data = MoleculeDataset(filepath=path, **cfg)
            datasets.append(data)

        if len(datasets) == 1:
            datasets = datasets[0]
        else:
            if use_iterable:
                datasets = ConcatIterableDataset(datasets)
            else:
                datasets = pt_data.ConcatDataset(datasets)
        return datasets

    def setup_dataloader_from_config(self, cfg: DictConfig, trainer: pl.Trainer, name: str, collate_fun: Callable):
        dataset = self._setup_dataset_from_config(cfg, trainer, name)

        # if self.replace_sampler_ddp:
        #     app_state = AppState()
        #     sampler = pt_data.DistributedSampler(dataset=dataset, num_replicas=app_state.world_size, 
        #                                          rank=app_state.local_rank,
        #                                          shuffle=cfg.shuffle, drop_last=cfg.drop_last)
        # else:
        #     sampler_name = pt_data.RandomSampler if cfg.shuffle else pt_data.SequentialSampler
        #     sampler = sampler_name(dataset)
        sampler_name = pt_data.RandomSampler if cfg.shuffle else pt_data.SequentialSampler # TODO BUG this is a hack to work around init_process_group
        sampler = sampler_name(dataset)
        
        dataloader = pt_data.DataLoader(dataset,
            sampler=sampler,
            batch_size=cfg.batch_size,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False), 
            drop_last=cfg.get("drop_last", False),
            collate_fn=collate_fun)
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

        self.log('lr', lr)
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_char_acc', char_acc, on_epoch=True, sync_dist=True)
        self.log('step_time', duration, on_step=True)

        tensorboard_logs = {'train/loss': loss.item(),
                            'train/char_acc': char_acc, 
                            'trainer/lr': lr,
                            'trainer/step_time': duration}

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

        # molecular_accuracy is entered twice so there is a version w/o slash which interferes with checkpointing
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

    def _eval_epoch_end(self, outputs: List[Dict], mode: str) -> Dict:
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

        logs =  {f'{mode}/step_avg': self.global_step,
                 f'{loss_label}_avg': eval_loss, 
                 f'{ppl_label}_avg': eval_ppl,
                 f'{token_label}_avg': eval_token_acc,
                 f'{mol_acc_label}_avg': eval_mol_acc}

        logging.info(f'Metrics from {mode} epoch end at step {self.global_step}: loss:{eval_loss}, perplexity:{eval_ppl}, token acc:{eval_token_acc}, molecular acc:{eval_mol_acc}')

        # self.log_dict(logs) # TODO BUG should this be removed?
        logs['log'] = logs.copy()

        logging.info(f'Finished final evaluation for {mode} step.')
        return logs

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        mode = 'val'
        self.log_dict(self._eval_epoch_end(outputs, mode), sync_dist=True)

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        mode = 'test'
        self.log_dict(self._eval_epoch_end(outputs, mode), sync_dist=True)

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