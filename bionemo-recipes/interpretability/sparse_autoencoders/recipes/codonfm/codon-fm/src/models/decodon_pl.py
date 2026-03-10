from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.distributed.fsdp.wrap import wrap

import pytorch_lightning as pl
from lightning import LightningModule

from torchmetrics import MeanMetric
from transformers.utils import logging
from peft import LoraConfig, get_peft_model, PeftType

logger = logging.get_logger(__name__)

from src.data.metadata import MetadataFields
from src.models.components.decodon_config import DeCodonConfig
from src.models.components.decodon import DeCodon
from src.models.utils import construct_pretrained_config, get_decay_parameter_names


class DecodonPL(LightningModule):
    """The DeCodon Module (Decoder only) without any task-specific head on top."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        vocab_size: int = 69,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1.0e-12,
        pad_token_id: int = 3, # - 3 is the padding token id in the tokenizer
        classifier_dropout: float = 0.1,
        gamma_init: float = 0.1,
        rotary_theta: float = 1e4,
        ignore_index: int = -100,
        lora: bool = False,
        lora_alpha: float = 32.0,
        lora_r: int = 16,
        lora_dropout: float = 0.1,
        finetune_strategy: str = "full",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.pretrained_config = construct_pretrained_config(
            self.hparams, DeCodonConfig
        )
        self.model = None
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def configure_model(self, state_dict: Optional[Dict[str, Any]] = None) -> None:
        """Configure the model, optionally loading a state dict.
        
        This method sets up the EnCodon model, handles PEFT (LoRA) configuration,
        and loads pretrained weights from a state dictionary if provided.

        Args:
            state_dict: A state dictionary to load into the model.
        """
        if self.model is not None:
            # If model already exists and no state_dict provided, nothing to do
            if state_dict is None:
                return
            # If model exists but state_dict provided, we need to check for PEFT params
            # and handle loading properly, so continue with the rest of the method
            # rather than doing early return

        if self.model is None:
            self.model = wrap(DeCodon(self.pretrained_config))

        has_peft_params = any("lora" in k for k in state_dict.keys()) if state_dict else False

        # Load base model weights if not loading a PEFT checkpoint
        if state_dict and not has_peft_params:
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[len("model.") :]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=True)

        if self.hparams.finetune_strategy in ["head_only_pretrained", "head_only_random"]:
            logger.info("Freezing backbone for head-only finetuning.")
            # The backbone consists of embeddings and encoder layers
            backbone = [self.model.embeddings, self.model.layers]
            for module in backbone:
                for param in module.parameters():
                    param.requires_grad = False
                    
        if self.hparams.finetune_strategy == "head_only_random":
            logger.info("Randomly initializing lm_head.")
            self.model.reset_cls_parameters()
            
        # Configure PEFT if LoRA is enabled or if loading a LoRA checkpoint
        if self.hparams.lora or has_peft_params:
            peft_config = LoraConfig(
                peft_type=PeftType.LORA,
                task_type="CAUSAL_LM",
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=["query", "value", "intermediate_dense"],
                inference_mode=False,
            )
            self.model = get_peft_model(self.model, peft_config)
            
            #  Ensure classifier head `model.cls` is trainable after applying LoRA
            cls_trainable = any(param.requires_grad for param in self.model.cls.parameters())
            if not cls_trainable:
                for param in self.model.cls.parameters():
                    param.requires_grad = True
            # Log LoRA parameter counts
            if self.hparams.lora and not has_peft_params:
                self.model.print_trainable_parameters()

        # Load the full state dict (including PEFT weights) if provided
        if state_dict and has_peft_params:
            self.load_state_dict(state_dict, strict=True)
        
    def forward(self, batch_data, return_hidden_states: bool = False):
        input_ids = batch_data[MetadataFields.INPUT_IDS]
        attention_mask = batch_data[MetadataFields.ATTENTION_MASK]
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden_states=return_hidden_states,
        )
        return output

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        output = self.forward(batch)
        preds = output.logits
        y = batch[MetadataFields.LABELS]
        y = y.to(preds.device)
        
        # Predictions are only over codons and special tokens (64 + 5 = 69)
        # Organism tokens are input context only, not prediction targets
        codon_vocab_size = 64 + 5  # codons + special tokens
        
        y = y.view(-1)
        preds = preds.view(-1, codon_vocab_size)
        loss = self.loss(preds, y)
        
        if loss.dtype in [torch.float16, torch.bfloat16]:
            loss = loss.to(torch.float32)

        return loss, preds, y

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, prog_bar=False, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.log(
            "val/loss", self.val_loss, prog_bar=False, on_step=False, on_epoch=True
        )

    def on_validation_end(self) -> None:
        self.val_loss.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.hparams.finetune_strategy in ["head_only_pretrained", "head_only_random"]:
            param_source = self.model.cls
        else:
            param_source = self.trainer.model
        
        decay_parameters = get_decay_parameter_names(
            param_source,
            none_applicable_layer_types=[nn.LayerNorm, nn.Embedding],
            disallowed_layer_names=["bias", "layernorm", "rmsnorm", "rotary_emb"],
        )
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_source.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
            },
            {
                "params": [
                    p
                    for n, p in param_source.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = self.hparams.optimizer(params=optimizer_grouped_parameters)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """
        Configures gradient clipping for both FSDP and non-FSDP modes.

        Args:
            optimizer: Optimizer instance.
            gradient_clip_val: Maximum allowed value for gradients.
            gradient_clip_algorithm: Clipping algorithm ('norm' or 'value').

        Raises:
            ValueError: If gradient_clip_val is negative or gradient_clip_algorithm is invalid.
        """
        if gradient_clip_val is not None and gradient_clip_val < 0:
            raise ValueError("gradient_clip_val must be non-negative")
            
        if gradient_clip_algorithm not in ('norm', 'value', None):
            raise ValueError("gradient_clip_algorithm must be one of: 'norm', 'value', None")

        if self.trainer.strategy.__class__.__name__ == "FSDPStrategy":
            if gradient_clip_algorithm not in ('norm', None):
                raise ValueError("FSDP only supports 'norm' gradient clipping")
            # For FSDP, use torch.nn.utils.clip_grad_norm_ directly
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
        else:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=gradient_clip_val, 
                gradient_clip_algorithm=gradient_clip_algorithm
            )

    def optimizer_step(self, *args, **kwargs):
        """
        Skipping updates in case of unstable gradients.
        Checks for both NaN and infinite values in gradients before calling the optimizer.
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            logger.warning(
                "Detected inf or nan values in gradients. Skipping model update."
            )
            self.zero_grad()
        
        super().optimizer_step(*args, **kwargs)
