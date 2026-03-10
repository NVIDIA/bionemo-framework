import os
import importlib
import functools
from typing import Any, Dict

import fiddle as fdl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
import torch
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from src.data.datamodule import CodonFMDataModule
from src.data.metadata import TrainerModes
from src.tokenizer import Tokenizer
from src.utils.grad_norm_callback import GradientNormLogger
from src.models.encodon_pl import EncodonPL
from src.models.components.encoder_layer import EncoderLayer
from src.models.decodon_pl import DecodonPL
from src.utils.scheduler import linear_scheduler_with_warmup_lr_lambda
from src.utils.pred_writer import PredWriter
from src.inference.encodon import EncodonInference
from src.inference.decodon import DecodonInference
from src.executors.base import CONTAINER_OUT_DIR, CONTAINER_CHECKPOINTS_DIR


def _validate_decodon_vocab_size(args):
    """
    Validate that vocab_size parameter matches expected size for decodon models with organism tokens.
    """
    # Only validate if organism_tokens_file is provided
    organism_tokens_file = getattr(args, 'organism_tokens_file', None)
    if not organism_tokens_file:
        return  # No organism tokens, no validation needed
    
    from pathlib import Path
    
    # Check if the organism tokens file exists
    tokens_path = Path(organism_tokens_file)
    if not tokens_path.exists():
        print(f"Warning: organism_tokens_file '{organism_tokens_file}' not found, skipping vocab_size validation")
        raise ValueError(f"Organism tokens file '{organism_tokens_file}' not found")
    
    # Count lines in organism tokens file
    with open(tokens_path, 'r') as f:
        num_organism_tokens = sum(1 for line in f if line.strip())
    
    base_vocab_size = 69  # 5 special tokens + 64 codons
    expected_vocab_size = base_vocab_size + num_organism_tokens
    actual_vocab_size = args.vocab_size
    # Validate vocab_size matches
    assert actual_vocab_size == expected_vocab_size, (
        f"DeCodon vocab_size mismatch! "
        f"Expected: {expected_vocab_size}, Actual: {actual_vocab_size}. "
        f"Calculation: {base_vocab_size} (base) + {num_organism_tokens} (organism tokens) = {expected_vocab_size}. "
        f"Make sure your --vocab_size argument matches the base tokens (69) plus the number of lines in '{organism_tokens_file}'."
    )
    
    print(f"DeCodon vocab_size validation passed: {actual_vocab_size} tokens "
          f"({base_vocab_size} base + {num_organism_tokens} organism tokens)")

# Datasets
def get_dataset_config(args: Any, process_item_cfg: fdl.Partial) -> fdl.Config:
    """Builds the dataset configuration."""
    
    class_name = args.dataset_name
    if class_name == "CodonMemmapDataset":
        module_path = "src.data.codon_memmap_dataset"
    elif class_name == "MutationDataset":
        module_path = "src.data.mutation_dataset"
    elif class_name == "CodonBertDataset":
        module_path = "src.data.codon_bert_dataset"
    elif class_name == "GenerationPromptDataset":
        module_path = "src.data.generation_prompt_dataset"
    elif class_name.startswith("MissenseDataset"):
        module_path = "src.data.missense_dataset"
    else:
        raise ValueError(f"Unknown dataset name: {class_name}")

    try:
        module = importlib.import_module(module_path)
        dataset_class = getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ValueError(f"Could not import dataset '{args.dataset_name}'. Please check the name.") from e

    tokenizer_cfg = fdl.Config(Tokenizer, 
                               model_max_length=args.context_length, 
                               organism_tokens_file=args.organism_tokens_file)
    if args.mode == TrainerModes.PREDICT:
        args.train_val_test_ratio = None

    common_args = {
        "data_path": args.data_path,
        "tokenizer": tokenizer_cfg,
        "context_length": args.context_length,
        "train_val_test_ratio": args.train_val_test_ratio,
        "process_item": process_item_cfg,
    }
    if class_name == "CodonMemmapDataset":
        dataset_cfg = fdl.Partial(
            dataset_class,
            **common_args,
            codon_weights_file=getattr(args, "codon_weights_file", None),
            groups_to_use=args.groups_to_use,
            taxid_exclusion_file=getattr(args, "taxid_exclusion_file", None),
            split_name_prefix=getattr(args, "split_name_prefix", ""),
            pretraining_task="clm" if args.process_item == "clm_memmap" else "mlm",
        )
    elif class_name == "MutationDataset":
        dataset_cfg = fdl.Partial(
            dataset_class,
            **common_args,
            label_col=getattr(args, "label_col", None),
            extract_seq=getattr(args, "extract_seq", False),
            ref_seq_col=getattr(args, "ref_seq_col", "ref_seq"),
            task="mlm" if "encodon" in args.model_name else "clm",
            causal=getattr(args, "causal", False),
        )
    elif class_name == "CodonBertDataset":
        dataset_cfg = fdl.Partial(
            dataset_class,
            data_path=args.data_path,
            tokenizer=tokenizer_cfg,
            process_item=process_item_cfg,
            value_col=args.value_col,
            ref_seq_col=args.ref_seq_col,
        )
    elif class_name == "GenerationPromptDataset":
        # GenerationPromptDataset doesn't need process_item or data_path
        organism_id = getattr(args, 'organism_id', None)
        if organism_id is None:
            raise ValueError("--organism_id is required for GenerationPromptDataset")
        dataset_cfg = fdl.Partial(
            dataset_class,
            organism_id=organism_id,
            tokenizer=tokenizer_cfg,
            num_sequences_per_organism=getattr(args, 'num_sequences_per_organism', 1),
        )
    elif class_name == "MissenseDataset":
        dataset_cfg = fdl.Partial(
            dataset_class,
            **common_args,
            num_variants_per_seq=args.missense_variants_per_seq,
            use_weights=args.missense_use_weights,
            use_paiv1=args.missense_use_paiv1,
            use_am=args.missense_use_am,
            center_weight_threshold=args.missense_center_weight_threshold,
            n_per_benign=args.missense_n_per_benign,
        )
    else:
        print(f"Warning: Using generic config for dataset '{args.dataset_name}'.")
        dataset_cfg = fdl.Partial(
            dataset_class,
            **common_args
        )

    return dataset_cfg


# Callbacks
def get_callbacks_config(args: Any) -> Dict[str, fdl.Config]:
    """Builds the callbacks configuration."""
    global CONTAINER_CHECKPOINTS_DIR
    callbacks = {
        "model_checkpoint": fdl.Config(
            ModelCheckpoint,
            dirpath=os.path.join(args.local_out_dir, "checkpoints") if args.cluster == "local" else CONTAINER_CHECKPOINTS_DIR,
            save_last=True,
            every_n_train_steps=getattr(args, 'checkpoint_every_n_train_steps', 2000),
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        "early_stopping": fdl.Config(
            EarlyStopping,
            monitor="val/loss",
            patience=1000,
            mode="min",
        ),
        "model_summary": fdl.Config(ModelSummary, max_depth=-1),
        "lr_monitor": fdl.Config(LearningRateMonitor, logging_interval="step", log_weight_decay=True),
        "grad_norm_callback": fdl.Config(GradientNormLogger,
                                         log_every_n_steps=100),
    }
    if args.mode == TrainerModes.PREDICT:
        callbacks["pred_writer"] = fdl.Config(
            PredWriter,
            output_dir=args.predictions_output_dir,
            write_interval="batch",
            caching_interval=1,
            merge_on_epoch_end=True,
            delete_after_merge=True,
        )
    return callbacks

# Data
def get_data_config(args: Any) -> fdl.Config:
    """Builds the data configuration."""
    if args.process_item == 'mlm_memmap':
        from src.data.preprocess.mlm_memmap import process_item as process_item_fn
        process_item_cfg = fdl.Partial(
            process_item_fn,
            mlm_probability=args.mlm_probability,
            mask_replace_prob=args.mask_replace_prob,
            random_replace_prob=args.random_replace_prob,
        )
    elif args.process_item == 'mutation_pred_mlm':
        from src.data.preprocess.mutation_pred import mlm_process_item as process_item_fn
        process_item_cfg = fdl.Partial(
            process_item_fn,
            mask_mutation=args.mask_mutation
        )
    elif args.process_item == 'mutation_pred_clm':
        from src.data.preprocess.mutation_pred import clm_process_item as process_item_fn
        process_item_cfg = fdl.Partial(process_item_fn, organism_token=args.organism_token)
    elif args.process_item == 'mutation_pred_likelihood':
        from src.data.preprocess.mutation_pred import likelihood_process_item as process_item_fn
        process_item_cfg = fdl.Partial(process_item_fn)
    elif args.process_item == 'clm_memmap':
        from src.data.preprocess.clm_memmap import process_item as process_item_fn
        process_item_cfg = fdl.Partial(process_item_fn)
    elif args.process_item == 'codon_sequence':
        from src.data.preprocess.codon_sequence import process_item as process_item_fn
        process_item_cfg = fdl.Partial(
            process_item_fn,
            context_length=args.context_length,
        )
    elif args.process_item == 'generation_prompt':
        # GenerationPromptDataset handles everything internally, no process_item needed
        process_item_cfg = None
    elif args.process_item == 'missense_seq':
        from src.data.preprocess.missense_seq import process_item as process_item_fn
        process_item_cfg = fdl.Partial(process_item_fn,
                                       context_length=args.context_length,
                                       mask_ref=args.mask_mutation,
                                       mlm_probability=args.mlm_probability,
                                       mask_replace_prob=args.mask_replace_prob,
                                       random_replace_prob=args.random_replace_prob,)
    elif args.process_item == 'missense_inference':
        from src.data.preprocess.mutation_pred import missense_inference_process_item as process_item_fn
        process_item_cfg = fdl.Partial(process_item_fn,
                                       context_length=args.context_length)
    else:
        raise ValueError(f"Unknown process_item: {args.process_item}")

    dataset_cfg = get_dataset_config(args, process_item_cfg)

    return fdl.Config(
        CodonFMDataModule,
        dataset=dataset_cfg,
        train_iters=args.max_steps,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        process_item=process_item_cfg,
        pin_memory=False,
        persistent_workers=False,
        world_size=args.num_nodes * args.num_gpus,
        mode = args.mode,
    )

# Logger
def get_logger_config(args: Any) -> fdl.Config:
    """Builds the logger configuration."""
    return fdl.Config(
        WandbLogger,
        name=args.exp_name,
        project=args.project_name,
        entity=args.entity,
        save_dir=args.local_out_dir if args.cluster == "local" else CONTAINER_OUT_DIR,
    )

# Model
MODEL_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "encodon_80m": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
    },
    "encodon_600m": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 12,
    },
    "encodon_1b": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 18,
    },
    "encodon_5b": {
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
    },
    "encodon_10b": {
        "hidden_size": 5120,
        "intermediate_size": 20480,
        "num_attention_heads": 40,
        "num_hidden_layers": 34,
    },
    "decodon_200m": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 16,
        "num_hidden_layers": 16,
    },
    "decodon_1b": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 18,
    }

}

def get_model_config(args: Any) -> fdl.Config:
    """Returns the model configuration."""
    arch = MODEL_ARCHITECTURES.get(args.model_name)
    if arch is None:
        raise ValueError(f"Unknown model name: {args.model_name}")
    if args.mode == TrainerModes.PRETRAIN or args.mode == TrainerModes.FINETUNE:
        scheduler = fdl.Partial(
            torch.optim.lr_scheduler.LambdaLR,
            lr_lambda=fdl.Partial(
                linear_scheduler_with_warmup_lr_lambda,
                total_iterations=args.lr_total_iterations,
                warmup_iterations=args.warmup_iterations,
            ),
        )
        extra_kwargs = {}
        if "decodon" in args.model_name:
            cls = DecodonPL
            # Validate vocab_size for decodon models
            _validate_decodon_vocab_size(args)
            extra_kwargs.update({
                "vocab_size": args.vocab_size,
            })
        else:
            cls = EncodonPL
            extra_kwargs.update({
                "loss_type": args.loss_type,
                "num_classes": getattr(args, "num_classes", 2),
                "use_downstream_head": getattr(args, "use_downstream_head", False),
                "cross_attention_hidden_dim": getattr(args, "cross_attention_hidden_dim", 256),
                "cross_attention_num_heads": getattr(args, "cross_attention_num_heads", 8),
                "max_position_embeddings": getattr(args, "context_length", 2048),
                "finetune_strategy": args.finetune_strategy,
            })
        return fdl.Config(
            cls,
            optimizer=fdl.Partial(
                torch.optim.AdamW,
                lr=args.lr,
                weight_decay=args.weight_decay,
            ),
            scheduler=scheduler,
            lora=getattr(args, 'lora', False) or args.finetune_strategy == "lora",
            lora_alpha=getattr(args, 'lora_alpha', 32.0),
            lora_r=getattr(args, 'lora_r', 16),
            lora_dropout=getattr(args, 'lora_dropout', 0.1),
            **arch,
            **extra_kwargs,
        )
    elif args.mode == TrainerModes.PREDICT:
        
        if "decodon" in args.model_name:
            inference_cls = DecodonInference
        else:
            inference_cls = EncodonInference
        
        config_kwargs = {
            "model_path": args.checkpoint_path,
            "task_type": args.task_type,
        }
        
        if "decodon" in args.model_name:
            config_kwargs["organism_tokens_file"] = getattr(args, 'organism_tokens_file', None)
            # Add generation config for SEQUENCE_GENERATION task
            if args.task_type == "sequence_generation":
                config_kwargs["generation_config"] = {
                    "max_new_tokens": getattr(args, 'max_new_tokens', 2048),
                    "temperature": getattr(args, 'generation_temperature', 0.9),
                    "top_k": getattr(args, 'generation_top_k', 50),
                    "top_p": getattr(args, 'generation_top_p', 0.95),
                    "use_stop_codons": not getattr(args, 'no_stop_codons', False),
                }
        
        return fdl.Config(
            inference_cls,
            **config_kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

# Trainer
def get_trainer_config(args: Any) -> Dict[str, Any]:
    """Builds the trainer configuration arguments."""
    trainer_kwargs = dict(
        num_nodes=args.num_nodes,
        devices=args.num_gpus,
        max_steps=args.max_steps,
        default_root_dir=args.local_out_dir if args.cluster == "local" else CONTAINER_OUT_DIR,
        _strategy_type="fsdp" if args.enable_fsdp else "ddp" if args.mode != TrainerModes.FINETUNE else "ddp_find_unused_parameters_true",
        _fsdp_config={
            "transformer_layer_cls_names": ["EncoderLayer"],
            "state_dict_type": "sharded" if args.sharded_state_dict else "full",
        } if args.enable_fsdp else None,
        precision="bf16-mixed" if getattr(args, 'bf16', False) else "32-true",
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=False,
        max_epochs=-1,
        min_epochs=1,
        sync_batchnorm=False,
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )
    if args.check_val_every_n_epoch:
        trainer_kwargs['check_val_every_n_epoch'] = args.check_val_every_n_epoch
    else:
        trainer_kwargs['val_check_interval'] = args.val_check_interval
    
    return trainer_kwargs

def create_strategy_from_config(trainer_kwargs: Dict[str, Any]) -> Any:
    """Creates the strategy object from trainer configuration.
    
    This function is called at task execution time to instantiate the strategy
    from the serializable configuration parameters.
    """
    strategy_type = trainer_kwargs.pop("_strategy_type", "ddp")
    fsdp_config = trainer_kwargs.pop("_fsdp_config", None)
    
    if strategy_type == "fsdp" and fsdp_config:        
        # Convert class names to actual class objects
        transformer_layer_cls_names = fsdp_config["transformer_layer_cls_names"]
        transformer_layer_cls = set()
        for cls_name in transformer_layer_cls_names:
            if cls_name == "EncoderLayer":
                transformer_layer_cls.add(EncoderLayer)
        
        
        strategy = FSDPStrategy(
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_layer_cls,
            ),
            state_dict_type=fsdp_config["state_dict_type"],
            forward_prefetch=True,
            limit_all_gathers=True,
        )
    else:
        strategy = strategy_type
    
    trainer_kwargs["strategy"] = strategy
    return trainer_kwargs


def get_one_logger_callback_config(args: Any) -> Dict[str, Any]:
    """Builds the one logger callback configuration."""
    return {
        "one_logger_async": True,
        "one_logger_project": args.project_name,
        "one_logger_run_name": args.exp_name,
        "log_every_n_train_iterations": args.log_every_n_steps,
        "app_tag_run_version": "1.0.0",
        "summary_data_schema_version": "1.0.0",
        "app_run_type": "training",
        "app_tag": f"{args.exp_name}_{args.context_length}_{args.train_batch_size}",
        "app_tag_run_name": args.exp_name,
        "world_size": args.num_nodes * args.num_gpus,
        "global_batch_size": args.train_batch_size,
        "batch_size": args.train_batch_size * args.num_gpus * args.num_nodes,
        "micro_batch_size": args.train_batch_size,
        "seq_length": args.context_length,
        "train_iterations_target": args.max_steps,
        "train_samples_target": args.max_steps * args.train_batch_size * args.num_gpus * args.num_nodes,
        "is_train_iterations_enabled": True,
        "is_baseline_run": False,
        "is_test_iterations_enabled": False,
        "is_validation_iterations_enabled": True,
        "is_save_checkpoint_enabled": True,
        "is_log_throughput_enabled": False,
        "save_checkpoint_strategy": "sync",
    }

# Main config
def get_config(args: Any) -> fdl.Config:
    """Combines the model, data, and trainer configs into a single config."""
    cfg = fdl.Config(dict)
    cfg.model = get_model_config(args)
    cfg.data = get_data_config(args)
    cfg.trainer = get_trainer_config(args)
    cfg.callbacks = get_callbacks_config(args)
    cfg.log = get_logger_config(args)
    cfg.one_logger_callback_config = get_one_logger_callback_config(args)
    return cfg 