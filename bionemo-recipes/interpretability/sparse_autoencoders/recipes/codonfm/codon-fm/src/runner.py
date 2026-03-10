import argparse
import logging
import os
import nemo_run as run
from dotenv import load_dotenv
load_dotenv()

from src.executors import get_executor_fn
from src.utils.logger import WandbPlugin, wandb_logger
from src.tasks import train, finetune, evaluate
from src.config import get_config
from src.utils.nemorun_utils import config_to_dict
from src.executors.base import CONTAINER_OUT_DIR, CONTAINER_PRETRAINED_CKPT_PATH, CONTAINER_DATA_DIR, CONTAINER_CHECKPOINTS_DIR
from src.data.metadata import TrainerModes

log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Codon-FM Runner Script")
    parser.add_argument(
        "mode",
        choices=[mode.value for mode in TrainerModes],
        help="Mode to run.",
    )
    # General arguments
    parser.add_argument("--user", type=str, required=True)
    parser.add_argument("--cluster", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--time", type=str, default="03:55:00")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dryrun", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default="codon-fm")
    parser.add_argument("--local_out_dir", type=str, default=None)
    parser.add_argument("--entity", type=str, default="bio-foundation-models")
    parser.add_argument("--enable_wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=False, default=None,
                       help="Path to data. Required for most datasets, but not for GenerationPromptDataset.")
    parser.add_argument(
        "--process_item", 
        type=str, 
        required=True, 
        choices=['mlm_memmap', 'mutation_pred_mlm', 'mutation_pred_likelihood', 'codon_sequence', 'clm_memmap', 'mutation_pred_clm', 'generation_prompt', 'missense_seq', 'missense_inference']
    )
    parser.add_argument("--dataset_name", type=str, required=True, choices=["CodonMemmapDataset", "MutationDataset", "CodonBertDataset", "MissenseDataset", "GenerationPromptDataset"])
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--groups_to_use", type=str, nargs="+", default=[])
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--train_val_test_ratio", type=float, nargs=3, default=[0.9998, 0.0002, 0.00])
    parser.add_argument("--taxid_exclusion_file", type=str, default=None)
    parser.add_argument("--split_name_prefix", type=str, default="")

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, choices=["encodon_80m", "encodon_600m", "encodon_1b", "encodon_5b", "encodon_10b", "decodon_200m", "decodon_1b"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=10_000_000)
    parser.add_argument("--lr_total_iterations", type=int, default=10_000_000)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--mask_replace_prob", type=float, default=0.8)
    parser.add_argument("--random_replace_prob", type=float, default=0.1)
    parser.add_argument("--mask_mutation", action="store_true", default=False)
    parser.add_argument("--warmup_iterations", type=int, default=10_000)

    # Pretrain specific
    parser.add_argument("--codon_weights_file", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--organism_tokens_file", type=str, default=None)

    # Eval specific
    parser.add_argument("--extract-seq", action="store_true", default=False, help="For mutation prediction, whether to extract sequence.")
    parser.add_argument("--predictions_output_dir", type=str, default=None, help="For evaluation, the directory to write predictions to.")
    parser.add_argument("--task_type", type=str, default=None, help="For evaluation, the task type to run.")
    parser.add_argument("--organism_token", type=str, default=None, help="Optional organism token for DeCodon mutation prediction (e.g., '9606' or '<9606>' for human). If not provided, no prefix will be added to sequences.")
    parser.add_argument("--causal", action="store_true", default=False, help="Use causal/autoregressive context extraction (max left context) for mutation prediction. Recommended for DeCodon models.")
    
    # Sequence generation specific arguments
    parser.add_argument("--organism_id", type=str, default=None, help="Organism ID to generate sequences for (e.g., 9606 for human). Used with GenerationPromptDataset.")
    parser.add_argument("--num_sequences_per_organism", type=int, default=1, help="Number of sequences to generate for the organism.")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of tokens to generate per sequence.")
    parser.add_argument("--generation_temperature", type=float, default=0.9, help="Sampling temperature for generation.")
    parser.add_argument("--generation_top_k", type=int, default=50, help="Top-k sampling parameter for generation.")
    parser.add_argument("--generation_top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter for generation.")
    parser.add_argument("--use_stop_codons", action="store_true", default=True, help="Stop generation at stop codons (TAA, TAG, TGA).")
    parser.add_argument("--no_stop_codons", action="store_true", default=False, help="Don't stop at stop codons.")
    
    # Finetune specific
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint for finetuning or evaluation.")
    parser.add_argument("--loss_type", choices=["regression", "classification", "missense_synom_agg"], default="regression")
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--value_col", type=str, default="value", help="Column name for target values when using CodonBertDataset.")
    parser.add_argument("--ref_seq_col", type=str, default="ref_seq")
    parser.add_argument("--resume_trainer_state", action="store_true", default=False)
    parser.add_argument("--checkpoint_every_n_train_steps", type=int, default=2000)
    parser.add_argument("--finetune_strategy", type=str, default="full", choices=["lora", "head_only_random", "head_only_pretrained", "full"], help="Finetuning strategy.")
    parser.add_argument("--lora", action="store_true", default=False, help="Whether to use LoRA for finetuning.")
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification tasks.")
    parser.add_argument("--use_downstream_head", action="store_true", default=False, help="Whether to use downstream cross-attention head.")
    parser.add_argument("--cross_attention_hidden_dim", type=int, default=512, help="Hidden dimension for cross attention.")
    parser.add_argument("--cross_attention_num_heads", type=int, default=8, help="Number of heads for cross attention.")
    parser.add_argument("--missense_use_weights", action="store_true", default=False, help="Whether to use weights for missense loss.")
    parser.add_argument("--missense_center_weight_threshold", type=float, default=0.7, help="Threshold for center weight in missense loss.")
    parser.add_argument("--missense_use_paiv1", action="store_true", default=False, help="Whether to use PAIV1 benign variants.")
    parser.add_argument("--missense_use_am", action="store_true", default=False, help="Whether to use AM benign variants.")
    parser.add_argument("--missense_n_per_benign", type=int, default=1, help="Number of benign variants to sample per benign variant.")
    parser.add_argument("--missense_variants_per_seq", type=int, default=1, help="Number of variants to sample per sequence.")

    # Common trainer flags
    parser.add_argument("--enable_fsdp", action="store_true", default=False)
    parser.add_argument("--val_check_interval", type=int, default=1000)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=None, help="Run validation every n epochs. Overrides val_check_interval.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before performing a weight update.")
    parser.add_argument("--sharded_state_dict", action="store_true", default=False, help="Whether to shard the state dict.")
    parser.add_argument("--limit_val_batches", type=int, default=50)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--enable_one_logger", action="store_true", default=False, help="Enable one logger callback for logging.")
    
    return parser


def main():
    global CONTAINER_OUT_DIR, CONTAINER_CHECKPOINTS_DIR, CONTAINER_DATA_DIR, CONTAINER_PRETRAINED_CKPT_PATH
    parser = get_parser()
    args = parser.parse_args()
    if args.mode == TrainerModes.PREDICT and not args.checkpoint_path:
        parser.error(f"--checkpoint_path is required for mode '{args.mode.value}'")
    
    # Validate data_path is provided for datasets that require it
    if args.dataset_name != "GenerationPromptDataset" and args.data_path is None:
        parser.error(f"--data_path is required for dataset '{args.dataset_name}'")
    
    # Validate organism_id is provided for GenerationPromptDataset
    if args.dataset_name == "GenerationPromptDataset" and not args.organism_id:
        parser.error("--organism_id is required for GenerationPromptDataset")
    cfg = get_config(args)

    # Get executor
    if args.cluster == "local":
        assert args.local_out_dir is not None, "local_out_dir is required for local mode"
        CONTAINER_OUT_DIR = args.local_out_dir
        CONTAINER_PRETRAINED_CKPT_PATH = args.checkpoint_path
        CONTAINER_CHECKPOINTS_DIR = os.path.join(args.local_out_dir, "checkpoints/")
        
    executor_fn = get_executor_fn(args.cluster)
    

    custom_mounts = []
    exp_name = args.exp_name
    executor = executor_fn(
        nodes=args.num_nodes,
        devices=args.num_gpus,
        time=args.time,
        exp_name=exp_name,
        project_name=args.project_name,
        custom_mounts=custom_mounts,
        finetune_ckpt_path=args.checkpoint_path if args.mode == TrainerModes.FINETUNE and args.checkpoint_path else None,
        eval_ckpt_path=args.checkpoint_path if args.mode == TrainerModes.PREDICT and args.checkpoint_path else None
    )
    
    cfg_dict = config_to_dict(cfg)
    # Setup WandB plugins
    run_plugins = []
    if args.enable_wandb and "WANDB_API_KEY" in os.environ:
        run_plugins.append(
            WandbPlugin(
                name=exp_name,
                logger_fn=wandb_logger(
                    name=exp_name,
                    project=args.project_name,
                    entity=args.entity,
                    output_dir=CONTAINER_OUT_DIR,
                ),
            )
        )
    else:
        log.info("WandB disabled or WANDB_API_KEY not found. Skipping WandB logging.")

    cfg_dict["seed"] = args.seed
    cfg_dict["out_dir"] = CONTAINER_OUT_DIR
    cfg_dict["enable_one_logger"] = args.enable_one_logger
    # Define task
    if args.mode == TrainerModes.PRETRAIN:
        task_fn = train
        ckpt_path = f"{CONTAINER_CHECKPOINTS_DIR}/last.ckpt"
        cfg_dict["ckpt_path"] = ckpt_path
        task = run.Partial(
            task_fn,
            config=cfg,
            ckpt_path=ckpt_path,
            seed=args.seed,
            config_dict=cfg_dict,
            out_dir=CONTAINER_OUT_DIR,
        )
    elif args.mode == TrainerModes.FINETUNE:
        task_fn = finetune
        ckpt_path = f"{CONTAINER_CHECKPOINTS_DIR}/last.ckpt"
        cfg_dict["ckpt_path"] = ckpt_path
        cfg_dict["pretrained_ckpt_path"] = args.checkpoint_path
        cfg_dict["resume_trainer_state"] = args.resume_trainer_state
        task = run.Partial(
            task_fn,
            config=cfg,
            pretrained_ckpt_path=CONTAINER_PRETRAINED_CKPT_PATH,
            seed=args.seed,
            resume_trainer_state=args.resume_trainer_state,
            config_dict=cfg_dict,
            out_dir=CONTAINER_OUT_DIR,
            ckpt_path=ckpt_path,
        )
    elif args.mode == TrainerModes.PREDICT:
        task_fn = evaluate
        task = run.Partial(
            task_fn,
            config=cfg,
            config_dict=cfg_dict,
            model_ckpt_path=args.checkpoint_path,
            seed=args.seed,
            out_dir=CONTAINER_OUT_DIR,
        )

    with run.Experiment(exp_name) as exp:
        for i in range(args.num_jobs):
            exp.add(
                task,
                executor=executor,
                plugins=run_plugins,
                name=f"{exp_name}-{i}",
                tail_logs=True if isinstance(executor, run.LocalExecutor) else False,
            )

        if args.dryrun:
            exp.dryrun()
        else:
            exp.run(sequential=True, detach=True)


if __name__ == "__main__":
    main() 