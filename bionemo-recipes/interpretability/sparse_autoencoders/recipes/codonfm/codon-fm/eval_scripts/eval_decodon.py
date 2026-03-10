import os
import subprocess
import argparse
from typing import Dict, List
from src.data.metadata import TrainerModes
from src.inference.task_types import TaskTypes
from src.executors.base import CONTAINER_OUT_DIR

"""
This script is used to evaluate DeCodon model checkpoints on various tasks.
"""

# 1. Define Datasets
DATASETS = {
    "clinvar_alphamissense": {
        "data_path": "/data/validation/processed/alphamissense_clinvar_processed.csv",
        "extra_args": [],
    },
    "cancer_hotspot_alphamissense": {
        "data_path": "/data/validation/processed/alphamissense_cancer_hotspot_processed.csv",
        "extra_args": [],
    },
    "ddd_asd_zhouetal": {
        "data_path": "/data/validation/processed/ddd_asd_zhouetal_processed_am.csv",
        "extra_args": [],
    },
    "ddd_alphamissense": {
        "data_path": "/data/validation/processed/alphamissense_ddd_processed.csv",
        "extra_args": [],
    },
    "clinvar_author": {
        "data_path": "/data/validation/processed/clinvar_author.csv",
        "extra_args": [],
    },
    "human_val_10k": {
        "data_path": "/data/ncbi/processed_unfiltered/",
        "extra_args": ["--split_name_prefix", "human_val_10k", "--context_length", "2048"],
    },
    "clinvar_synom": {
        "data_path": "/data/validation/processed/synonymous_variants_singleton_vs_common_cds_frac_matched_with_baselines_scores.csv",
        "extra_args": [],
    },
    "clinvar_synom_with_baselines_scores": {
        "data_path": "/data/validation/processed/clinvar_synom_with_baselines_scores.csv",
        "extra_args": [],
    },
    "chd_missense": {
        "data_path": "/data/validation/processed/chd_dnm_filtered_canonical_transcripts_ddd_asd_ctrls_am_scores_cds_features.csv",
        "extra_args": [],
    },
    "ribonn_mean_te": {
        'data_path': '/data/validation/processed/latest_for_paper/data_with_human_TE_cellline_all_NA_plain.processed.csv',
        'extra_args': ['--value_col', 'mean_te', '--ref_seq_col', 'cds_sequence'],
    },
    "mrfp_expression": {
        'data_path': '/data/validation/processed/latest_for_paper/mRFP_Expression.csv',
        'extra_args': ['--value_col', 'value', '--ref_seq_col', 'ref_seq'],
    },
    # Special placeholder for sequence generation - uses --organism_id instead of data_path
    "generation": {
        'data_path': None,  # Not used for generation
        'extra_args': [],
    },
}

# 2. Define Task-specific configurations for DeCodon models
TASK_CONFIGS = {
    TaskTypes.EMBEDDING_PREDICTION: {
        "process_item": "codon_sequence",
        "dataset_name": "CodonBertDataset",
        "extra_args": [],
        "datasets": ["ribonn_mean_te", "mrfp_expression"],
    },
    TaskTypes.MUTATION_PREDICTION: {
        "process_item": "mutation_pred_clm",
        "dataset_name": "MutationDataset",
        "extra_args": ["--extract-seq"],
        "datasets": [
            "clinvar_alphamissense",
            "cancer_hotspot_alphamissense",
            "ddd_asd_zhouetal",
            "ddd_alphamissense",
            "clinvar_author",
            "clinvar_synom",
            "clinvar_synom_with_baselines_scores",
            "chd_missense",
        ],
    },
    TaskTypes.SYNONYMOUS_MUTATION_PREDICTION: {
        "process_item": "mutation_pred_clm",
        "dataset_name": "MutationDataset",
        "extra_args": [],
        "datasets": [
            "synonymous_variant_author",
        ],
    },
    TaskTypes.NEXT_CODON_PREDICTION: {
        "process_item": "clm_memmap",
        "dataset_name": "CodonMemmapDataset",
        "extra_args": [],
        "datasets": ["human_val_10k"],
    },
    TaskTypes.SEQUENCE_GENERATION: {
        "process_item": "generation_prompt",
        "dataset_name": "GenerationPromptDataset",
        "extra_args": [],
        "datasets": ["generation"],  # uses --organism_id instead
    },
}

TASKS = {}
for task_type, config in TASK_CONFIGS.items():
    TASKS[task_type] = {}
    # make a copy of config to avoid modifying the original
    task_specific_config = config.copy()
    dataset_names = task_specific_config.pop("datasets")
    for dataset_name in dataset_names:
        if dataset_name in DATASETS:
            TASKS[task_type][dataset_name] = {**task_specific_config, **DATASETS[dataset_name]}


def calculate_vocab_size(organism_tokens_file: str) -> int:
    """Calculate vocab size from organism tokens file."""
    base_vocab_size = 69  # 5 special tokens + 64 codons
    try:
        with open(organism_tokens_file, 'r') as f:
            organism_token_count = sum(1 for line in f if line.strip())
        vocab_size = base_vocab_size + organism_token_count
        print(f"Calculated vocab_size: {vocab_size} ({base_vocab_size} base + {organism_token_count} organism tokens)")
        return vocab_size
    except FileNotFoundError:
        print(f"Warning: organism_tokens_file '{organism_tokens_file}' not found, using default vocab_size=69")
        return 69


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeCodon model checkpoints with specified parameters.")
    parser.add_argument("--user", type=str, default="sdarabi", help="User name for runner.")
    parser.add_argument("--cluster", type=str, default="local", help="Cluster to run on.")
    parser.add_argument("--time", type=str, default="03:55:00", help="Time limit for job (HH:MM:SS format).")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use.")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers to use.")
    parser.add_argument("--local_out_dir", type=str, default="/data/predictions", help="Output directory (for local cluster).")
    parser.add_argument("--predictions_output_dir", type=str, default="/results/", help="Output directory for predictions (for non-local clusters).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Full path to the checkpoint file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--model_name", type=str, required=True, choices=["decodon_trial", "decodon_200m", "decodon_1b", "decodon_50m"])
    parser.add_argument(
        "--task",
        nargs=2,
        metavar=("TASK_TYPE", "DATASET_NAME"),
        action="append",
        required=False,
        help="Specify a task to run, consisting of a task type and a dataset name. Can be specified multiple times. E.g., --task MUTATION_PREDICTION clinvar_alphamissense. If not specified, all tasks will be run.",
    )
    parser.add_argument("--exp_suffix", type=str, default="", required=False,
                       help="Suffix to add to the experiment name.")
    
    # DeCodon-specific arguments
    parser.add_argument("--vocab_size", type=int, default=None,
                       help="Vocabulary size (optional - will be auto-calculated from organism tokens)")
    parser.add_argument("--context_length", type=int, default=2048,
                       help="Maximum sequence length for models")
    parser.add_argument("--split_name_prefix", type=str, default="",
                       help="Split name prefix for dataset caching (e.g., '2taxa', '14taxa')")
    parser.add_argument("--organism_tokens_file", type=str, required=True,
                       help="Path to organism tokens file (REQUIRED for DeCodon models, e.g., '/data/human_sc_only_organism_tokens.txt')")
    parser.add_argument("--taxid_exclusion_file", type=str, default=None,
                       help="Path to taxid exclusion file (e.g., '/data/taxids_to_remove__keep_only_human_sc.json')")
    parser.add_argument("--organism_token", type=str, default=None,
                       help="Optional organism token for DeCodon mutation prediction (e.g., '9606' or '<9606>' for human). If not provided, no prefix will be added to sequences.")
    parser.add_argument("--groups_to_use", type=str, nargs="+", default=[],
                       help="List of organism groups to use for evaluation (e.g., Primates vertebrate_mammalian)")
    parser.add_argument("--causal", action="store_true", default=True,
                       help="Use causal/autoregressive context extraction (max left context) for mutation prediction.")
    
    # Sequence generation specific arguments
    parser.add_argument("--organism_id", type=str, default=None,
                       help="Organism ID to generate sequences for (e.g., 9606 for human). Required for SEQUENCE_GENERATION task.")
    parser.add_argument("--num_sequences_per_organism", type=int, default=100,
                       help="Number of sequences to generate for the organism.")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                       help="Maximum number of tokens to generate per sequence.")
    parser.add_argument("--generation_temperature", type=float, default=0.9,
                       help="Sampling temperature for generation.")
    parser.add_argument("--generation_top_k", type=int, default=50,
                       help="Top-k sampling parameter for generation.")
    parser.add_argument("--generation_top_p", type=float, default=0.95,
                       help="Top-p (nucleus) sampling parameter for generation.")
    parser.add_argument("--no_stop_codons", action="store_true", default=False,
                       help="Don't stop generation at stop codons.")
    
    args = parser.parse_args()

    # Auto-calculate vocab_size from organism tokens file if not provided
    if args.vocab_size is None:
        args.vocab_size = calculate_vocab_size(args.organism_tokens_file)
    else:
        print(f"Using manually specified vocab_size: {args.vocab_size}")

    # construct a list of (task_type, dataset_name) tuples
    if args.task:
        task_dataset = [tuple(t) for t in args.task]
    else:
        print("No specific task provided, running all tasks.")
        task_dataset = []
        for task_type, datasets in TASKS.items():
            for dataset_name in datasets.keys():
                task_dataset.append((task_type, dataset_name))

    # Validate tasks
    for task_type, dataset_name in task_dataset:
        if task_type not in TASK_CONFIGS:
            parser.error(f"Invalid task type: {task_type}. Choices are {list(TASK_CONFIGS.keys())}")
        allowed_datasets = TASK_CONFIGS[task_type]["datasets"]
        if dataset_name not in allowed_datasets:
            parser.error(f"Invalid dataset for task type {task_type}: {dataset_name}. Choices are {allowed_datasets}")

    # derive a directory name for outputs from the checkpoint path
    # assumes a structure like .../experiment_name/checkpoints/checkpoint.ckpt
    dir_name = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint_path)))
    print(f"Processing checkpoint: {args.checkpoint_path}")
    
    # Determine output directory based on cluster type
    if args.cluster == "local":
        output_base_dir = args.local_out_dir
    else:
        output_base_dir = args.predictions_output_dir
    
    for task_type, dataset_name in task_dataset:
        task = TASKS[task_type][dataset_name]
        exp_name = f"{dir_name}_{args.model_name}_{dataset_name}"
        project_name = "decodon-eval"
        if task_type == TaskTypes.SEQUENCE_GENERATION:
            exp_name += f"_organism_id_{args.organism_id}_num_sequences_{args.num_sequences_per_organism}_temperature_{args.generation_temperature}_top_k_{args.generation_top_k}_top_p_{args.generation_top_p}"
            project_name = "decodon-generation"
        if args.exp_suffix:
            exp_name += f"_{args.exp_suffix}"
        ckpt_name = os.path.basename(args.checkpoint_path).strip("ckpt").strip(".")
        predictions_dir = os.path.join(output_base_dir, dir_name, f"{ckpt_name}", task_type, dataset_name)
        
        # Build base command
        evaluate_command = [
            "python",
            "-m",
            "src.runner", TrainerModes.PREDICT,
            "--user", args.user,
            "--cluster", args.cluster,
            "--time", args.time,
            "--val_batch_size", str(args.batch_size),
            "--checkpoint_path", args.checkpoint_path,
            "--model_name", args.model_name,
            "--dataset_name", task["dataset_name"],
            "--process_item", task["process_item"],
            "--exp_name", exp_name,
            "--num_gpus", str(args.num_gpus),
            "--num_jobs", "1",
            "--num_nodes", str(args.num_nodes),
            "--num_workers", str(args.num_workers),
            "--predictions_output_dir", predictions_dir,
            "--task_type", task_type,
            "--project_name", project_name,
        ]
        

        if task_type == TaskTypes.SEQUENCE_GENERATION:
            # SEQUENCE_GENERATION uses --organism_id instead of --data_path
            if args.organism_id is None:
                parser.error("--organism_id is required for SEQUENCE_GENERATION task")
            evaluate_command.extend(["--organism_id", args.organism_id])
            evaluate_command.extend(["--num_sequences_per_organism", str(args.num_sequences_per_organism)])
            evaluate_command.extend(["--max_new_tokens", str(args.max_new_tokens)])
            evaluate_command.extend(["--generation_temperature", str(args.generation_temperature)])
            evaluate_command.extend(["--generation_top_k", str(args.generation_top_k)])
            evaluate_command.extend(["--generation_top_p", str(args.generation_top_p)])
            if args.no_stop_codons:
                evaluate_command.append("--no_stop_codons")
            # Use a dummy data_path (not used but required by runner.py arg parser)
            evaluate_command.extend(["--data_path", "/dev/null"])
        else:
            evaluate_command.extend(["--data_path", task["data_path"]])
        
        # Add DeCodon-specific parameters
        if args.vocab_size is not None:
            evaluate_command.extend(["--vocab_size", str(args.vocab_size)])
        if args.context_length:
            evaluate_command.extend(["--context_length", str(args.context_length)])
        if args.split_name_prefix:
            evaluate_command.extend(["--split_name_prefix", args.split_name_prefix])
        if args.organism_tokens_file:
            evaluate_command.extend(["--organism_tokens_file", args.organism_tokens_file])
        if args.taxid_exclusion_file:
            evaluate_command.extend(["--taxid_exclusion_file", args.taxid_exclusion_file])
        if args.groups_to_use:
            evaluate_command.extend(["--groups_to_use"] + args.groups_to_use)
        if args.organism_token:
            evaluate_command.extend(["--organism_token", args.organism_token])
        if args.causal:
            evaluate_command.append("--causal")
        
        if args.cluster == "local":
            evaluate_command.extend([
                "--local_out_dir", f"{output_base_dir}/{dir_name}",
            ])

        # Add task-level extra arguments
        task_config = TASK_CONFIGS[task_type]
        if task_config["extra_args"]:
            evaluate_command.extend(task_config["extra_args"])
        
        # Add dataset-level extra arguments
        if task["extra_args"]:
            evaluate_command.extend(task["extra_args"])

        print(f"Running task for {dir_name} with model {args.model_name} and dataset {dataset_name}")
        print("Command:", " ".join(evaluate_command))
        subprocess.run(evaluate_command, check=True)

    print("All tasks completed!")


if __name__ == "__main__":
    main()