import os
import subprocess
import argparse
from typing import Dict, List
from src.data.metadata import TrainerModes
from src.inference.task_types import TaskTypes
from src.executors.base import CONTAINER_OUT_DIR

"""
This script is used to evaluate Encodon model checkpoints on various tasks.
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
    "mavedb_author": {
        "data_path": "/data/validation/processed/mavedb_author.csv",
        "extra_args": [],
    },
    "pretrain_eval": {
        "data_path": "/data/ncbi/processed_unfiltered/",
        "extra_args": [],
    },
    "human_val_10k": {
        "data_path": "/data/ncbi/processed_unfiltered/",
        "extra_args": ["--split_name_prefix", "human_val_10k", "--context_length", "2048", "--min_seq_length", "100", "--max_seq_length", "150000"],
    },
    "synonymous_variant_author": {
        "data_path": "/data/synonymous_variant_author/data.csv",
        "extra_args": [],
    },
    "chd_missense": {
        "data_path": "/data/validation/processed/chd_dnm_filtered_canonical_transcripts_ddd_asd_ctrls_am_scores_cds_features.csv",
        "extra_args": [],
    },
    "clinvar_synom_with_baselines_scores": {
        "data_path": "/data/validation/processed/latest_for_paper/clinvar_synom_with_baselines_scores.csv",
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
    "cosmic_mutantcensus": {
        "data_path": "/data/validation/processed/cosmic/cosmic_mutantcensus.csv",
        'extra_args': []
    },
    "gnomad_af0.01_canonical_genes": {
        "data_path": "/data/validation/processed/gnomad_af0.01_canonical_genes.csv",
        'extra_args': []
    },

}

# 2. Define Task-specific configurations for Encodon models
TASK_CONFIGS = {
    TaskTypes.MUTATION_PREDICTION: {
        "process_item": "mutation_pred_mlm",
        "dataset_name": "MutationDataset",
        "extra_args": ["--mask_mutation", "--extract-seq"],
        "datasets": [
            "clinvar_alphamissense",
            "clinvar_synom_with_baselines_scores",
            "cancer_hotspot_alphamissense",
            "ddd_asd_zhouetal",
            "ddd_alphamissense",
            "clinvar_author",
            "chd_missense",
            "cosmic_mutantcensus",
            "gnomad_af0.01_canonical_genes",
        ],
    },
    TaskTypes.SYNONYMOUS_MUTATION_PREDICTION: {
        "process_item": "mutation_pred_mlm",
        "dataset_name": "MutationDataset",
        "extra_args": [],
        "datasets": [
            "synonymous_variant_author",
        ],
    },
    TaskTypes.FITNESS_PREDICTION: {
        "process_item": "mutation_pred_likelihood",
        "dataset_name": "MutationDataset",
        "extra_args": ["--mask_mutation"],
        "datasets": ["mavedb_author"],
    },
    TaskTypes.MASKED_LANGUAGE_MODELING: {
        "process_item": "mlm_memmap",
        "dataset_name": "CodonMemmapDataset",
        "extra_args": [],
        "datasets": ["pretrain_eval", "human_val_10k"],
    },
    TaskTypes.EMBEDDING_PREDICTION: {
        "process_item": "codon_sequence",
        "dataset_name": "CodonBertDataset",
        "extra_args": [],
        "datasets": ["ribonn_mean_te", "mrfp_expression"],
    },
    TaskTypes.MISSENSE_PREDICTION: {
        "process_item": "missense_inference",
        "dataset_name": "MutationDataset",
        "extra_args": ["--mask_mutation", "--extract-seq"],
        "datasets": [
            "ddd_asd_zhouetal",
            "clinvar_alphamissense",
            "cancer_hotspot_alphamissense",
        ],
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



def main():
    parser = argparse.ArgumentParser(description="Evaluate Encodon model checkpoints with specified parameters.")
    parser.add_argument("--user", type=str, default="sdarabi", help="User name for runner.")
    parser.add_argument("--cluster", type=str, default="local", help="Cluster to run on.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use.")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers to use.")
    parser.add_argument("--local_out_dir", type=str, default="/results/", help="Output directory.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Full path to the checkpoint file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--model_name", type=str, required=True, choices=["encodon_80m", "encodon_600m", "encodon_1b", "encodon_5b", "encodon_10b"])
    parser.add_argument(
        "--task",
        nargs=2,
        metavar=("TASK_TYPE", "DATASET_NAME"),
        action="append",
        required=False,
        help="Specify a task to run, consisting of a task type and a dataset name. Can be specified multiple times. E.g., --task MUTATION_PREDICTION clinvar_alphamissense. If not specified, all tasks will be run.",
    )
    
    args = parser.parse_args()

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

    for task_type, dataset_name in task_dataset:
        task = TASKS[task_type][dataset_name]
        exp_name = f"{dir_name}_{args.model_name}_{dataset_name}"
        ckpt_name = os.path.basename(args.checkpoint_path).strip("ckpt").strip(".")
        predictions_dir = os.path.join(args.local_out_dir, dir_name, f"{ckpt_name}", task_type, dataset_name)
        evaluate_command = [
            "python",
            "-m",
            "src.runner", TrainerModes.PREDICT,
            "--user", args.user,
            "--cluster", args.cluster,
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
            "--data_path", task["data_path"],
            "--predictions_output_dir", predictions_dir,
            "--task_type", task_type,
            "--project_name", "codon-fm-eval",
        ]
        
        if args.cluster == "local":
            evaluate_command.extend([
                "--local_out_dir", f"{args.local_out_dir}/{dir_name}",
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