import os
import argparse
import pandas as pd
import numpy as np
import torch
import sys
import time
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef, ndcg_score
import nemo.lightning as nl
from pathlib import Path
import gc

from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

from torch.utils.data import Dataset
from lightning.pytorch import LightningDataModule
from nemo.lightning.data import WrappedDataLoader

# Import functions from eval_ppl.py to avoid redundancy
try:
    from eval_ppl import (
        HyenaPredictor, 
        hyena_predict_forward_step, 
        hyena_predict_data_step, 
        pad_collate_fn
    )
except ImportError as e:
    raise ImportError(
        f"Could not import from eval_ppl.py: {e}. "
        "Make sure eval_ppl.py is in the same directory."
    )

# Functions imported from eval_ppl.py:
# - HyenaPredictor  
# - hyena_predict_forward_step
# - hyena_predict_data_step
# - pad_collate_fn

# TODO: figure out why ndcg code returns error 
# (currently i do try block and ignore unsuccessful as nan)
# TODO: save logic to have subfolder for each model


class DMSDataset(Dataset):
    """Dataset for DMS sequences from CSV data."""
    
    def __init__(self, sequences, tokenizer):
        """
        Args:
            sequences: List of DNA/RNA sequences from DMS data
            tokenizer: Tokenizer to use for sequence encoding
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Tokenize the sequence (same method as SimpleFastaDataset)
        tokens = self.tokenizer.text_to_ids(seq)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Create position IDs
        position_ids = torch.arange(len(tokens), dtype=torch.long)
        
        # Create loss mask (all ones since we want to compute loss on all tokens)
        loss_mask = torch.ones(len(tokens), dtype=torch.long)
        
        return {
            "tokens": tokens_tensor,
            "position_ids": position_ids,
            "labels": tokens_tensor,  # For autoregressive loss
            "loss_mask": loss_mask,
            "seq_idx": torch.tensor(idx, dtype=torch.long),
        }


class DMSDataModule(LightningDataModule):
    """DataModule for DMS prediction."""
    
    def __init__(
        self, 
        sequences, 
        tokenizer, 
        batch_size=1, 
        tensor_parallel_size=1, 
        num_workers=8
    ):
        super().__init__()
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.dataset = DMSDataset(self.sequences, self.tokenizer)
    
    def predict_dataloader(self):
        return WrappedDataLoader(
            mode="predict",
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda batch: pad_collate_fn(
                batch, self.tokenizer, self.tensor_parallel_size
            ),
        )


def get_likelihood_results_path(args):
    """Generate path for saving likelihood results."""
    model_name = args.ckpt_dir.stem  # Use checkpoint directory name as model identifier
    file_name = args.DMS_id + "_" + model_name + "_likelihoods.csv"
    results_dir = os.path.join(
        args.output_performance_file_folder, 
        args.DMS_path.split('/')[-2], 
        args.taxon, 
        model_name
    )
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, file_name)
    return output_path


def get_fitness_results_path(args):
    """Generate path for saving fitness results."""
    model_name = args.ckpt_dir.stem  # Use checkpoint directory name as model identifier
    file_name = args.DMS_id + "_" + model_name + "_fitness.csv"
    results_dir = os.path.join(
        args.output_performance_file_folder, 
        args.DMS_path.split('/')[-2], 
        args.taxon, 
        model_name
    )
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, file_name)
    return output_path


def logits_to_logprobs(logits, input_ids):
    """
    Official Evo2 scoring approach: log_softmax + gather instead of cross_entropy.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        
    Returns:
        torch.Tensor: Log probabilities for each token [batch_size, seq_len-1]
    """
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]  # Remove last position
    input_ids = input_ids[:, 1:]                 # Remove first position (autoregressive)
    
    logprobs = torch.gather(
        softmax_logprobs,       # Gather likelihoods...
        2,                      # along the vocab dimension...
        input_ids.unsqueeze(-1) # using the token ids to index.
    ).squeeze(-1)
    return logprobs


def get_model_likelihoods(model, trainer, tokenizer, DMS_df, args, seq_len=8192):
    """
    Computes model likelihoods for sequences in the DMS dataframe using NeMo Lightning framework.
    
    Args:
        model: The HyenaPredictor model
        trainer: The NeMo Lightning trainer
        tokenizer: The byte-level tokenizer
        DMS_df: DataFrame containing sequences with 'mutated_sequence' column
        args: Command line arguments containing batch_size, tensor_parallel_size, etc.
        seq_len: Maximum sequence length for the model
    
    Returns:
        DataFrame with same index as DMS_df and 'log_likelihood' column containing scores.
        
    Note:
        The reduce_method (sum/mean) is already configured in the model via log_prob_collapse_option.
    """
    print(
        f"Computing model likelihoods for {len(DMS_df)} DMS sequences "
        "using NeMo Lightning framework..."
    )
    
    # Validate required columns
    if 'mutated_sequence' not in DMS_df.columns:
        raise ValueError("DMS dataframe must contain 'mutated_sequence' column")
    
    # Extract sequences from DMS dataframe
    sequences = DMS_df['mutated_sequence'].tolist()
    
    # Check for sequences that might be too long
    long_sequences = []
    for i, seq in enumerate(sequences):
        tokens = tokenizer.text_to_ids(seq)
        if len(tokens) > seq_len:
            long_sequences.append((i, len(tokens)))
    
    if long_sequences:
        print(f"Warning: {len(long_sequences)} sequences exceed context window of {seq_len} tokens")
        print(f"Longest sequence: {max(long_sequences, key=lambda x: x[1])[1]} tokens")
        print("These sequences will be truncated during processing")
    
    # Create dataset and datamodule
    datamodule = DMSDataModule(
        sequences=sequences, 
        tokenizer=tokenizer, 
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        num_workers=getattr(args, 'num_workers', 8)
    )
    
    print(f"Running inference on {len(sequences)} sequences...")
    
    # try:
    #     # Run prediction using trainer (same as eval_ppl.py)
    results = trainer.predict(model, datamodule=datamodule)

    # except Exception as e:
    #     print(f"Error during model prediction: {e}")
    #     raise
    
    # Process results with robust distributed handling
    # Use torch.distributed for more reliable rank checking
    is_main_rank = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main_rank = torch.distributed.get_rank() == 0
    
    # Process results if available, regardless of rank (more robust for TP setups)
    if results and len(results) > 0:
        try:
            # Extract log probabilities and sequence indices
            log_probs = torch.cat([r["log_probs_seqs"] for r in results])
            seq_indices = torch.cat([r["seq_idx"] for r in results])
            
            # Sort by original sequence index to match DMS_df order
            sorted_indices = torch.argsort(seq_indices)
            sorted_log_probs = log_probs[sorted_indices].cpu().numpy()
            
            # Convert to the format expected by fitness evaluation
            likelihoods = sorted_log_probs.tolist()
            
            if is_main_rank:
                print(f"Computed {len(likelihoods)} log-likelihoods using {args.reduce_method} method")
                print(f"Sample log-likelihoods: {likelihoods[:5]}")
        except Exception as e:
            print(f"Error processing results: {e}")
            likelihoods = [None] * len(DMS_df)
    else:
        if is_main_rank:
            print("No results from trainer.predict() - using None values")
        likelihoods = [None] * len(DMS_df)
    
    # Create result DataFrame with same index as input
    result_df = pd.DataFrame({'log_likelihood': likelihoods}, index=DMS_df.index)
    
    # Save likelihoods to CSV
    output_path = get_likelihood_results_path(args)
    try:
        result_df.to_csv(output_path)
        print(f"Saved likelihoods to: {output_path}")
    except Exception as e:
        print(f"Warning: Failed to save likelihoods to {output_path}: {e}")
    
    return result_df


def get_performance_results(
    merged_df, 
    DMS_score_column, 
    model_score_column, 
    DMS_binary_score_column
):
    """
    Computes performance metrics of interest.
    
    Args:
        merged_df: DataFrame with both DMS scores and model predictions
        DMS_score_column: Name of column containing experimental DMS scores
        model_score_column: Name of column containing model predictions
        DMS_binary_score_column: Name of column containing binary DMS scores
        
    Returns:
        Dictionary containing performance metrics
    """
    performance_results = {}
    
    # Validate required columns
    required_cols = [DMS_score_column, model_score_column]
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    y_true = merged_df[DMS_score_column]
    y_pred = merged_df[model_score_column]
    
    # Remove NaN values for continuous metrics
    valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    if valid_mask.sum() == 0:
        print("Warning: No valid (non-NaN) data points for performance evaluation")
        return {
            key: np.nan 
            for key in ['spearman', 'spearman_pvalue', 'ndcg', 'auc', 'mcc']
        }
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Continuous metrics
    try:
        spearman_result = spearmanr(y_true_valid, y_pred_valid)
        performance_results['spearman'] = spearman_result[0]
        performance_results['spearman_pvalue'] = spearman_result[1]
    except Exception as e:
        print(f"Warning: Failed to compute Spearman correlation: {e}")
        performance_results['spearman'] = np.nan
        performance_results['spearman_pvalue'] = np.nan
    
    try:
        performance_results['ndcg'] = ndcg_score(
            np.asarray([y_true_valid]), 
            np.asarray([y_pred_valid])
        )
    except Exception as e:
        print(f"Warning: Failed to compute NDCG: {e}")
        performance_results['ndcg'] = np.nan
        
    # Binary metrics
    if DMS_binary_score_column in merged_df.columns:
        merged_df_no_nan_binary = merged_df[
            merged_df[DMS_binary_score_column].notna()
        ].copy()
        
        if (len(merged_df_no_nan_binary) > 0 and 
            len(merged_df_no_nan_binary[DMS_binary_score_column].unique()) > 1):
            try:
                y_true_binary = merged_df_no_nan_binary[DMS_binary_score_column]
                y_pred_binary = merged_df_no_nan_binary[model_score_column]
                
                performance_results['auc'] = roc_auc_score(y_true_binary, y_pred_binary)
                
                # Convert to binary predictions using median threshold
                y_pred_binary_thresh = (
                    y_pred_binary > np.median(y_pred_binary)
                ).astype(int)
                performance_results['mcc'] = matthews_corrcoef(
                    y_true_binary, y_pred_binary_thresh
                )
            except Exception as e:
                print(f"Warning: Failed to compute binary metrics: {e}")
                performance_results['auc'] = np.nan
                performance_results['mcc'] = np.nan
        else:
            print("Warning: Insufficient binary data for AUC/MCC computation")
            performance_results['auc'] = np.nan
            performance_results['mcc'] = np.nan
    else:
        print(f"Warning: Binary score column '{DMS_binary_score_column}' not found")
        performance_results['auc'] = np.nan
        performance_results['mcc'] = np.nan

    return performance_results


def eval_DMS_file(model, trainer, tokenizer, args, seq_len=8192):
    """
    Evaluate a single DMS file.
    
    Args:
        model: The HyenaPredictor model
        trainer: The NeMo Lightning trainer  
        tokenizer: The byte-level tokenizer
        args: Command line arguments
        seq_len: Maximum sequence length for the model
    """
    print(f"Loading DMS data from: {args.DMS_path}")
    
    try:
        DMS_df = pd.read_csv(args.DMS_path)
    except Exception as e:
        print(f"Error loading DMS file {args.DMS_path}: {e}")
        return
    
    print(f"Loaded {len(DMS_df)} sequences from DMS file")
    
    # Use consistent DMS_id throughout
    DMS_id = args.DMS_id
    output_path = get_likelihood_results_path(args)

    print(f"Output path: {output_path}")
    if os.path.exists(output_path):
        print(f"Loading model likelihoods from: {output_path}")
        try:
            model_scores_df = pd.read_csv(output_path, index_col=0)
            print(f"Loaded {len(model_scores_df)} model likelihoods")
            
            # Verify that loaded data matches current DMS data
            if len(model_scores_df) != len(DMS_df):
                print(
                    f"Warning: Loaded likelihood data has {len(model_scores_df)} entries, "
                    f"but DMS data has {len(DMS_df)} entries"
                )
                print("Recomputing likelihoods...")
                model_scores_df = get_model_likelihoods(
                    model, trainer, tokenizer, DMS_df, args, seq_len
                )
        except Exception as e:
            print(f"Error loading pre-computed likelihoods: {e}")
            print("Recomputing likelihoods...")
            model_scores_df = get_model_likelihoods(
                model, trainer, tokenizer, DMS_df, args, seq_len
            )
    else:
        # Compute model likelihoods using NeMo Lightning approach
        print(f"Computing model likelihoods with reduce_method='{args.reduce_method}'...")
        model_scores_df = get_model_likelihoods(
            model, trainer, tokenizer, DMS_df, args, seq_len
        )

    # Merge DMS data with model predictions
    merged_df = pd.concat([DMS_df, model_scores_df], axis=1)

    # Evaluate performance
    try:
        performance_results = get_performance_results(
            merged_df, 
            args.DMS_score_column, 
            args.model_score_column, 
            args.DMS_binary_score_column
        )
    except Exception as e:
        print(f"Error computing performance results: {e}")
        return
    
    performance_df = pd.DataFrame([performance_results])
    performance_df.index = [DMS_id]
    
    # Add DMS reference information
    try:
        DMS_reference_df = pd.read_csv(args.DMS_reference_file_path)
        DMS_reference_entry = DMS_reference_df[DMS_reference_df['DMS_id'] == DMS_id]
        
        if len(DMS_reference_entry) > 0:
            for col in DMS_reference_entry.columns:
                value_series = DMS_reference_entry[col]
                performance_df[col] = value_series.tolist()[0]
        else:
            print(f"Warning: No reference entry found for DMS_id '{DMS_id}'")
    except Exception as e:
        print(f"Warning: Error loading DMS reference file: {e}")
   
    # Save fitness results
    experiment_type = args.DMS_path.split('/')[-2]
    model_name = args.ckpt_dir.stem
    output_dir = os.path.join(
        args.output_performance_file_folder, 
        experiment_type, 
        args.taxon, 
        model_name
    )
    os.makedirs(output_dir, exist_ok=True)
    
    fitness_output_path = os.path.join(
        output_dir, 
        DMS_id + "_" + model_name + "_fitness.csv"
    )
    try:
        performance_df.to_csv(fitness_output_path, index=False)
        print(f"Saved fitness results to: {fitness_output_path}")
    except Exception as e:
        print(f"Error saving fitness results: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Model fitness evaluation with NeMo Lightning framework'
    )
    
    # Model and checkpoint arguments (same as eval_ppl.py)
    parser.add_argument(
        "--ckpt-dir", 
        type=Path, 
        default="/workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k", 
        help="NeMo2 checkpoint directory for inference."
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="7b",
        choices=sorted(HYENA_MODEL_OPTIONS.keys()),
        help="Model size to use. Defaults to '7b'.",
    )
    parser.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated.",
    )
    
    # GPU and parallelism arguments (same as eval_ppl.py)
    parser.add_argument(
        "--devices", 
        type=str, 
        default="auto", 
        help="CUDA devices to use (e.g., '0,1' or 'auto'). Defaults to 'auto'."
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=1, 
        help="Order of tensor parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--pipeline-model-parallel-size", 
        type=int, 
        default=1, 
        help="Order of pipeline parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--context-parallel-size", 
        type=int, 
        default=1, 
        help="Order of context parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--no-sequence-parallel",
        action="store_true",
        help=(
            "When using TP, skip sequence parallelism. Otherwise sequence parallelism "
            "is used whenever tensor parallelism is used."
        ),
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1, 
        help="Batch size for prediction. Defaults to 1."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=8, 
        help="Number of workers for dataloader. Defaults to 8."
    )
    
    # DMS data arguments (fitness-specific)
    parser.add_argument(
        '--DMS_filenames', 
        type=str, 
        help='Path to a list of DMS filenames (text file with one filename per line, or CSV file if --DMS_filenames_column is specified)'
    )
    parser.add_argument(
        '--DMS_filenames_column', 
        type=str, 
        default='csv_filename',
        help='Column name to read DMS filenames from when --DMS_filenames points to a CSV file (if not specified, treats file as text with one filename per line)'
    )
    parser.add_argument(
        '--DMS_path', 
        type=str, 
        help='Path to DMS file'
    )
    parser.add_argument(
        '--DMS_scores_folder', 
        default='data/eval_dataset/DMS_ProteinGym_substitutions/nucleotides', 
        type=str, 
        help='Path to folder with model scores on DMS assays'
    )
    parser.add_argument(
        '--output_performance_file_folder', 
        default='results/test', 
        type=str, 
        help='Path to folder to save performance analysis'
    )
    parser.add_argument(
        '--DMS_reference_file_path', 
        default='data/eval_dataset/DMS_ProteinGym_substitutions/DMS_substitutions.csv', 
        type=str, 
        help='Path to DMS reference file'
    )
    parser.add_argument(
        '--DMS_score_column', 
        default='DMS_score', 
        type=str, 
        help='Name of DMS score column in DMS file'
    )
    parser.add_argument(
        '--DMS_binary_score_column', 
        default='DMS_score_bin', 
        type=str, 
        help='Name of DMS binary score column in DMS file'
    )
    parser.add_argument(
        '--model_score_column', 
        default='log_likelihood', 
        type=str, 
        help='Name of model score column'
    )
    parser.add_argument(
        '--reduce_method', 
        default='sum', 
        choices=['sum', 'mean'], 
        help='Likelihood reduction method: sum (total) or mean (length-normalized)'
    )

    args = parser.parse_args()

    # Validate DMS_filenames arguments
    if args.DMS_filenames_column and not args.DMS_filenames:
        parser.error("--DMS_filenames_column requires --DMS_filenames to be specified")
    
    if args.DMS_filenames and args.DMS_filenames.endswith('.csv') and not args.DMS_filenames_column:
        parser.error("When --DMS_filenames points to a CSV file, --DMS_filenames_column must be specified")

    # Reproducibility (same as eval_ppl.py)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    # Load DMS reference data
    try:
        DMS_reference_df = pd.read_csv(args.DMS_reference_file_path)
        print(f"Loaded DMS reference file with {len(DMS_reference_df)} entries")
    except Exception as e:
        print(f"Warning: Error loading DMS reference file {args.DMS_reference_file_path}: {e}")
        print("Creating empty reference dataframe")
        DMS_reference_df = pd.DataFrame()
        DMS_reference_df['DMS_id'] = []
        DMS_reference_df['taxon'] = []

    # Setup NeMo Lightning model and trainer (same as eval_ppl.py)
    sequence_parallel = args.tensor_parallel_size > 1 and not args.no_sequence_parallel
    model_parallel_size = (
        args.tensor_parallel_size * 
        args.pipeline_model_parallel_size * 
        args.context_parallel_size
    )
    
    # Handle device selection similar to eval_ppl.py
    if args.devices == "auto":
        devices = model_parallel_size
    else:
        # Parse device string like "0,1" or "2"
        device_list = [int(d.strip()) for d in args.devices.split(',')]
        devices = len(device_list)
        if devices != model_parallel_size:
            print(
                f"Warning: Number of devices ({devices}) doesn't match "
                f"model parallel size ({model_parallel_size})"
            )
            # Set CUDA_VISIBLE_DEVICES if specific devices are requested
            os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    
    if model_parallel_size > torch.cuda.device_count():
        raise ValueError(
            f"Requested model parallel size {model_parallel_size} is greater than the "
            f"number of available CUDA devices {torch.cuda.device_count()}"
        )

    seq_len = 8192 if "arc_longcontext" not in args.model_size else 32000
    print(f"Using sequence length: {seq_len}")

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=nl.MegatronStrategy(
            drop_last_batch=False,
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            context_parallel_size=args.context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            sequence_parallel=sequence_parallel,
            save_ckpt_format=args.ckpt_format,
            ckpt_load_strictness=None,
            data_sampler=nl.MegatronDataSampler(
                micro_batch_size=args.batch_size,
                global_batch_size=args.batch_size,
                seq_len=seq_len,
                output_log=False,
            ),
        ),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
    )

    config = HYENA_MODEL_OPTIONS[args.model_size](
        forward_step_fn=hyena_predict_forward_step,
        data_step_fn=hyena_predict_data_step,
        distribute_saved_activations=(
            False if sequence_parallel and args.tensor_parallel_size > 1 else True
        ),
    )
    
    # Disable optimizer setup for inference-only mode
    trainer.strategy._setup_optimizers = False  # type: ignore

    # FIX: Prevent repeated checkpoint loading during inference
    # Previously resume_if_exists=True was causing the model to reload checkpoints
    # for every trainer.predict() call, leading to intermittent distributed loading failures
    # Now checkpoint is loaded only once at startup for reliable inference
    resume = nl.AutoResume(
        resume_if_exists=False,  # Prevent repeated checkpoint loading during inference
        resume_ignore_no_checkpoint=False,
        resume_past_end=False,
        restore_config=nl.RestoreConfig(
            path=str(args.ckpt_dir),
            load_model_state=True,
            load_optim_state=False,
            load_artifacts=True,
        ),
    )
    
    tokenizer = get_nmt_tokenizer("byte-level")
    model = HyenaPredictor(
        config,
        tokenizer=tokenizer,
        output_log_prob_seqs=True,
        log_prob_collapse_option=args.reduce_method,
    )
    resume.setup(trainer, model)
    print("Model loaded.")  # One-time checkpoint load completed - no more reloading during inference

    if args.DMS_filenames:
        if args.DMS_filenames_column and args.DMS_filenames.endswith('.csv'):
            # Read DMS filenames from a CSV file
            try:
                DMS_filenames_df = pd.read_csv(args.DMS_filenames)
                if args.DMS_filenames_column not in DMS_filenames_df.columns:
                    print(f"Error: Column '{args.DMS_filenames_column}' not found in {args.DMS_filenames}")
                    print(f"Available columns: {list(DMS_filenames_df.columns)}")
                    sys.exit(1)
                DMS_filenames = DMS_filenames_df[args.DMS_filenames_column].tolist()
                print(f"Reading DMS filenames from column '{args.DMS_filenames_column}' of {args.DMS_filenames}")
            except FileNotFoundError:
                print(f"Error: CSV file not found at {args.DMS_filenames}")
                sys.exit(1)
            except Exception as e:
                print(f"Error reading CSV file {args.DMS_filenames}: {e}")
                sys.exit(1)
        else:
            # Read DMS filenames from a text file (one filename per line)
            try:
                with open(args.DMS_filenames, 'r') as f:
                    DMS_filenames = f.read().splitlines()
                print(f"Reading DMS filenames from text file: {args.DMS_filenames}")
            except FileNotFoundError:
                print(f"Error: Text file not found at {args.DMS_filenames}")
                sys.exit(1)
            except Exception as e:
                print(f"Error reading DMS filenames from {args.DMS_filenames}: {e}")
                sys.exit(1)
            
        for DMS_filename in DMS_filenames:
            # Force memory cleanup before each file
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations
            gc.collect()   # Python garbage collection
            
            print(f"\n=== Evaluating {DMS_filename} ===")
            
            args.DMS_path = os.path.join(args.DMS_scores_folder, DMS_filename)
            args.DMS_id = DMS_filename.replace('.csv','')
            
            # Look up taxon from reference data
            try:
                taxon_mask = DMS_reference_df['DMS_id'] == args.DMS_id
                taxon_series = DMS_reference_df[taxon_mask]['taxon']
                args.taxon = taxon_series.tolist()[0]
            except (IndexError, KeyError):
                print(f"Warning: Could not find taxon for DMS_id '{args.DMS_id}' in reference file")
                args.taxon = "unknown"
                
            if os.path.exists(get_fitness_results_path(args)):
                print(f"⏭️  Skipping {DMS_filename} (already exists)")
                continue
            
            # Try processing with error handling
            start_time = time.time()
            try:
                eval_DMS_file(model, trainer, tokenizer, args, seq_len)
                elapsed = time.time() - start_time
                print(f"✅ SUCCESS in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - start_time  
                print(f"❌ FAILED after {elapsed:.1f}s: {str(e)[:100]}...")
                # Continue with next file instead of crashing
                continue
            
    elif args.DMS_path:
        args.DMS_id = os.path.basename(args.DMS_path).replace('.csv','')
        
        # Look up taxon from reference data
        try:
            taxon_mask = DMS_reference_df['DMS_id'] == args.DMS_id
            taxon_series = DMS_reference_df[taxon_mask]['taxon']
            args.taxon = taxon_series.tolist()[0]
        except (IndexError, KeyError):
            print(f"Warning: Could not find taxon for DMS_id '{args.DMS_id}' in reference file")
            args.taxon = "unknown"
            
        if os.path.exists(get_fitness_results_path(args)):
            print(f"Skipping {args.DMS_id} because it already exists")
        else:
            eval_DMS_file(model, trainer, tokenizer, args, seq_len)
    else:
        raise ValueError("Either --DMS_filenames or --DMS_path must be provided")



