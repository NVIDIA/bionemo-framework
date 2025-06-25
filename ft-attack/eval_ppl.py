import argparse
from typing import Literal, Optional
import numpy as np
import torch
from Bio.Seq import Seq
import torch.nn.functional as F
import csv
from Bio import SeqIO
import nemo.lightning as nl
import tempfile
from pathlib import Path
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim
from megatron.core.utils import get_batch_on_this_cp_rank

from nemo.collections.llm.gpt.model.base import get_packed_seq_params
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS, HyenaModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.data import WrappedDataLoader
from torch import Tensor
from lightning.pytorch import LightningDataModule
from Bio import Entrez
from torch.utils.data import Subset

# local imports for this script
from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset
from bionemo.llm.lightning import LightningPassthroughPredictionMixin


# Example usage:
# python direct_eval_local.py --model_name evo2_7b --fna_file /mnt/efs/boyiwei/IMG_VR/IMG_VR_2022-12-19_7.1/prokaryotic_host_sequences.fna.gz --gene_focus full

CheckpointFormats = Literal["torch_dist", "zarr"]


def pad_collate_fn(batch, tokenizer, tensor_parallel_size=1):
    """A custom collate function that pads sequences to the maximum length in a batch.
    
    Also ensures the final length is divisible by tensor_parallel_size for sequence parallelism.
    """
    # Find the maximum sequence length in the batch
    max_len = max(len(item["tokens"]) for item in batch)
    
    # Round up to the nearest multiple of tensor_parallel_size
    if tensor_parallel_size > 1:
        remainder = max_len % tensor_parallel_size
        if remainder != 0:
            max_len = max_len + (tensor_parallel_size - remainder)

    # Pad each sequence to the max_len
    for item in batch:
        num_padding = max_len - len(item["tokens"])
        if num_padding > 0:
            padding = torch.full((num_padding,), tokenizer.pad_id, dtype=item["tokens"].dtype)
            item["tokens"] = torch.cat([item["tokens"], padding])
            item["position_ids"] = torch.arange(max_len, dtype=item["position_ids"].dtype)
            mask_padding = torch.zeros(num_padding, dtype=item["loss_mask"].dtype)
            item["loss_mask"] = torch.cat([item["loss_mask"], mask_padding])

    # Default collate can now handle the batch because all items have the same length
    return torch.utils.data.default_collate(batch)


def _gather_along_cp_dim(input_, seq_dim: int = 1):
    """Gather tensors and concatenate along the last dimension."""
    world_size = parallel_state.get_context_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=parallel_state.get_tensor_model_parallel_group()
    )
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=seq_dim).contiguous()

    return output


class HyenaPredictor(LightningPassthroughPredictionMixin, HyenaModel):
    """A predictor for the Hyena model. This adds in the predict step and the passthrough method."""

    def __init__(
        self,
        *args,
        output_log_prob_seqs: bool = False,
        log_prob_collapse_option: Literal["sum", "mean"] = "mean",
        **kwargs,
    ):
        """Initialize the predictor with our needs around computing log probabilities."""
        super().__init__(*args, **kwargs)
        self.output_log_prob_seqs = output_log_prob_seqs
        self.log_prob_collapse_option = log_prob_collapse_option

    def predict_step(self, batch, batch_idx: Optional[int] = None) -> Optional[dict[str, Tensor]]:
        """Alias for forward_step, also log the pad mask since sequences may not all have the same length."""
        if len(batch) == 0:
            return None
        forward_out = self.forward_step(batch)
        if not isinstance(forward_out, Tensor):
            return forward_out
        # Reminder: the model's predictions for input i land at output i+1. To get everything to align, we prepend the
        # EOS token to the input sequences and take the outputs for all but the first token.
        forward_out_tp_gathered = _gather_along_last_dim(
            forward_out, group=parallel_state.get_tensor_model_parallel_group()
        )
        # else:
        #     forward_out_tp_gathered = _collect_into_dim(forward_out, dim=-1)
        forward_out_gathered = _gather_along_cp_dim(forward_out_tp_gathered)
        assert self.tokenizer.vocab_size == forward_out_gathered.shape[-1]  # type: ignore
        if self.output_log_prob_seqs:
            softmax_logprobs = torch.log_softmax(forward_out_gathered, dim=-1)
            softmax_logprobs = softmax_logprobs[:, :-1]
            input_ids = batch["tokens"][:, 1:]
            assert softmax_logprobs.shape[1] == input_ids.shape[1]

            logprobs = torch.gather(
                softmax_logprobs,  # Gather likelihoods...
                2,  # along the vocab dimension...
                input_ids.unsqueeze(-1),  # using the token ids to index.
            ).squeeze(-1)
            log_prob_seqs = torch.sum(logprobs * batch["loss_mask"][:, 1:].float(), dim=-1)
            if self.log_prob_collapse_option == "mean":
                log_prob_seqs = log_prob_seqs / (batch["loss_mask"][:, 1:].float().sum(dim=-1) + 1e-8)
            return {"log_probs_seqs": log_prob_seqs.cpu(), "seq_idx": batch["seq_idx"].cpu()}
        else:
            # If the user wants to match back to logits, then they will need to do the offsetting logic themselves.
            return {
                "token_logits": forward_out_gathered.cpu(), # return the average of the log probs across the entire sequence.
                "pad_mask": batch["loss_mask"].cpu(),
                "seq_idx": batch["seq_idx"].cpu(),
            }


def hyena_predict_forward_step(model, batch) -> torch.Tensor:
    """Performs a forward step for the Hyena model.

    Args:
        model: The Hyena model
        batch: Dictionary containing input batch data with keys:
            - tokens: Input token IDs
            - position_ids: Position IDs
            - labels: Labels for loss computation
            - loss_mask: Mask for loss computation

    Returns:
        torch.Tensor: Output from the model forward pass
    """
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        # "labels": batch["labels"],
        # "loss_mask": batch["loss_mask"],
    }

    forward_args["attention_mask"] = None
    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)
    return model(**forward_args)


def hyena_predict_data_step(dataloader_iter) -> dict[str, torch.Tensor]:
    """Data step for the Hyena model prediction. Modified from the original gpt data step to include the seq_idx."""
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch  # type: ignore

    required_device_keys = set()
    required_host_keys = set()

    required_device_keys.add("attention_mask")
    if "cu_seqlens" in _batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    if parallel_state.is_pipeline_first_stage():
        required_device_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask", "seq_idx"))

    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_cp_rank(_batch_required_keys)

    return output


class PredictDataModule(LightningDataModule):
    """Create a dataloader for prediction."""

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = 1, tensor_parallel_size: int = 1):
        """Create a dataloader for prediction."""
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the dataloader."""
        self.tokenizer = get_nmt_tokenizer("byte-level")

    def predict_dataloader(self):
        """Create a dataloader for prediction."""
        # need to use this to communicate that we are in predict mode and safe to not drop last batch
        return WrappedDataLoader(
            mode="predict",
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda batch: pad_collate_fn(batch, self.tokenizer, self.tensor_parallel_size),
        )


def compute_perplexity(model, record, gene_focus="env"):
    """This function will be replaced by the new workflow but is kept for reference during refactoring."""
    raise NotImplementedError("This function is replaced by the new bionemo-framework-based workflow.")


def extract_gene_from_record(record, gene_name="env"):
    """
    Extract the specified gene/CDS from a GenBank record.
    
    Args:
        record: A BioPython SeqRecord from a GenBank file
        gene_name: The name of the gene to extract (e.g., env, gag, pol)
        
    Returns:
        str: The DNA sequence of the specified gene, or None if not found
    """
    if gene_name.lower() == "full":
        # Return the full genome
        return str(record.seq)
    
    gene_name = gene_name.lower()
    
    # Function to check if a feature matches our target gene
    def is_matching_gene(feature, name):
        # Check gene qualifier
        gene = feature.qualifiers.get("gene", [""])[0].lower()
        if gene == name:
            return True
        
        # Check product qualifier
        product = feature.qualifiers.get("product", [""])[0].lower()
        if name in product:
            return True
        
        # Check note qualifier
        note = feature.qualifiers.get("note", [""])[0].lower() if "note" in feature.qualifiers else ""
        if name in note:
            return True
        
        # Special cases for HIV genes
        if name == "env" and any(x in product for x in ["gp160", "gp120", "envelope"]):
            return True
        if name == "gag" and "polyprotein" in product and "gag" in product:
            return True
        if name == "pol" and "polymerase" in product:
            return True
        
        return False
    
    # Method 1: Look for CDS with matching gene name
    for feature in record.features:
        if feature.type == "CDS" and is_matching_gene(feature, gene_name):
            gene_sequence = str(feature.location.extract(record.seq))
            print(f"Found {gene_name} gene: {feature.qualifiers.get('gene', ['Unknown'])[0]}, "
                  f"product: {feature.qualifiers.get('product', ['Unknown'])[0]}")
            return gene_sequence
    
    # Method 2: Look for any feature with the gene name in qualifiers
    for feature in record.features:
        if gene_name in str(feature.qualifiers).lower():
            gene_sequence = str(feature.location.extract(record.seq))
            print(f"Found feature related to {gene_name}: {feature.type}")
            return gene_sequence
    
    print(f"Could not find {gene_name} gene in the record.")
    return None


def main():
    """
    Evaluate the perplexity of a sequence using an Evo2 model.
    """
    parser = argparse.ArgumentParser(
        description="""Evaluate the perplexity of a sequence from a local GenBank file."""
    )
    # --- Old args ---
    # parser.add_argument("--ckpt_dir", default='/workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k',
    #                     help="Model to use for perplexity calculation.")
    # parser.add_argument("--gb_file_list", type=str, default=None,
    #                    help="Path to a text file containing a list of GenBank file paths (one per line).")
    # parser.add_argument("--fna_file", type=str, default=None,
    #                    help="Path to a (possibly gz-compressed) FASTA/FA/FNA file. If provided, the first 100 sequences in this file will be evaluated.")
    # parser.add_argument("--gene_focus", type=str, default="env",
    #                    help="""Specify which gene to extract from the sequence (default: env).
    #                    Options include: env, gag, pol, or 'full' for the entire genome.""")
    # parser.add_argument("--num_seqs_fna", type=int, default=100,
    #                    help="Number of sequences to evaluate from the FASTA file.")
    # --- New args (from predict.py) ---
    parser.add_argument("--fasta", type=Path, help="Fasta path from which to generate logit predictions.")
    parser.add_argument("--gb_file_list", type=str, help="Path to a text file with a list of GenBank file paths.")
    parser.add_argument("--ckpt-dir", type=Path, default="/workspaces/bionemo-framework/checkpoints/nemo2_evo2_7b_8k", help="NeMo2 checkpoint directory for inference.")
    parser.add_argument("--prepend-bos", action="store_true", help="Prepend BOS token to sequences. Defaults to False.")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--pipeline-model-parallel-size", type=int, default=1, help="Order of pipeline parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--no-sequence-parallel",
        action="store_true",
        help="When using TP, skip sequence parallelism. Otherwise sequence parallelism is used whenever tensor "
        "parallelism is used. sequence parallelism should save a small amount of GPU memory so it's on"
        " by default.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for prediction. Defaults to 1.")
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
    parser.add_argument(
        "--gene_focus",
        type=str,
        default="full",
        help="""Specify which gene to extract from the sequence (default: full). Used only with --gb_file_list.
                       Options include: env, gag, pol, or 'full' for the entire genome.""",
    )
    parser.add_argument("--num_seqs_fna", type=int, default=100, help="Number of sequences to evaluate from FASTA.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory to save the output.")

    args = parser.parse_args()

    if not args.fasta and not args.gb_file_list:
        raise ValueError("Either --fasta or --gb_file_list must be provided.")

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    tmp_dir = tempfile.TemporaryDirectory()
    fasta_path_for_predict = args.fasta

    if args.gb_file_list:
        raise NotImplementedError("This part is work in progress.")
        print(f"Processing GenBank files from {args.gb_file_list}...")
        fasta_path_for_predict = Path(tmp_dir.name) / "sequences.fasta"
        accessions = []
        with open(args.gb_file_list, "r", newline="") as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                # Try to locate the column automatically
                try:
                    idx = header.index("accession_id")
                except ValueError:
                    # Default to the last column if header not found / no header
                    idx = -1
                    # If the first row actually contained data rather than header, include it
                    if header and header[0].strip():
                        accessions.append(header[idx].strip())

                for row in reader:
                    if len(row) > abs(idx) and row[idx].strip():
                        accessions.append(row[idx].strip())
        with open(fasta_path_for_predict, "w") as fasta_handle:
            for acc in accessions:
                print(f"\nFetching record for: {acc}")
                try:
                    handle = Entrez.efetch(db="nucleotide", id=acc, rettype="gb", retmode="text")
                    record = SeqIO.read(handle, "genbank")
                    handle.close()
                    fasta_handle.write(f">{record.id}__{args.gene_focus}\n{record.seq}\n")
                except Exception as e:
                    print(f"Error fetching {acc}: {e}")
                    continue
        

    # --- Setup Trainer and Model from predict.py ---
    sequence_parallel = args.tensor_parallel_size > 1 and not args.no_sequence_parallel
    model_parallel_size = (
        args.tensor_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    if model_parallel_size > torch.cuda.device_count():
        raise ValueError(
            f"Requested model parallel size {model_parallel_size} is greater than the "
            f"number of available CUDA devices {torch.cuda.device_count()}"
        )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=model_parallel_size,
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
                seq_len=8192,
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
        distribute_saved_activations=False if sequence_parallel and args.tensor_parallel_size > 1 else True,
    )
    trainer.strategy._setup_optimizers = False  # type: ignore

    resume = nl.AutoResume(
        resume_if_exists=True,
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
        log_prob_collapse_option="mean",
    )
    resume.setup(trainer, model)

    full_dataset = SimpleFastaDataset(fasta_path_for_predict, tokenizer, prepend_bos=args.prepend_bos)
    if args.fasta and args.num_seqs_fna > 0 and len(full_dataset) > args.num_seqs_fna:
        print(f"Selecting the first {args.num_seqs_fna} sequences from the dataset.")
        dataset_to_use = Subset(full_dataset, range(args.num_seqs_fna))
    else:
        dataset_to_use = full_dataset

    print(f"Using {len(dataset_to_use)} sequences from {fasta_path_for_predict}")
    datamodule = PredictDataModule(dataset_to_use, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)

    print(f"\nComputing perplexity for {len(dataset_to_use)} sequences...")
    results = trainer.predict(model, datamodule=datamodule)

    # --- Process results and plot ---
    if trainer.is_global_zero:
        if results:
            # The trainer returns a flat list of dictionaries, one for each sample.
            log_probs = torch.cat([r["log_probs_seqs"] for r in results])  # type: ignore
            seq_indices = torch.cat([r["seq_idx"] for r in results])  # type: ignore

            perplexities = torch.exp(-log_probs)

            # Sort by original sequence index
            sorted_indices = torch.argsort(seq_indices)
            sorted_perplexities = perplexities[sorted_indices].cpu().numpy()

            # Access the underlying dataset, whether it's the full one or a subset
            source_dataset = dataset_to_use.dataset if isinstance(dataset_to_use, Subset) else dataset_to_use
            assert isinstance(source_dataset, SimpleFastaDataset)
            original_indices = dataset_to_use.indices if isinstance(dataset_to_use, Subset) else list(range(len(dataset_to_use)))
            
            # Map the indices from the results back to the original fasta file indices
            result_original_indices = [original_indices[int(i.item())] for i in seq_indices[sorted_indices]]
            
            sorted_headers = [source_dataset.seqids[i] for i in result_original_indices]

            perplexity_results = [
                {"file": header, "perplexity": ppl} for header, ppl in zip(sorted_headers, sorted_perplexities)
            ]

            values = [item["perplexity"] for item in perplexity_results]
            print("\nPerplexity results:")
            for item in perplexity_results:
                print(f"{item['file']}: {item['perplexity']:.4f}")

            # Create violin plot
            if args.gb_file_list:
                list_file_name = Path(args.gb_file_list).stem
            else:
                list_file_name = Path(args.fasta).stem + "_" + args.ckpt_dir.stem
            try:
                import matplotlib.pyplot as plt

                # Create figure
                fig, ax = plt.subplots(figsize=(8, 6))

                # Create violin plot
                parts = ax.violinplot([values], positions=[1], showmeans=True, showextrema=True)

                # Customize violin appearance
                if 'bodies' in parts:
                    for pc in parts['bodies']:
                        pc.set_facecolor('lightblue')
                        pc.set_alpha(0.7)

                # Add individual points (using np which is already imported at the top)
                x_jitter = np.random.normal(1, 0.04, size=len(values))
                ax.scatter(x_jitter, values, alpha=0.6, s=20, color='darkblue')

                # Set labels and title
                ax.set_ylabel('Perplexity', fontsize=12)
                ax.set_title(
                    f'Perplexity Distribution (Gene Focus: {args.gene_focus} File: {list_file_name})',
                    fontsize=14,
                    pad=20,
                )

                # Set y-axis limits
                # y_min, y_max = 1.5, 4.0
                # ax.set_ylim(y_min, y_max)

                # Remove x-axis ticks
                ax.set_xticks([])

                # Add grid for better readability
                ax.grid(axis='y', alpha=0.3)

                # Save the plot
                file_name = f"{args.output_dir}/ppl_violin_plot_{list_file_name}.pdf"
                plt.tight_layout()
                plt.savefig(file_name, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Violin plot saved to {file_name}")
            except ImportError:
                print("\nMatplotlib is not installed. Skipping plot generation. `pip install matplotlib`")

        else:
            print("No valid perplexity values were computed. No plot generated.")

    tmp_dir.cleanup()


if __name__ == "__main__":
    main()
