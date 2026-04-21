from __future__ import annotations

from contextlib import nullcontext
import gc
from pathlib import Path
from statistics import mean, pstdev

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import load_file
import torch

from dataset import create_dataset
from distributed_config import DistributedConfig
from eval_fsdp2 import _build_plain_runtime, _load_te_state_dict, _resolve_checkpoint_dir
from quantization import ComponentPrecisionConfig, resolve_layer_precision


RECIPE_DIR = Path(__file__).resolve().parent


def compose_eval_args(
    config_name: str = "eval_real_3B_fp8native",
    overrides: list[str] | None = None,
    *,
    artifact_root: str | Path | None = None,
):
    config_dir = str((RECIPE_DIR / "hydra_config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        args = compose(config_name=config_name, overrides=overrides or [])
    if artifact_root is not None:
        args.artifact_root = str(artifact_root)
    return args


def clone_args(base_args):
    return OmegaConf.create(OmegaConf.to_container(base_args, resolve=True))


def build_mode_args(
    base_args,
    *,
    pair_precision: str,
    linear_precision: str,
    tri_impl: str | None = None,
    hybrid_precision: dict[str, bool] | None = None,
    bf16_native_rung: str | None = None,
    mixed_tail: dict[str, object] | None = None,
):
    args = clone_args(base_args)
    args.pair_precision = pair_precision
    args.linear_precision = linear_precision
    if tri_impl is not None:
        args.component_precision.tri_impl = tri_impl
    if hybrid_precision is not None:
        args.hybrid_precision = hybrid_precision
    if bf16_native_rung is not None:
        args.bf16_native_rung = bf16_native_rung
    if mixed_tail is not None:
        args.mixed_tail = mixed_tail
    return args


def _make_checkpoint_load_state(base_args):
    block_precision = resolve_layer_precision(
        num_layers=base_args.model.num_blocks,
        fp8_enabled=base_args.fp8_config.enabled,
        fp4_enabled=base_args.fp4_config.enabled,
        fp8_layers=OmegaConf.to_container(base_args.fp8_layers, resolve=True) if base_args.fp8_layers is not None else None,
        fp4_layers=OmegaConf.to_container(base_args.fp4_layers, resolve=True) if base_args.fp4_layers is not None else None,
    )
    fp8_recipe = None
    fp4_recipe = None
    if base_args.fp8_config.enabled:
        from transformer_engine.common.recipe import Format

        fp8_recipe = hydra.utils.get_class(base_args.fp8_config.fp8_recipe)(
            fp8_format=Format[base_args.fp8_config.fp8_format], **base_args.fp8_config.fp8_recipe_kwargs
        )
    if base_args.fp4_config.enabled:
        from transformer_engine.common.recipe import Format

        fp4_recipe = hydra.utils.get_class(base_args.fp4_config.fp4_recipe)(
            fp4_format=Format[base_args.fp4_config.fp4_format], **base_args.fp4_config.fp4_recipe_kwargs
        )
    component_precision = ComponentPrecisionConfig(**OmegaConf.to_container(base_args.component_precision, resolve=True))
    return block_precision, fp8_recipe, fp4_recipe, component_precision


def load_state_dict_for_eval(base_args, dist_config: DistributedConfig, device: torch.device):
    resolved_ckpt_dir, checkpoint_info = _resolve_checkpoint_dir(base_args)
    block_precision, fp8_recipe, fp4_recipe, component_precision = _make_checkpoint_load_state(base_args)
    checkpoint_type = checkpoint_info["checkpoint_type"]
    if checkpoint_type == "safetensors":
        state_dict = load_file(str(resolved_ckpt_dir / "model.safetensors"))
    else:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", device_id=device)
        state_dict = _load_te_state_dict(
            base_args,
            dist_config,
            device,
            resolved_ckpt_dir,
            block_precision,
            fp8_recipe,
            fp4_recipe,
            component_precision,
        )
    return state_dict, checkpoint_info


def extract_dataset_sample(eval_dataset_cfg, sample_index: int):
    dataset = create_dataset(**OmegaConf.to_container(eval_dataset_cfg, resolve=True))
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} out of range for dataset of length {len(dataset)}")
    sample = dataset[sample_index]
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).contiguous()
        else:
            batch[key] = [value]
    metadata = {
        "dataset_length": len(dataset),
        "sample_index": sample_index,
        "pdb_id": sample.get("pdb_id", ""),
        "chain_id": sample.get("chain_id", ""),
        "num_residues": int(sample.get("num_residues", 0)),
    }
    return batch, metadata


def run_plain_forward(model, batch: dict[str, torch.Tensor], args, plain_infer, *, activation_probe=None, dump_activation_stats: bool = False):
    model_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    use_bf16_autocast = args.pair_precision not in (
        plain_infer.PAIR_PRECISION_FP8_EXTREME,
        plain_infer.PAIR_PRECISION_FP8_HYBRID,
        plain_infer.PAIR_PRECISION_FP8_NATIVE,
        plain_infer.PAIR_PRECISION_FP8_NATIVE_GOLD_PACKS,
        plain_infer.PAIR_PRECISION_FP8_NATIVE_MIXED_TAIL,
    )
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16_autocast else nullcontext()
    with torch.no_grad(), autocast_ctx:
        return model(
            model_inputs,
            num_recycling=args.model.get("num_recycling", 0),
            activation_probe=activation_probe,
            dump_activation_stats=dump_activation_stats,
        )


def prepare_miniformer_input(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    model_inputs = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        esm_out = model.backbone(input_ids=model_inputs["input_ids"], attention_mask=model_inputs.get("attention_mask"))
        s_s = model.fc_s_2(torch.relu(model.fc_s_1(esm_out["representations"])))
        s_z = model.fc_z_2(torch.relu(model.fc_z_1(esm_out["attentions"])))
        pair_mask = model_inputs["mask"][:, None, :] * model_inputs["mask"][:, :, None]
        residx = torch.arange(s_s.shape[1], device=s_s.device).unsqueeze(0).expand(s_s.shape[0], -1)
        s_z = torch.cat(
            [s_z, model.fold.seq_to_pair(s_s), model.fold.positional_embedding(residx, mask=pair_mask)],
            dim=-1,
        )
        s_z = model.fold.projection(s_z)
        pair_mask = pair_mask.to(s_z)
        shape = tuple(s_z.shape[:3]) + (model.fold.disto_bins,)
        dists = torch.zeros(shape, device=s_z.device, dtype=s_z.dtype)
        s_z_c = s_z + model.fold.recycle(dists)
    return s_z_c, pair_mask


def build_plain_runtime_from_args(base_args, device: torch.device, state_dict: dict[str, torch.Tensor], artifact_root: Path, status_path: Path):
    return _build_plain_runtime(base_args, device, state_dict, artifact_root, status_path)


def destroy_distributed_if_initialized() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_timings(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": mean(values),
        "std_ms": pstdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "max_ms": max(values),
    }
