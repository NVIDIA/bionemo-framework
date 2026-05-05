# BIO-478 Torch Compile Perf Gap Addendum - 2026-04-28

## Verdict

Option (a) landed. Native BF16 + `torch.compile(default)` now matches stock BF16 + `torch.compile(default)` on the canonical synthetic trunk benchmark while preserving TOM #48 composability and FP8 compile performance.

The gap was caused by Inductor opacity around `tri_mul_xbdnn_cublas`, specifically the hidden `cat({out1_bdnn,out2_bdnn}).permute(...)` output layout boundary. The fix keeps eager xbdnn unchanged but exposes the compile path as two registered `tri_mul_bdnn_cublas` custom ops plus a visible `torch.cat`, allowing Inductor to recover the stock cat/layernorm fusion.

## Headline Benchmark Table

Fresh paired after-fix session, same hardware, same seed, 3 reps each. Values are median-of-medians.

| Configuration | Median (ms) | Peak alloc (GiB) | Peak reserved (GiB) | Delta vs ours-baseline | Graph breaks |
|---|---:|---:|---:|---:|---:|
| Stock BF16 (`bmm`) + `torch.compile(default)` | 84.4236 | 1.9711 | 3.1172 | -14.86% | 0 |
| Our BF16 native + `torch.compile(default)` - before fix | 99.1573 | 2.0414 | 3.1797 | baseline | 0 |
| Our BF16 native + `torch.compile(default)` - after fix | 84.6304 | 1.9711 | 3.1172 | -14.65% | 0 |
| Our FP8 native + `torch.compile(default)` - after fix | 232.9976 | 2.4081 | 3.2129 | n/a | 0 |

Interpretation: option (a) landed. For TOM #42 framing, the native BF16 compile path now matches stock compile at the canonical BIO-478 shape while preserving the FP8 composability result.

All after-fix benchmark reps reported `graph_count=1` and `graph_break_count=0`.

## Kernel-Level Explanation

| Condition | Launches | NCU GPU Time (ms) | Copy/Transpose Launches | Copy/Transpose Time (ms) | GEMM Launches | GEMM Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Stock BF16 compile | 930 | 117.845 | 194 | 45.963 | 681 | 52.115 |
| Native BF16 compile, before fix | 1074 | 142.989 | 338 | 70.180 | 681 | 53.017 |
| Native BF16 compile, after fix | 930 | 117.867 | 194 | 46.004 | 681 | 52.096 |

Before the fix, native BF16 had exactly 144 extra copy/transpose launches, or 3 extra launches per 48 blocks. After the fix, native BF16 has the same launch count and same copy/transpose total as stock.

## Files Changed

| File | Reason |
|---|---|
| `bionemo-recipes/recipes/esm2_minifold_te/plain_minifold_infer.py` | Adds compile-mode graph shaping for `tri_mul_xbdnn`: two visible `tri_mul_bdnn_cublas` calls plus `torch.cat`; eager remains on `tri_mul_xbdnn_cublas`. |
| `bionemo-recipes/recipes/esm2_minifold_te/tri_mul_ext/tri_mul_ext/ops.py` | Fixes fake output strides for `tri_mul_bdnn_cublas` to match the C++ permuted BNND view. |

## Validation

- Exact triangular equivalence at canonical shape: `max_abs_diff=0.0`, `torch.equal=True` in `diagnostics/followup_visible_bdnn_equivalence.json`.
- Targeted pytest: `48 passed, 1 skipped` in `logs/followup_pytest_targeted.log`.
- Post-fix Dynamo: `dynamo_graphs/followup_native_after_torch_logs.log` shows `tri_mul_bdnn_cublas_default` and visible `torch.cat` at `plain_minifold_infer.py:177`.
- Post-fix NCU: `ncu_profiles/followup_after_canonical_bf16_native_compile.ncu-rep`.
- Aggregate kernel diff: `ncu_profiles/followup_after_kernel_aggregate.json`.

Production eval rerun was attempted but blocked before evaluation by environment/model-cache issues: default 650M-vs-3B config mismatch on the first attempt, then HuggingFace 429 plus missing local 3B model weights on the corrected attempt. Existing accepted production BF16 artifact remains `/scratch/claude_tasks/accuracy_validation/artifacts/bf16_baseline_eval_metrics.json` (`lDDT=0.906798`) with the same eager `cublas_xbdnn` BF16 configuration. The new code path is compile-only and the direct triangular equivalence check is bitwise exact.

## Artifact Pointers

- Status report: `status_report.md`
- Phase A kernel diff: `kernel_diff_report.md`
- Phase B fusion analysis: `fusion_analysis.md`
- Phase C experiment log: `fusion_hint_experiments.md`
- Benchmark summary: `benchmarks/followup_after_canonical_compile_summary.json`
- NCU aggregate: `ncu_profiles/followup_after_kernel_aggregate.json`

## Next Move

BIO-478 can close as option (a) for the canonical BF16 compile gap. BIO-480 can reuse the paired FP8 after-fix baseline (`232.9976 ms`, 0 breaks, 1 graph) for FP8-specific compile-performance work.
