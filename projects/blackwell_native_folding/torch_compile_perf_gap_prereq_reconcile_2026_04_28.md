# BIO-478 Prerequisite Reconciliation Addendum

Timestamp: 2026-04-27T23:25:00Z

## Verdict

FP8 + torch.compile composability is restored on the target branch. The canonical FP8 native compile path now reports graph_break_count=0 and graph_count=1, and BF16 native compile remains within the required 99 ms bound.

One guardrail note: canonical FP8 eager measured 236.90 ms, above the brief's 234 ms target. A raw-pybind control measured 236.95 ms in the same final-code environment, so the eager miss is not introduced by the wrapper port. The restored FP8 compile path is 233.09 ms and passes the stated compile-performance bound.

## Files Ported

| File | Action | Reasoning |
|---|---|---|
| minifold_native_ext/minifold_native_ext/ops.py | Added from source and adapted | Registers minifold native kernels as torch.library custom ops with fake implementations; adapted for target-only BF16/debug bindings. |
| minifold_native_ext/minifold_native_ext/__init__.py | Re-export wrapper API | Makes plain imports resolve to compile-safe wrapper functions. |
| plain_minifold_infer.py | Surgical import-path merge | Preserves target accuracy/BF16-native code while switching minifold native calls from direct _C module loading to package wrappers. |

## Sanity Gate

| Configuration | Median ms | graph_break_count | graph_count | Artifact |
|---|---:|---:|---:|---|
| Stock BF16 compile | 84.4003 | 0 | 1 | /scratch/claude_tasks/torch_compile_registration/compile_perf_gap_claude_2026_04_28/prereq_reconcile/benchmarks/canonical_bf16_stock_compile_post_port.json |
| BF16 native compile | 98.9196 | 0 | 1 | /scratch/claude_tasks/torch_compile_registration/compile_perf_gap_claude_2026_04_28/prereq_reconcile/benchmarks/canonical_bf16_native_compile_post_port.json |
| FP8 native compile pre-port | 233.50 | -1 | 0 | /scratch/claude_tasks/torch_compile_registration/compile_perf_gap_claude_2026_04_28/diagnostics/fp8_compile_graphbreak_reproducer.json |
| FP8 native compile post-port | 233.0919 | 0 | 1 | /scratch/claude_tasks/torch_compile_registration/compile_perf_gap_claude_2026_04_28/prereq_reconcile/benchmarks/canonical_fp8_native_compile_post_port.json |
| FP8 native eager post-port | 236.9037 | n/a | n/a | /scratch/claude_tasks/torch_compile_registration/compile_perf_gap_claude_2026_04_28/prereq_reconcile/benchmarks/canonical_fp8_native_eager_post_port_fastpath.json |
| FP8 raw-pybind eager control | 236.9461 | n/a | n/a | /scratch/claude_tasks/torch_compile_registration/compile_perf_gap_claude_2026_04_28/prereq_reconcile/benchmarks/canonical_fp8_native_eager_raw_pybind_control.json |

## Evidence

- Build logs: `build_minifold_native_ext.log`, `build_tri_mul_ext.log`, `build_fp8_bmm_ext.log`.
- Smoke import: `smoke_import.log`.
- Dynamo diagnostics: `diagnostics/fp8_compile_dynamo_explain_post_port.json`, `diagnostics/bf16_compile_dynamo_explain_post_port.json`, `diagnostics/stock_bf16_compile_dynamo_explain_post_port.json`.

Production-path lDDT eval was deferred because the canonical compile sanity gate was the blocker for BIO-478 Phase A and the port did not alter numerical kernels. Next step is to resume BIO-478 Phase A from this commit and rerun the perf-gap brief's opening sanity gate.
