# Evo2 Feature Parity Test Results

Test execution date: 2026-03-03
Environment: RTX 5080 (16GB), 1 GPU, no PBSS data access

## Test Results Summary

### Tests Run in Both Containers (Feature Parity Validation)

| Test | Old Container (evo2_old_container) | New Container (evo2_image) | Match? |
|---|---|---|---|
| Training finetune (1 GPU) | PASSED (66.88s) `test_train_evo2_finetune_runs` | PASSED (86.95s) `test_fine_tuning[tp_1_pretrain]` | Yes |
| Stop-and-go (1 GPU) | PASSED (49.99s) `TestEvo2StopAndGo::test_stop_and_go_consistency` | PASSED (85.06s) `test_stop_and_go[1-1-1-False-bf16_mixed]` | Yes |
| Stop at max steps + continue | PASSED (76.77s) `test_train_evo2_stop_at_max_steps_and_continue[no_fp8]` | N/A (covered by test_fine_tuning) | N/A |
| Predict basic (1 GPU) | PASSED (49.71s) `test_predict_evo2_runs` 1 passed, 3 skipped | PASSED (94.34s) `test_predict_evo2_runs` 1 passed, 3 skipped | Yes |
| Infer basic | PASSED (110.09s) `test_run_infer` 2 passed | PASSED (62.41s) `test_infer_runs` 1 passed | Yes |

### Tests Requiring 2+ GPUs (Auto-Skipped)

| Test | Status |
|---|---|
| `test_distributed_training_gradient_equivalence` (both) | Skipped (requires 2 GPUs) |
| `test_fine_tuning[tp_2_pretrain]` (new) | Skipped (requires 2 GPUs) |
| Multi-GPU predict/infer parametrizations | Skipped (requires 2+ GPUs) |

### Tests Requiring PBSS Data Access (Not Available)

| Test | Status |
|---|---|
| `test_forward_manual` (both) | Would skip (no BIONEMO_DATA_SOURCE=pbss) |
| `test_batch_generate_coding_sequences` (both) | Would skip (no BIONEMO_DATA_SOURCE=pbss) |
| `test_predict_evo2_equivalent_with_log_probs` (both) | Would skip (no checkpoint access) |

## Conclusion

All matching tests that could run with the available hardware (1 GPU, no PBSS) **passed in both containers**, confirming feature parity for Hyena models on single-GPU configurations.

The new implementation (evo2_megatron) demonstrates equivalent behavior to the old implementation (bionemo-evo2) for:
- Training with mock data (finetune workflow)
- Stop-and-go training (checkpoint resume)
- Prediction pipeline
- Inference pipeline

## Container Details

- **Old**: `evo2_old_container:latest` - PyTorch 2.8.0, NeMo/NeMo2 based
- **New**: `evo2_image:latest` - PyTorch 2.11.0, Megatron-Bridge based
