# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monkey-patch FusedAdam.step() to support MXFP8 block-scaling QuantizedTensor parameters.

Based on NVIDIA/TransformerEngine PR #2753 (merged to main, commit fcceeb96, March 13 2026):
  https://github.com/NVIDIA/TransformerEngine/pull/2753

This patch adds a QuantizedTensor code path that routes block-scaling quantized parameters
(MXFP8Tensor, Float8BlockwiseQTensor, NVFP4Tensor) through the FP32 Adam kernel on master
weights, then writes back via quantize_() after the optimizer step.

Required because TE <= v2.13 only handles Float8Tensor (delayed scaling) in FusedAdam.step().
MXFP8Tensor falls through to the BF16 path where p.data dereferences a wrapper_subclass with
data_ptr()==0, causing Xid 31 (GPU memory page fault).

Remove this patch once the TE version includes PR #2753 (check for QuantizedTensor in
FusedAdam.step source).
"""

import logging


logger = logging.getLogger(__name__)


def apply_patch():
    """Monkey-patch FusedAdam.step() to handle QuantizedTensor (MXFP8, Float8Blockwise)."""
    import inspect

    from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam

    # Check if already patched or fix is upstream
    src = inspect.getsource(FusedAdam.step)
    if "QuantizedTensor" in src:
        logger.info("FusedAdam already supports QuantizedTensor — patch not needed.")
        return False

    import torch
    from torch.distributed.tensor import DTensor
    from transformer_engine.pytorch.tensor import QuantizedTensor
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    _original_step = FusedAdam.step

    @torch.no_grad()
    def _patched_step(self, closure=None, grad_scaler=None):
        """FusedAdam.step() with QuantizedTensor support (PR #2753 backport)."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if len(group["params"]) == 0:
                continue

            device = group["params"][0].device
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]

            # Step counter
            if "step" in group:
                if not self.capturable:
                    group["step"] += 1
                else:
                    group["step"] += torch.tensor([1], dtype=torch.int, device=device)
            else:
                group["step"] = 1 if not self.capturable else torch.tensor([1], dtype=torch.int, device=device)

            # create lists for multi-tensor apply
            import transformer_engine_torch as tex
            from transformer_engine.pytorch.optimizers.fused_adam import get_fp8_meta
            from transformer_engine.pytorch.optimizers.multi_tensor_apply import multi_tensor_applier

            p_main_of_fp8_model = []
            p_main_of_f16_model = []
            g_of_fp8_model = []
            g_of_f16_model = []
            g_of_f32_model = []
            m_of_fp8_model = []
            m_of_f16_model = []
            m_of_f32_model = []
            v_of_fp8_model = []
            v_of_f16_model = []
            v_of_f32_model = []
            p_fp8_model = []
            p_f16_model = []
            p_f32_model = []
            scales = []
            amaxes = []
            scale_invs = []

            unscaled_lists = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}
            scaled_lists = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}
            state_scales = {"exp_avg": [], "exp_avg_sq": [], "master_param": []}

            out_dtype = tex.DType.kFloat32

            has_fp16 = False
            has_bf16 = False

            # --- PR #2753 addition: track quantized params for writeback ---
            quantized_params_to_update = []

            for p in group["params"]:
                state = self.state[p]

                store_param_remainders = self.store_param_remainders and p.dtype == torch.bfloat16

                if len(state) == 0:
                    self.initialize_state(p, store_param_remainders)

                if self.use_decoupled_grad:
                    p_grad = p.decoupled_grad if hasattr(p, "decoupled_grad") else None
                else:
                    p_grad = p.grad

                if p_grad is None:
                    continue
                if p_grad.data.is_sparse:
                    raise RuntimeError("FusedAdam does not support sparse gradients.")

                # Unscaling
                unscaled_state = {}
                for name in ["exp_avg", "exp_avg_sq", "master_param"]:
                    if name in state:
                        if name == "master_param" and store_param_remainders:
                            unscaled_state[name] = self.state[p][name]
                            assert unscaled_state[name].dtype == torch.int16
                        else:
                            unscaled = self.get_unscaled_state(p, name)
                            unscaled_state[name] = unscaled
                        if self.name_to_dtype_map[name] != torch.float32:
                            unscaled_lists[name].append(unscaled)
                            scaled_lists[name].append(state[name])
                            state_scales[name].append(self._scales[p][name])

                if isinstance(p, Float8Tensor):
                    out_dtype = p._fp8_dtype
                    p_fp8_model.append(p._data.data)
                    scale, amax, scale_inv = get_fp8_meta(p)
                    scales.append(scale)
                    amaxes.append(amax)
                    scale_invs.append(scale_inv)
                    if self.master_weights:
                        p_main_of_fp8_model.append(unscaled_state["master_param"].data)
                    g_of_fp8_model.append(p_grad.data)
                    m_of_fp8_model.append(unscaled_state["exp_avg"])
                    v_of_fp8_model.append(unscaled_state["exp_avg_sq"])
                # --- PR #2753: QuantizedTensor branch (MXFP8, Float8Blockwise, NVFP4) ---
                elif isinstance(p, QuantizedTensor) or (
                    isinstance(p, DTensor) and isinstance(p._local_tensor, QuantizedTensor)
                ):
                    if not self.master_weights:
                        local_p = p._local_tensor if isinstance(p, DTensor) else p
                        raise RuntimeError(
                            "FusedAdam without master_weights does not support "
                            f"{type(local_p).__name__} parameters. Use master_weights=True."
                        )
                    if self.capturable:
                        raise RuntimeError(
                            "FusedAdam does not support block-scaling quantized weights with capturable=True."
                        )
                    # Route to FP32 master-weight path. Adam updates the FP32 master,
                    # then we write back to the quantized param after kernels run.
                    # NOTE: Adam calculation and quantization are unfused for block-scaling
                    # parameters. Fusion needs to be done later.
                    p_f32_model.append(unscaled_state["master_param"].data)
                    g_of_f32_model.append(p_grad.data.float())
                    m_of_f32_model.append(unscaled_state["exp_avg"])
                    v_of_f32_model.append(unscaled_state["exp_avg_sq"])
                    quantized_params_to_update.append((p, unscaled_state["master_param"]))
                elif p.dtype in [torch.float16, torch.bfloat16]:
                    has_fp16 = has_fp16 or p.dtype == torch.float16
                    has_bf16 = has_bf16 or p.dtype == torch.bfloat16
                    p_f16_model.append(p.data)
                    if self.master_weights:
                        p_main_of_f16_model.append(unscaled_state["master_param"].data)
                    g_of_f16_model.append(p_grad.data)
                    m_of_f16_model.append(unscaled_state["exp_avg"])
                    v_of_f16_model.append(unscaled_state["exp_avg_sq"])
                elif p.dtype == torch.float32:
                    p_f32_model.append(p.data)
                    g_of_f32_model.append(p_grad.data)
                    m_of_f32_model.append(unscaled_state["exp_avg"])
                    v_of_f32_model.append(unscaled_state["exp_avg_sq"])
                else:
                    raise RuntimeError("FusedAdam only support model weights in fp32, fp16, bf16 and fp8")

                if self.capturable and len(p_fp8_model) > 0:
                    raise RuntimeError("FusedAdam does not support FP8 model weights with capturable=True.")

                if has_fp16 and has_bf16:
                    if self.store_param_remainders:
                        raise RuntimeError(
                            "FusedAdam doesn't support a mix of FP16/BF16 weights + Store param remainder."
                        )
                    raise RuntimeError("FusedAdam does not support a mix of float16 and bfloat16 model weights.")

            def apply_multi_tensor_adam(adam_func, tensor_lists, inv_scale=None, out_dtype=None):
                inv_scale_arg = () if inv_scale is None else (inv_scale,)
                out_dtype_arg = () if out_dtype is None else (out_dtype,)
                multi_tensor_applier(
                    adam_func,
                    self._dummy_overflow_buf,
                    tensor_lists,
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    self.adam_w_mode,
                    bias_correction,
                    group["weight_decay"],
                    *inv_scale_arg,
                    *out_dtype_arg,
                )

            if self.capturable:
                found_inf = (
                    grad_scaler._check_inf_per_device(self)[device]
                    if grad_scaler is not None
                    else torch.zeros((1,), device=device)
                )
                self._dummy_overflow_buf.copy_(found_inf)

                scale, inv_scale = None, None
                if grad_scaler:
                    scale = grad_scaler._get_scale_async()
                    inv_scale = scale.double().reciprocal().float()
                else:
                    inv_scale = torch.ones((1,), dtype=torch.float32, device=device)

                if self.master_weights:
                    if len(p_fp8_model) > 0:
                        tensor_lists = [
                            g_of_fp8_model,
                            p_fp8_model,
                            m_of_fp8_model,
                            v_of_fp8_model,
                            p_main_of_fp8_model,
                            scales,
                            amaxes,
                            scale_invs,
                        ]
                        apply_multi_tensor_adam(
                            self.multi_tensor_adam_capturable_fp8, tensor_lists, inv_scale, out_dtype
                        )
                    if len(p_f16_model) > 0:
                        tensor_lists = [
                            g_of_f16_model,
                            p_f16_model,
                            m_of_f16_model,
                            v_of_f16_model,
                            p_main_of_f16_model,
                        ]
                        apply_multi_tensor_adam(self.multi_tensor_adam_capturable_master, tensor_lists, inv_scale)
                    if len(p_f32_model) > 0:
                        tensor_lists = [
                            g_of_f32_model,
                            p_f32_model,
                            m_of_f32_model,
                            v_of_f32_model,
                        ]
                        apply_multi_tensor_adam(self.multi_tensor_adam_capturable, tensor_lists, inv_scale)
                else:
                    if len(p_f16_model) > 0:
                        tensor_lists = [g_of_f16_model, p_f16_model, m_of_f16_model, v_of_f16_model]
                        apply_multi_tensor_adam(self.multi_tensor_adam_capturable, tensor_lists, inv_scale)
                    if len(p_f32_model) > 0:
                        tensor_lists = [g_of_f32_model, p_f32_model, m_of_f32_model, v_of_f32_model]
                        apply_multi_tensor_adam(self.multi_tensor_adam_capturable, tensor_lists, inv_scale)

            elif self.master_weights:
                if len(p_f16_model) > 0:
                    tensor_lists = [
                        g_of_f16_model,
                        p_f16_model,
                        m_of_f16_model,
                        v_of_f16_model,
                        p_main_of_f16_model,
                    ]
                    if self.store_param_remainders and has_bf16 and not has_fp16:
                        apply_multi_tensor_adam(self.multi_tensor_adam_param_remainder, tensor_lists)
                    else:
                        apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
                if len(p_fp8_model) > 0:
                    tensor_lists = [
                        g_of_fp8_model,
                        p_fp8_model,
                        m_of_fp8_model,
                        v_of_fp8_model,
                        p_main_of_fp8_model,
                        scales,
                        amaxes,
                        scale_invs,
                    ]
                    apply_multi_tensor_adam(self.multi_tensor_adam_fp8, tensor_lists, out_dtype)
                if len(p_f32_model) > 0:
                    tensor_lists = [
                        g_of_f32_model,
                        p_f32_model,
                        m_of_f32_model,
                        v_of_f32_model,
                    ]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
            else:
                if len(p_f16_model) > 0:
                    tensor_lists = [g_of_f16_model, p_f16_model, m_of_f16_model, v_of_f16_model]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)
                if len(p_f32_model) > 0:
                    tensor_lists = [g_of_f32_model, p_f32_model, m_of_f32_model, v_of_f32_model]
                    apply_multi_tensor_adam(self.multi_tensor_adam, tensor_lists)

            # --- PR #2753: Write updated FP32 master weights back to quantized params ---
            for p, master_w in quantized_params_to_update:
                local_p = p._local_tensor if isinstance(p, DTensor) else p
                local_p.quantize_(master_w.data)

            # Scaling
            for name in ["exp_avg", "exp_avg_sq", "master_param"]:
                if len(unscaled_lists[name]) > 0:
                    for unscaled, scaled, scale in zip(unscaled_lists[name], scaled_lists[name], state_scales[name]):
                        self._apply_scale(name, unscaled, scaled, scale)

            del unscaled_lists

        return loss

    FusedAdam.step = _patched_step
    logger.info("Applied MXFP8 QuantizedTensor patch to FusedAdam.step() (PR #2753 backport)")
    return True
