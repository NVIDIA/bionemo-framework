import torch
from nemo.collections.nlp.modules.common.megatron.fused_layer_norm import get_layer_norm


def esm_get_layer_norm(normalized_shape, *args, **kwargs):
    use_pt_layernorm = kwargs.pop('use_pt_layernorm', False)
    if use_pt_layernorm:
        eps = kwargs.pop('eps', 1e-05)
        return torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=True)
    else:
        return get_layer_norm(normalized_shape, *args, **kwargs)
