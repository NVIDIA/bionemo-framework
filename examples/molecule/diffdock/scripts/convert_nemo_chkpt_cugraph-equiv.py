# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Usage: convert_nemo_chkpt_cugraph-equiv.py
# Description: Convert the DiffDock nemo checkpoint file for
# cugraph-equivariant update
#

import argparse
import os
import re
import sys
import tempfile

import torch
import torch.distributed as dist
from cugraph_equivariant.nn import FullyConnectedTensorProductConv
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import logging
from omegaconf import OmegaConf, open_dict

from bionemo.model.molecule.diffdock.infer import DiffDockModelInference
from bionemo.utils.hydra import load_model_config


# Check if this script is run in an interactive session
is_interactive = hasattr(sys, "ps1")
if is_interactive:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # initialize the process group
    dist.init_process_group("gloo", rank=0, world_size=1)

parser = argparse.ArgumentParser()
parser.add_argument("finput", type=str, help="Path to the input nemo file")
parser.add_argument("foutput", type=str, help="Path to the output nemo file")
parser.add_argument(
    "--swap_mlp_weight_blocks",
    action="store_true",
    default=False,
    help="Turn this on if the updated model calls\
                    FullyConnectedTensorProductConv with src_scalars and\
                    dst_scalars arguments rather than merging them into\
                    the edge_emb argument. This is needed because the mlp\
                    that maps the edge_emb to tp weights has a different\
                    semantic in the new vs old model.",
)
parser.add_argument("--verbose", action="store_true", default=False, help="verbose output")
parser.add_argument("--warn_extra", action="store_true", default=False, help="Emit extra warnings")
parser.add_argument(
    "--strict",
    action="store_true",
    default=False,
    help="If turned on, will require all model parameters\
                    to be populated with the input nemo file except for\
                    those known not to receive update, e.g., the tensor\
                    product backend's registered buffer",
)

args = parser.parse_args()

finput = args.finput
foutput = args.foutput
swap_mlp_weight_blocks = args.swap_mlp_weight_blocks
verbose = args.verbose
warn_extra = args.warn_extra
is_strict = args.strict

BIONEMO_HOME = os.getenv("BIONEMO_HOME")

cfg = load_model_config(config_name="infer", config_path=f"{BIONEMO_HOME}/examples/molecule/diffdock/conf")

# load the yaml config from the nemo file to determine whether
# this is the score or the confidence model
with tempfile.TemporaryDirectory() as tmpdir:
    SaveRestoreConnector._unpack_nemo_file(path2file=finput, out_folder=tmpdir)
    path_model_weights = os.path.join(tmpdir, "model_weights.ckpt")
    path_config_yaml = os.path.join(tmpdir, "model_config.yaml")
    input_dict = SaveRestoreConnector._load_state_dict_from_disk(path_model_weights)
    cfg_chkpt = OmegaConf.load(path_config_yaml)

is_confidence_model = cfg_chkpt.all_atoms
is_score_model = not is_confidence_model

with open_dict(cfg):
    # override the default nemo file path with the user input
    cfg.score_infer.restore_from_path = finput
model = DiffDockModelInference(cfg.score_infer if is_score_model else cfg.confidence_infer, interactive=is_interactive)
# The current DiffDock nemo file assumes the inner level state_dict
# to be loaded so rather than model.state_dict() or model.save_to()
# we have to do model.model.state_dict() or model.model.save_to()
output_dict = model.model.state_dict()

n_keys_updated = 0
n_keys_ignored = 0
n_keys_tp_path = 0
# regex to rename 'fc' parameters to 'mlp' ones
re_conv_mlp = re.compile(r"^(.*conv\S+)mlp(\S+)$")
# regex to rename conv.*batch_norm to the corr. individual batch_norm
re_batch_norm = re.compile(r"^(.*)batch_norm(.*)([.]\w+)$")
# regexes to match the first MLP layer weight for src<->dst block swapping
re_lig_conv_mlp_weight = re.compile(r"^(net\.)?(rec_to_)?(lig_conv_layers\.\S+)mlp\.0\.weight$")
re_rec_conv_mlp_weight = re.compile(r"^(net\.)?rec_conv_layers\.\S+mlp\.0\.weight$")
re_conv_mlp_weight = re.compile(r"^(net\.)?conv_layers\.(\d+)\.mlp\.0\.weight$")
for k, v in output_dict.items():
    k_input, modify_mlp = re.subn(re_conv_mlp, r"\1fc\2", k)
    if not modify_mlp:
        k_input, modify_batch_norm = re.subn(re_batch_norm, r"\1conv\2.batch_norm\3", k)
    if k_input in input_dict:
        if verbose:
            logging.info(f"cloning input {k_input}'s parameter into {k}")
        # this could be a conv layer if k_input results from a re.sub success
        # otherwise it is a non-conv layer
        output_dict[k] = input_dict[k_input].clone()
        n_keys_updated = n_keys_updated + 1
        if swap_mlp_weight_blocks:
            # If the key matches one of those conv layers that need mlp weight
            # swapping, proceed with the swapping. This swapping is necessary
            # to enable numerical consistency of the FullyConnectedTensorProductConv
            # output with the TensorProductConvLayer output in the cases:
            # score model's: lig_conv_layers, rec_to_lig_conv_layers, rec_conv_layers,
            # confidence model's: conv_layers[9*l + i] with 0 <= i <= 8
            is_lig_conv_mlp_weight = re.match(re_lig_conv_mlp_weight, k) is not None
            is_rec_conv_mlp_weight = re.match(re_rec_conv_mlp_weight, k) is not None
            m_conv_mlp_weight = re.match(re_conv_mlp_weight, k)
            id_conv_mlp_weight = m_conv_mlp_weight.groups()[-1] if m_conv_mlp_weight is not None else None
            is_confidence_lig_conv_mlp_weight = id_conv_mlp_weight is not None and int(id_conv_mlp_weight) % 9 <= 8
            if is_lig_conv_mlp_weight or is_rec_conv_mlp_weight or is_confidence_lig_conv_mlp_weight:
                # do additional check for the output layer's type
                m_re_conv_mlp = re.match(re_conv_mlp, k)
                if m_re_conv_mlp is None:
                    # this should never fire
                    raise RuntimeError(
                        f"Target model's parameter {k} is expected" f"to have the keyword 'conv' but it doesn't"
                    )
                else:
                    # regardless of the prefix in name `k`, m_re_conv_mlp's 2nd
                    # but last matched group (groups()[-2]) always splits into
                    # something like: [..., 'xxxconvxxx', index, '']
                    layers_index = m_re_conv_mlp.groups()[-2].split(".")
                    name_layers = layers_index[-3]
                    index = int(layers_index[-2]) if layers_index[-2] != "" else None
                    layer = model.model.net._modules[name_layers]
                    if isinstance(layer, torch.nn.modules.container.ModuleList) and index is not None:
                        # layer is a nn.Sequential so we need to retrieve the
                        # layer corr. to `index`
                        layer = layer[index]
                    if not isinstance(layer, FullyConnectedTensorProductConv):
                        raise RuntimeError(
                            f"Target model's parameter {k} comes "
                            f"from layer {name_layers}, which is "
                            f"expected to be of a "
                            f"FullyConnectedTensorProductConv but "
                            f"it's not"
                        )
                if verbose:
                    logging.info(f"swapping {k}'s weight matrix blocks")
                # swap the last two sets of model.ns-size columns
                ns = model.model.net.ns
                weight_src = output_dict[k][:, ns : 2 * ns].clone()
                weight_dst = output_dict[k][:, 2 * ns :].clone()
                output_dict[k][:, ns : 2 * ns] = weight_dst
                output_dict[k][:, 2 * ns :] = weight_src
    else:
        n_keys_ignored = n_keys_ignored + 1
        if ".tp." in k:
            n_keys_tp_path = n_keys_tp_path + 1
        elif warn_extra:
            # cugraph_equivariant store pytorch registred buffer or non-trainable
            # parameters in 'tp.*', which we don't have to worry about
            logging.warning(
                f"Target model has a parameter set named {k} but "
                f"could not find the corresponding parameter named "
                f"{k_input} in the input file: "
                f"{finput}"
            )
if is_strict:
    if n_keys_ignored != n_keys_tp_path:
        raise RuntimeError(
            "--strict mode requires number of ignored parameter"
            "update match the number of registred buffer for the"
            "tensor product backend, which is not the case: "
            f"{n_keys_ignored} vs {n_keys_tp_path}"
        )
    if n_keys_ignored + n_keys_updated != len(output_dict):
        raise RuntimeError(
            "--strict mode requires the number of updated and\
                           ignored parameters sum to the total number of \
                           parameters, which is not the case: "
            f"{n_keys_ignored} + {n_keys_updated} vs"
            f"{len(output_dict)}"
        )

model.model.load_state_dict(output_dict, strict=True)
model.model.save_to(foutput)
