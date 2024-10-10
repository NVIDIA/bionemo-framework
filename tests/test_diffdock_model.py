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


import copy
import os
import re

import e3nn
import pytest
import torch
import torch_geometric
from cugraph_equivariant.nn import FullyConnectedTensorProductConv
from omegaconf import DictConfig, open_dict
from pytorch_lightning import seed_everything
from torch_geometric.loader import DataLoader

from bionemo.data.diffdock.inference import build_inference_datasets
from bionemo.model.molecule.diffdock.infer import DiffDockModelInference
from bionemo.model.molecule.diffdock.utils.diffusion import set_time
from bionemo.utils.hydra import load_model_config
from bionemo.utils.tests import check_model_exists, teardown_apex_megatron_cuda


# some pytorch functions use cuBlas or cuDNN with TF32 enabled for acceleration,
# which can potentially result in the loss of precision of the returned values.
# This in turn affect comparing pytorch results with our tp results. Turn off
# TF32 here to make sure we get precise result to compare with
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
# This works in parallel with torch.use_deterministic_algorithms
# see: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
torch.backends.cudnn.enabled = False
torch.use_deterministic_algorithms(True, warn_only=True)
e3nn.set_optimization_defaults(optimize_einsums=False)
seed_everything(seed=0, workers=True)
torch_geometric.seed_everything(0)

BIONEMO_HOME = os.getenv("BIONEMO_HOME")
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PREPEND_CONFIG_DIR = os.path.join(THIS_FILE_DIR, "./conf")
ROOT_DIR = "diffdock"

assert BIONEMO_HOME is not None, "environment variable BIONEMO_HOME is not set"
fname_checkpoint_score_model = os.path.join(BIONEMO_HOME, "models/molecule/diffdock/diffdock_score.nemo")
fname_checkpoint_confidence_model = os.path.join(BIONEMO_HOME, "models/molecule/diffdock/diffdock_confidence.nemo")


@pytest.fixture(scope="function")
def cfg(config_path_for_tests, tmp_path) -> DictConfig:
    cfg = load_model_config(config_name="diffdock_infer_test", config_path=config_path_for_tests)
    cfg.out_dir = tmp_path
    yield cfg
    teardown_apex_megatron_cuda()


@pytest.mark.needs_checkpoint
def test_model_exists():
    assert fname_checkpoint_score_model is not None
    check_model_exists(fname_checkpoint_score_model)
    assert fname_checkpoint_confidence_model is not None
    check_model_exists(fname_checkpoint_confidence_model)


dirname_test_data = os.path.join(BIONEMO_HOME, "examples/tests/test_data/molecule/diffdock/model_io")

fname_score_model_layers_io = os.path.join(dirname_test_data, "score_model_layers_io.pt")


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(fname_checkpoint_score_model)
@pytest.mark.skip_if_no_file(fname_score_model_layers_io)
def test_diffdock_score_model_layers(cfg):
    # Replace the default checkpoint nemo file path with the new one.
    # TODO: this step can be skipped after we rename the new nemo file
    # to replace the old nemo file
    with open_dict(cfg):
        cfg.score_infer.restore_from_path = fname_checkpoint_score_model
    model = DiffDockModelInference(cfg.score_infer)
    model.eval()
    model.requires_grad_()
    model.zero_grad()

    model_layers_io = torch.load(fname_score_model_layers_io)

    re_conv_layer = re.compile(r"^model\.net\.(\w*)conv(\S*)$")
    for name, layer in model.named_modules():
        if isinstance(layer, FullyConnectedTensorProductConv):
            name_batch_norm, has_batch_norm = re.subn(re_conv_layer, r"\1batch_norm\2", name)
            inputs = model_layers_io[name]
            expected = inputs["output"]
            irreps_node = inputs["irreps_node"]
            edge_indices_tpconv = inputs["edge_indices_tpconv"]
            irreps_sh = inputs["irreps_sh"]
            edge_emb = inputs["edge_emb"]
            src_scalars = inputs["src_scalars"]
            dst_scalars = inputs["dst_scalars"]
            edge_indices = edge_indices_tpconv.flip(dims=(0,))
            result = layer(
                irreps_node,
                irreps_sh,
                edge_emb,
                (edge_indices, (irreps_node.shape[0], expected.shape[0])),
                src_scalars=src_scalars,
                dst_scalars=dst_scalars,
            )
            if has_batch_norm:
                name_batch_norm_layers = name_batch_norm.split(".")
                if len(name_batch_norm_layers) > 1:
                    name_batch_norm = name_batch_norm_layers[-2]
                    index = int(name_batch_norm_layers[-1])
                    layer_batch_norm = model.model.net._modules[name_batch_norm][index]
                else:
                    layer_batch_norm = model.model.net._modules[name_batch_norm]
                result = layer_batch_norm(result)
            torch.testing.assert_close(result, expected)
            # check the gradients
            result.backward(inputs["dLdy"])
            assert layer.mlp is not None, f"score model's layer {name} doesn't have a MLP"
            torch.testing.assert_close(layer.mlp[-1].weight.grad, inputs["mlp[-1].weight.grad"])
            torch.testing.assert_close(layer.mlp[-1].bias.grad, inputs["mlp[-1].bias.grad"])


fname_score_model_io = os.path.join(dirname_test_data, "score_model_io.pt")


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(fname_checkpoint_score_model)
@pytest.mark.skip_if_no_file(fname_score_model_io)
def test_diffdock_score_model(cfg):
    # Replace the default checkpoint nemo file path with the new one.
    # TODO: this step can be skipped after we rename the new nemo file
    # to replace the old nemo file
    with open_dict(cfg):
        cfg.score_infer.restore_from_path = fname_checkpoint_score_model
    model = DiffDockModelInference(cfg.score_infer)
    model.eval()

    model_io = torch.load(fname_score_model_io)
    t_tr = model_io["t_tr"]
    t_rot = model_io["t_rot"]
    t_tor = model_io["t_tor"]
    samples_per_complex = model_io["samples_per_complex"]
    batch_size = model_io["batch_size"]
    device = torch.device("cuda")

    _, _, _, test_loader = build_inference_datasets(cfg)

    for _, orig_complex_graph in enumerate(test_loader):
        if not orig_complex_graph.success[0]:
            continue
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(samples_per_complex)]
        name = orig_complex_graph.name[0]
        loader = DataLoader(data_list, batch_size=batch_size)
        assert len(loader) == 1
        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, False, model.device)
            complex_graph_batch["ligand"].pos = model_io[name]["ligand_pos"]
            with torch.no_grad():
                tr_score, rot_score, tor_score = model.model.net(complex_graph_batch)
            torch.testing.assert_close(tr_score, model_io[name]["tr_score"], atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(rot_score, model_io[name]["rot_score"], atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(tor_score, model_io[name]["tor_score"], atol=1e-3, rtol=1e-3)


fname_score_model_grad_io = os.path.join(dirname_test_data, "score_model_grad_io.pt")


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(fname_checkpoint_score_model)
@pytest.mark.skip_if_no_file(fname_score_model_grad_io)
def test_diffdock_score_model_grad(cfg):
    # Replace the default checkpoint nemo file path with the new one.
    # TODO: this step can be skipped after we rename the new nemo file
    # to replace the old nemo file
    with open_dict(cfg):
        cfg.score_infer.restore_from_path = fname_checkpoint_score_model
    model = DiffDockModelInference(cfg.score_infer)
    model.eval()
    model.requires_grad_()

    model_grad_io = torch.load(fname_score_model_grad_io)
    batch_size = model_grad_io["batch_size"]
    t_tr = model_grad_io["t_tr"]
    t_rot = model_grad_io["t_rot"]
    t_tor = model_grad_io["t_tor"]
    device = torch.device("cuda")

    _, _, _, test_loader = build_inference_datasets(cfg)

    for _, orig_complex_graph in enumerate(test_loader):
        if not orig_complex_graph.success[0]:
            continue
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(batch_size)]
        name = orig_complex_graph.name[0]
        loader = DataLoader(data_list, batch_size=batch_size)
        assert len(loader) == 1
        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, False, model.device)
            complex_graph_batch["ligand"].pos = model_grad_io[name]["ligand_pos"]
            seed_everything(seed=0, workers=True)
            torch_geometric.seed_everything(0)
            model.zero_grad(set_to_none=True)
            tr_score, rot_score, tor_score = model.model.net(complex_graph_batch)
            torch.testing.assert_close(tr_score, model_grad_io[name]["tr_score"])
            torch.testing.assert_close(rot_score, model_grad_io[name]["rot_score"])
            torch.testing.assert_close(tor_score, model_grad_io[name]["tor_score"])
            loss = (
                ((tr_score - torch.ones_like(tr_score, device=device)) ** 2).mean()
                + ((rot_score - torch.ones_like(rot_score, device=device)) ** 2).mean()
                + ((tor_score - torch.ones_like(tor_score, device=device)) ** 2).mean()
            )
            loss.backward()
            result_dict = {}
            for k, v in model.named_parameters():
                if v.requires_grad and v.grad is not None and not torch.equal(torch.zeros_like(v.grad), v.grad):
                    result_dict[k] = v.grad.clone()

            assert (
                result_dict.keys() == model_grad_io[name]["param_grad"].keys()
            ), "result parameter names are different than the expected names"

            for k, v_grad in result_dict.items():
                torch.testing.assert_close(v_grad, model_grad_io[name]["param_grad"][k])


fname_confidence_model_layers_io = os.path.join(dirname_test_data, "confidence_model_layers_io.pt")


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(fname_checkpoint_confidence_model)
@pytest.mark.skip_if_no_file(fname_confidence_model_layers_io)
def test_diffdock_confidence_model_layers(cfg):
    # Replace the default checkpoint nemo file path with the new one.
    # TODO: this step can be skipped after we rename the new nemo file
    # to replace the old nemo file
    with open_dict(cfg):
        cfg.confidence_infer.restore_from_path = fname_checkpoint_confidence_model
    model = DiffDockModelInference(cfg.confidence_infer)
    model.eval()
    model.requires_grad_()
    model.zero_grad()

    model_layers_io = torch.load(fname_confidence_model_layers_io)

    re_conv_layer = re.compile(r"^model\.net\.(\w*)conv(\S*)$")
    for name, layer in model.named_modules():
        if isinstance(layer, FullyConnectedTensorProductConv):
            name_batch_norm, has_batch_norm = re.subn(re_conv_layer, r"\1batch_norm\2", name)
            inputs = model_layers_io[name]
            expected = inputs["output"]
            irreps_node = inputs["irreps_node"]
            edge_indices_tpconv = inputs["edge_indices_tpconv"]
            irreps_sh = inputs["irreps_sh"]
            edge_emb = inputs["edge_emb"]
            src_scalars = inputs["src_scalars"]
            dst_scalars = inputs["dst_scalars"]
            edge_indices = edge_indices_tpconv.flip(dims=(0,))
            result = layer(
                irreps_node,
                irreps_sh,
                edge_emb,
                (edge_indices, (irreps_node.shape[0], expected.shape[0])),
                src_scalars=src_scalars,
                dst_scalars=dst_scalars,
            )
            if has_batch_norm:
                name_batch_norm_layers = name_batch_norm.split(".")
                if len(name_batch_norm_layers) > 1:
                    name_batch_norm = name_batch_norm_layers[-2]
                    index = int(name_batch_norm_layers[-1])
                    layer_batch_norm = model.model.net._modules[name_batch_norm][index]
                else:
                    layer_batch_norm = model.model.net._modules[name_batch_norm]
                result = layer_batch_norm(result)
            torch.testing.assert_close(result, expected)
            # check the gradients
            result.backward(inputs["dLdy"])
            assert layer.mlp is not None, f"score model's layer {name} doesn't have a MLP"
            torch.testing.assert_close(layer.mlp[-1].weight.grad, inputs["mlp[-1].weight.grad"])
            torch.testing.assert_close(layer.mlp[-1].bias.grad, inputs["mlp[-1].bias.grad"])


fname_confidence_model_io = os.path.join(dirname_test_data, "confidence_model_io.pt")


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(fname_checkpoint_confidence_model)
@pytest.mark.skip_if_no_file(fname_confidence_model_io)
def test_diffdock_confidence_model(cfg):
    # Replace the default checkpoint nemo file path with the new one.
    # TODO: this step can be skipped after we rename the new nemo file
    # to replace the old nemo file
    with open_dict(cfg):
        cfg.confidence_infer.restore_from_path = fname_checkpoint_confidence_model
    model = DiffDockModelInference(cfg.confidence_infer)
    model.eval()

    model_io = torch.load(fname_confidence_model_io)
    t_tr = model_io["t_tr"]
    t_rot = model_io["t_rot"]
    t_tor = model_io["t_tor"]
    samples_per_complex = model_io["samples_per_complex"]
    batch_size = model_io["batch_size"]
    device = torch.device("cuda")

    _, _, confidence_test_dataset, _ = build_inference_datasets(cfg)
    test_loader = DataLoader(dataset=confidence_test_dataset, batch_size=1, shuffle=False)
    for _, orig_complex_graph in enumerate(test_loader):
        if not orig_complex_graph.success[0]:
            continue
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(samples_per_complex)]
        name = orig_complex_graph.name[0]
        loader = DataLoader(data_list, batch_size=batch_size)
        assert len(loader) == 1
        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, True, model.device)
            complex_graph_batch["ligand"].pos = model_io[name]["ligand_pos"]
            with torch.no_grad():
                confidence_score = model.model.net(complex_graph_batch)
            torch.testing.assert_close(confidence_score, model_io[name]["confidence_score"], atol=1e-3, rtol=1e-3)


fname_confidence_model_grad_io = os.path.join(dirname_test_data, "confidence_model_grad_io.pt")


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.skip_if_no_file(fname_checkpoint_confidence_model)
@pytest.mark.skip_if_no_file(fname_confidence_model_grad_io)
def test_diffdock_confidence_model_grad(cfg):
    # Replace the default checkpoint nemo file path with the new one.
    # TODO: this step can be skipped after we rename the new nemo file
    # to replace the old nemo file
    with open_dict(cfg):
        cfg.confidence_infer.restore_from_path = fname_checkpoint_confidence_model
    model = DiffDockModelInference(cfg.confidence_infer)
    model.eval()
    model.requires_grad_()

    model_grad_io = torch.load(fname_confidence_model_grad_io)
    batch_size = model_grad_io["batch_size"]
    t_tr = model_grad_io["t_tr"]
    t_rot = model_grad_io["t_rot"]
    t_tor = model_grad_io["t_tor"]
    device = torch.device("cuda")

    _, _, confidence_test_dataset, _ = build_inference_datasets(cfg)
    test_loader = DataLoader(dataset=confidence_test_dataset, batch_size=1, shuffle=False)

    for _, orig_complex_graph in enumerate(test_loader):
        if not orig_complex_graph.success[0]:
            continue
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(batch_size)]
        name = orig_complex_graph.name[0]
        loader = DataLoader(data_list, batch_size=batch_size)
        assert len(loader) == 1
        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, True, model.device)
            complex_graph_batch["ligand"].pos = model_grad_io[name]["ligand_pos"]
            seed_everything(seed=0, workers=True)
            torch_geometric.seed_everything(0)
            model.zero_grad(set_to_none=True)

            confidence_score = model.model.net(complex_graph_batch)
            torch.testing.assert_close(confidence_score, model_grad_io[name]["confidence_score"])
            target_score = torch.ones_like(confidence_score, device=device)
            loss = ((confidence_score - target_score) ** 2).mean()
            loss.backward()

            result_dict = {}
            for k, v in model.named_parameters():
                if v.requires_grad and v.grad is not None and not torch.equal(torch.zeros_like(v.grad), v.grad):
                    result_dict[k] = v.grad.clone()

            assert (
                result_dict.keys() == model_grad_io[name]["param_grad"].keys()
            ), "result parameter names are different than the expected names"

            for k, v_grad in result_dict.items():
                torch.testing.assert_close(v_grad, model_grad_io[name]["param_grad"][k])
