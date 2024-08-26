esm2nv readme for internal use
==============================

The goal of this readme is to collect instructions useful for on-boarding
Nvidia-internal developers to the openfold project.

(1) Data preprocessing
----------------------

Extract sample training data and preprocess dataset.

Note that ESM2 pretraining involves splitting the dataset on
uniref50 but the training sequences are sampled from members of the clusters in
uniref90. Validation and testing are still performed on uniref50 level. Hence,
the model is not pretrained from a single fasta file as in ESM-1b and ESM-1v.

Alternatively, users can simply add `do_preprocessing=True` as the command line
option.

Note that ESM2 pretraining involves splitting the dataset on uniref50
but the training sequences are sampled from members of the clusters in
uniref90. Validation and testing are still performed on uniref50 level.
Hence, the model is not pretrained from a single fasta file as in ESM-1b
and ESM-1v.

``` {.bash}
# unzip sample data
cd examples/tests/test_data && \
unzip uniref202104_esm2_qc_test200_val200.zip && \
cd ${BIONEMO_HOME}

# preprocess example uniref50-90 dataset
python examples/protein/esm2nv/pretrain.py \
  --config-path=conf \
  ++do_training=False \
  ++model.data.val_size=500 \
  ++model.data.test_size=100 \
  ++model.data.train.uf50_datapath=/workspace/bionemo/examples/tests/test_data/uniref202104_esm2_qc_test200_val200/uniref50_train_filt.fasta \
  ++model.data.train.uf90_datapath=/workspace/bionemo/examples/tests/test_data/uniref202104_esm2_qc_test200_val200/ur90_ur50_sampler.fasta \
  ++model.data.train.cluster_mapping_tsv=/workspace/bionemo/examples/tests/test_data/uniref202104_esm2_qc_test200_val200/mapping.tsv \
  ++model.data.dataset_path=/workspace/bionemo/examples/tests/test_data/uniref202104_esm2_qc_test200_val200
```

Instead of the default uniref50-90 sequences, users can pretrain on custom
train, val and test sets by the following.

``` {.bash}
# preprocess example NVIDIA fasta
python examples/protein/esm2nv/pretrain.py \
  --config-path=conf \
  ++do_preprocessing=True \
  ++do_training=False \
  ++model.data.train.custom_pretraining_fasta_path=${BIONEMO_HOME}/examples/tests/test_data/protein/esm2nv/example_train.fasta \
  ++model.data.val.custom_pretraining_fasta_path=${BIONEMO_HOME}/examples/tests/test_data/protein/esm2nv/example_val.fasta \
  ++model.data.test.custom_pretraining_fasta_path=${BIONEMO_HOME}/examples/tests/test_data/protein/esm2nv/example_test.fasta \
  ++model.data.train.dataset_path=${BIONEMO_HOME}/examples/tests/test_data/protein/esm2nv/example_dataset
```

(2) Pretraining
---------------

Pretraining ESM2 on sample dataset using uniref50-90 dataset.

``` {.bash}
# training on multiple devices with tensor model parallelism
python examples/protein/esm2nv/pretrain.py \
  --config-path=conf \
  --config-name=pretrain_esm2_8M \
  ++do_preprocessing=False \
  ++do_training=True \
  ++do_testing=False \
  ++model.data.dataset_path=examples/tests/test_data/uniref202104_esm2_qc_test200_val200 \
  ++trainer.devices=2 \
  ++model.tensor_model_parallel_size=2 \
  ++model.micro_batch_size=4 \
  ++trainer.max_steps=100 \
  ++trainer.val_check_interval=10 \
  ++exp_manager.create_wandb_logger=False \
  ++exp_manager.checkpoint_callback_params.save_top_k=5

# downstream task validation will significantly slow down pretraining
# recommend to validate offline through the FLIP benchmark later
#
#   ++model.dwnstr_task_validation.enabled=True \
#   ++model.dwnstr_task_validation.dataset.dataset_path=examples/tests/test_data/protein/downstream
```

Pretraining ESM2 on custom dataset requires additional overrides.

``` {.bash}
python examples/protein/esm2nv/pretrain.py \
  --config-path=conf \
  --config-name=pretrain_esm2_8M \
  ++do_preprocessing=False \
  ++do_training=True \
  ++do_testing=False \
  ++trainer.devices=2 \
  ++model.tensor_model_parallel_size=2 \
  ++model.micro_batch_size=4 \
  ++trainer.max_steps=100 \
  ++trainer.val_check_interval=10 \
  ++exp_manager.create_wandb_logger=False \
  ++exp_manager.checkpoint_callback_params.save_top_k=5 \
  ++model.data.train.custom_pretraining_fasta_path=${BIONEMO_HOME}/examples/tests/test_data/protein/esm2nv/example_train.fasta \
  ++model.data.val.custom_pretraining_fasta_path=${BIONEMO_HOME}/examples/tests/test_data/protein/esm2nv/example_val.fasta \
  ++model.data.test.custom_pretraining_fasta_path=${BIONEMO_HOME}/examples/tests/test_data/protein/esm2nv/example_test.fasta \
  ++model.data.dataset_path=${BIONEMO_HOME}/examples/tests/test_data/protein/esm2nv/example_dataset \
  ++model.data.train.dataset_path=${BIONEMO_HOME}/examples/tests/test_data/protein/esm2nv/example_dataset
```

Known issue(s) \* OOM locally when turning on downstream task validation even on
the most conservative setting

(3) FLIP benchmark
------------------

FLIP is a collection of tasks to benchmark the expressive-ness of embeddings
from a protein language model (pLM). Christian Dallago et al. published the
[preprint](https://www.biorxiv.org/content/10.1101/2021.11.09.467890v2.abstract)
back in 2022. The goal is to identify a pLM such that a downstream model built
on top of these sequence embedding can have maximal performance.

### (3.1) Download fasta directly from FLIP public source

Different from FLIP preprocessed dataset in FLIP validation during pretraining,
we need the sequences in fasta format un-processed. We can download the raw
sequences from [public source](http://data.bioembeddings.com/public/FLIP/fasta).

``` {.bash}
DATA_PATH="${BIONEMO_HOME}/data/FLIP"
mkdir -p ${DATA_PATH}

pushd ${DATA_PATH} && \
wget http://data.bioembeddings.com/public/FLIP/fasta/all_fastas.zip && \
unzip all_fastas.zip && \
rm all_fastas.zip && \
popd
```

### (3.2) Extract ESM2 embeddings for specific FLIP task

Next, we extract the embeddings of the input sequences, specified by task, using
`bionemo/model/infer,py`.

``` {.bash}
# Choose fasta input for the specific FLIP task
# secondary_structure/sequences.fasta
# scl/mixed_soft.fasta
# conservation/sequences.fasta
# bind/sequences.fasta
# sav/mixed.fasta
# meltome/mixed_split.fasta
# gb1/two_vs_rest.fasta
# aav/seven_vs_many.fasta

FLIP_DATA_FILE=secondary_structure/sequences.fasta
TASK_NAME=$(echo $FLIP_DATA_FILE | tr "/" "\n" | head -n 1)

NEMO_PATH=esm2nv_650M_converted.nemo  # public weight from from download_models.py
DATA_PATH="${BIONEMO_HOME}/data/FLIP"
RESULT_PATH="${BIONEMO_HOME}/results"
DATA_FILE=${DATA_PATH}/${FLIP_DATA_FILE}
OUT_FILE=${RESULT_PATH}/${FLIP_DATA_FILE}

mkdir -p ${RESULT_PATH}

# inference - dump embedding in hdf5 format
# warning: not tested on multi-node environment
# warning: depend on MR: https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/merge_requests/999
python -m bionemo.model.infer \
    --config-dir ${BIONEMO_HOME}/examples/protein/esm2nv/conf \
    --config-name infer.yaml \
    trainer.num_nodes=__NUMBER_OF_NODES___ \
    trainer.devices=__NUMBER_OF_GPUS__ \
    ++model.data.output_fname="${OUT_FILE}" \
    ++model.data.dataset_path="${DATA_FILE}" \
    ++model.downstream_task.restore_from_path="${NEMO_PATH}" \
    ++model.tokenizer.model_name="facebook/esm2_t33_650M_UR50D" \
    ++trainer.precision=32 \
    ++model.data.modify_percent=0 \
    ++model.data.output_format="h5"  # new feature on MR 999
```

bionemo-service-benchmark
=========================

Before using bionemo-service-benchmark, users need to first clone the
[repo](https://gitlab-master.nvidia.com/clara-discovery/bionemo-service-benchmarking)
and pull/build the docker image locally. The benchmark will automatically train
and evaluate a downstream model. This is different from
`cfg.model.dwnstr_task_validation.enabled` in
`examples/protein/esm2nv/pretrain.py` where the pretrained weights are passed
into the benchmark.

To obtain performance metrics from the embeddings:

``` {.bash}
BIONEMO_MOUNT=""
SERVICE_BENCHMARK_MOUNT=""
TASK_NAME="secondary_structure"

# OUT_EMBED_H5=/bionemo/results/${TASK_NAME}/sequences.fasta_embeddings.h5
OUT_EMBED_H5="/bionemo/results/${TASK_NAME}/sequences.fasta_hiddens.h5"

docker run -it --name bionemo-service-benchmark --gpus all --rm \
    -v ${BIONEMO_MOUNT}:/bionemo \
    -v ${SERVICE_BENCHMARK_MOUNT}:/bionemo-service-benchmarking \
    nvcr.io/nvidian/cvai_bnmo_trng/bionemo-service-benchmarking:latest \
    bash -c """
cd /bionemo-service-benchmarking/e2e_bench/protein_embedding/
/home/root/mambaforge/envs/benchmarking/bin/python ../../run.py \
    --config-name ${TASK_NAME} \
    ++analysis_hooks=[contrib.analysis_hooks.html.HTMLTableHook] \
    ++benchmark_data.schema_data.embeddings_file=${OUT_EMBED_H5} \
    ++output.path=${OUT_EMBED_H5}.json
"""
```

Users can then compare the metrics in the output json to the golden values on
[this confluence
page](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=CLD&title=Protein+Sequence+Predictive+Benchmarks).

Additional comments: 1. Apparently secondary\_structure task is a more stable
metric compared to others due to larger amount of data.

(4) Encoder fine-tuning
-----------------------

Fine-tuning is commonly defined as "training the model on a new dataset from a
set of pretrained weights", which can be done on a downstream
regression/classification task or on a language model (LM) task on another
dataset. In BioNeMo, "fine-tuning" specifically refers to the former while the
latter is referred to as "second-stage pretraining". Since many protein, nucleic
acid and small-molecule language models leverage the NeMo/Megatron encoder model
in BioNeMo, they all inherit from the same base class `EncoderFinetuning`, which
further inherits from `ModelPT`. Note that it is not a NeMo encoder model.
Instead, it hosts the encoder model under `self.model` instantiated from the
configuration provided. This means it has the PTL logic for gradient
accumulation in constrast to the mechanism in Megatron (refer to NeMo know-how
below).

``` {.bash}
# Download nemo checkpoint
python download_artifacts.py --models esm2nv_650m --model_dir models/protein/esm2nv/

# Prepare FLIP data
python examples/protein/downstream/downstream_flip.py \
    --config-path=/workspace/bionemo/examples/protein/esm2nv/conf \
    --config-name=downstream_flip_sec_str \
    ++do_preprocessing=True \
    ++do_training=False \
    ++do_testing=False

# (Optional) reduce dataset size for quick turnaround
TASK_NAME="secondary_structure_short"
for name in train val test; do
    mkdir -p data/FLIP/${TASK_NAME}/$name
    head -n 100 data/FLIP/${TASK_NAME}/$name/x000.csv > data/FLIP/${TASK_NAME}/$name/x000.csv
done

# FLIP downstream training
ACCUMULATE_GRAD_BATCHES=2  # the loss curve should be different depending on this value

python examples/protein/downstream/downstream_flip.py \
    --config-path=/workspace/bionemo/examples/protein/esm2nv/conf \
    --config-name=downstream_flip_sec_str \
    ++trainer.max_epochs=5 \
    ++trainer.val_check_interval=10 \
    ++trainer.devices=2 \
    ++trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES} \
    ++model.data.task_name=${TASK_NAME} \
    ++exp_manager.wandb_logger_kwargs.offline=true \
    ++exp_manager.resume_if_exists=false \
    ++model.micro_batch_size=4
```

(0) NeMo know-how
-----------------

Check out [this
tutorial](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/QuickStart.md)
for an excellent summary of how Megatron Core models should be written.

### (0.1) Gradiant accumulation

Inherited from [MegatronLM](https://github.com/NVIDIA/Megatron-LM), gradient
accumulate in NeMo is done by expanding global\_batch\_size in
[get\_forward\_backward\_func](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py#L19)
wrapper.

> While this is single GPU training, the batch size specified by
> --micro-batch-size is a single forward-backward path batch-size and the code
> will perform gradient accumulation steps until it reaches global-batch-size
> which is the batch size per iteration.
>
> --- MegatronLM front page README

Therefore `accumulate_grad_batches` should always be set to 1 in vanilla NeMo.
However, unlike NeMo where the actual gradient accumulation is inferred from
global batch size, world size and parallelism, in
[BioNeMo](https://gitlab-master.nvidia.com/clara-discovery/bionemo/-/blame/dev/bionemo/model/utils.py?ref_type=heads#L118),
we infer global batch size from `accumulate_grad_batches` and the rest instead.

### (0.2) Controlling forward step

[get\_forward\_backward\_func](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py#L19)
takes a `forward_step_func` and a `loss_func` to return a parallelism-aware
wrapped forward function. Copy-pasting from the docstring directly,

            def loss_func(loss_mask, output_tensor):
                losses = output_tensor.float()
                loss_mask = loss_mask.view(-1).float()
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

                # Reduce loss for logging.
                averaged_loss = average_losses_across_data_parallel_group([loss])

                return loss, {'lm loss': averaged_loss[0]}

            def forward_step(data_iterator, model):
                data, loss_mask = next(data_iterator)
                output = model(data)
                return output, partial(loss_func, loss_mask)


            forward_backward_func(forward_step_func=forward_step, ...)

References
----------

For the ESM-2 language model and ESMFold:

``` {.bibtex}
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
