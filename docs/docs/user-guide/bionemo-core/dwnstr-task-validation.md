# Validation with a Downstream Task

## Overview

Validation with a downstream task allows the performance of model embeddings to be assessed on a downstream task of choice during the validation step of model training. To enable this functionality, labeled train and test datasets for a downstream task must be provided. A lightweight predictive ML model is trained and tested on provided datasets at the end of every validation epoch. Loss and metrics for the task are added to the Weights and Biases dashboard.

Validation with a downstream task is useful for monitoring model training progress with metrics that are meaningful for the intended model purpose, such as secondary structure prediction accuracy for ProtT5nv and ESM-1nv or physchem property prediction error for MegaMolBART.

## Input Data Format

Downstream task validation requires labeled training and test dataset for the downstream task of choice. Files are expected to be in the `.csv` format. The content of the files depends on the type of a downstream task.

## Supported Downstream Tasks

BioNeMo framework supports a wide range of tasks for validation, and these tasks can be customized as desired. Supported tasks can be broken down into the following categories:

### Sequence-Level Regression

Prediction of a single continuous value from sequence embeddings. The downstream task dataset should contain one continuous value per input sequence (protein sequence or SMILES string). An example task is a physchem property prediction such as lipophilicity. Below is the sample dataset for lipophilicity. The `smiles` column provides the input data (SMILES strings) and `exp` provides the experimentally determined octanol/water distribution coefficient.

```bash
CMPD_CHEMBLID,exp,smiles
CHEMBL596271,3.54,Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14
CHEMBL1951080,-1.18,COc1cc(OC)c(cc1NC(=O)CSCC(=O)O)S(=O)(=O)N2C(C)CCc3ccccc23
CHEMBL1771,3.69,COC(=O)[C@@H](N1CCc2sccc2C1)c3ccccc3Cl
CHEMBL234951,3.37,OC[C@H](O)CN1C(=O)C(Cc2ccccc12)NC(=O)c3cc4cc(Cl)sc4[nH]3
```

### Sequence-Level Classification

Prediction of a class label from sequence embeddings can be binary or multi-class classification. The downstream task datasets should contain one label per input sequence (protein sequence or SMILES string). An example task is prediction of protein subcellular localization. Below is the sample dataset for subcellular localization. `sequence` column supplies protein amino acid sequence, and `scl_label` column supplies class label of the protein's subcellular localization in text format. This text labels are converted into numeric class labels as the dataset object is created.

```bash
id,sequence,scl_label
Sequence0,MGCMKSKQTFPFPTIYEGEKQHESEEPFMPEERCLPRMASPVNVKEEVKEPPGTNTVILEYAHRLSQDILCDALQQWACNNIKYHDIPYIESEGP,Cell_membrane
Sequence1,MFTLKKSQLLLFFPGTINLSLCQDETNAEEERRDEEVAKMEEIKRGLLSGILGAGKHIVCGLSGLC,Extracellular
Sequence2,MSVPIDRINTVDTLANTLESNLKFFPEKIHFNIRGKLIVIDRVELPILPTTVLALLFPNGFHLLLKHEGGDVSSCFKCHKIDPSICKWILDTYLPLFKK,Cytoplasm
Sequence13,MLFLKLVASVLALMTIVPAQAGLIGKRKPKVMIINMFSLEANAWLSQMDDLYANNITVVGLNRLYPQVHCNTQQTICQMTTGEGKSNAAS,Endoplasmic_reticulum
```

### Token-Level Classification

Prediction of a class label for each token in the input sequence from sequence hidden states. Supports multiple classification heads, each can be binary or multi-class classification task. Additionally supports masking some token positions. Masked positions are excluded from loss and accuracy calculation both for training and testing. The downstream task dataset should contain a class label for each token in the input sequence. Optional masks of 0s and 1s can be provided. Example task is secondary structure prediction. Below is the sample dataset for secondary structure prediction. `sequence` column supplies protein amino acid sequence, `3state` column supplies secondary structure class labels for each amino acid in the protein sequence (`C` -- coil, `H` -- helix, `E` -- sheet), and the `resolved` column provides info if the residue at this position is resolved, that will be used as mask.

```bash
id,sequence,3state,resolved
1es5-A,VTKPTIAAVGGYAMNNGTGTTLYTKAADTRRSTGSTTKIMTAKVVLAQSNLNLDAKVTIQKAYSDYVVANNASQAHLIVGDKVTVRQLLYGLMLPSGCDAAYALADKYGSGSTRAARVKSFIGKMNTAATNLGLHNTHFDSFDGIGNGANYSTPRDLTKIASSAMKNSTFRTVVKTKAYTAKTVTKTGSIRTMDTWKNTNGLLSSYSGAIGVKTGAGPEAKYCLVFAATRGGKTVIGTVLASTSIPARESDATKIMNYGFAL,CCCCCCCCCEEEEEECCCCCEEEEECCCCCECCHHHHHHHHHHHHHCCCCCCCCCEEECCHHHHHHHHHCCCCCCCCCCCCEEEHHHHHHHHHCCCCHHHHHHHHHHHCCCCCHHHHHHHHHHHHHHHHHHCCCCCCECCCCCCCCCCCCEECHHHHHHHHHHHCCCHHHHHHHCCCEECCEEECCCCCEEECCCEECCCCHHHHCCCEEEEEEEEECCCEEEEEEEEEECCEEEEEEEEEECCHHHHHHHHHHHHHHHHHC,0011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
2a6h-E,MAEPGIDKLFGMVDSKYRLTVVVAKRAQQLLRHGFKNTVLEPEERPKMQTLEGLFDDPNAETWAMKELLTGRLVFGENLVPEDRLQKEMERIYPGEREE,CCCCCHHHHHHHCCCHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCCCCHHHCCCCCHHHHHHHHCCCCCCCCCCCCCCCHHHHHHHHHCCCCCCC,011111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111000
5b1a-P,MTHQTHAYHMVNPSPWPLTGALSALLMTSGLTMWFHFNSMTLLMIGLTTNMLTMYQWWRDVIRESTFQGHHTPAVQKGLRYGMILFIISEVLFFTGFFWAFYHSSLAPTPELGGCWPPTGIHPLNPLEVPLLNTSVLLASGVSITWAHHSLMEGDRKHMLQALFITITLGVYFTLLQASEYYEAPFTISDGVYGSTFFVATGFHGLHVIIGSTFLIVCFFRQLKFHFTSNHHFGFEAAAWYWHFVDVVWLFLYVSIYWWGS,CCCCCCCCCCCCCCCHHHHHHHHHHHHHHHHHHHHHCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCHHHCCCCCCCCCCCCCCCCHHHHHHHHHHHHHHHHHHHHHHHHCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCCCCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHCC,001111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
```

## Downstream Task Model Architectures

BioNeMo framework offers two lightweight predictive model architectures for downstream task validation -- multi-layer perceptron model (MLP model) and a convolutional network (ConvNet). The MLP model is always used for sequence-level tasks, and ConvNet is always used for token-level tasks.

### MLP Model

The MLP model is a fully connected neural network architecture with three hidden layers and an output prediction head. The first hidden layer is the same size as the embeddings vector, and the second and third layers have sizes `256` and `128` respectively. The size of the prediction head is either `1` for regression tasks or the number of classes for the classification tasks.

Additionally, `ReLU` activation function and a dropout with `p=0.1` are applied after each hidden layer transformation.

MLP model architecture is not part of the YAML configuration file but can be manually adjusted in `bionemo/model/core/mlp_model.py` if desired.

### ConvNet model

The ConvNet model architecture consists of 2D convolutional layer with `in_channels` equal to the embeddings vector size, `out_channels=32`, `kernel_size=(7, 1)` and `padding=(3, 0)`. The convolutional layer is followed by the `ReLU` activation function and a dropout with `p=0.25`. The last layers are configurable classification prediction heads. The YAML configuration file specifies the number of heads and classes for each head. Other architectural parameters of the ConvNet are not part of the configuration file but can be manually adjusted in `bionemo/model/core/cnn.py` if desired.

## Intended Use

### Setup

Validation with a downstream task is implemented in the form of a callback that is called at the end of every validation epoch. Parameters for the callback must be provided in the YAML configuration file. The callback is set up and added to the trainer with the following commands at the beginning of the training script:

```python

from bionemo.callbacks import setup_dwnstr_task_validation_callbacks


def main(cfg) -> None:
  callbacks = setup_dwnstr_task_validation_callbacks(cfg)
  trainer = setup_trainer(cfg, callbacks=callbacks)
```

Downstream task validation is defined in the `model.dwnstr_task_validation` section of the YAML configuration file. The parameter `model.dwnstr_task_validation.enabled` can be set to `True` to enable the feature, or to `False ` to disable it:

```yaml
model:
  dwnstr_task_validation:
    enabled: True
```

### Model Configuration

If downstream task validation is enabled, a set of parameters must be specified in `model.dwnstr_task_validation.dataset`. These parameters can be split into three categories -- universal, task-specific, and optimization.

### Universal Parameters

* `class` defines a type of callback. Available options are: `bionemo.model.core.dwnstr_task_callbacks.SingleValuePredictionCallback` for sequence-level classification or regression and `bionemo.model.core.dwnstr_task_callbacks.PerTokenPredictionCallback` for token-level classification.

* `task_type` defines a type of downstream task. Available options are `classification` or `regression`. `PerTokenPredictionCallback` supports only `classification`.

* `infer_target` defines the inference class for embeddings extraction and must match the main model architecture. Available options are:
    * `bionemo.model.molecule.megamolbart.infer.MegaMolBARTInference` for MegaMolBART
    * `bionemo.model.protein.prott5nv.infer.ProtT5nvInference` for ProtT5nv
    * `bionemo.model.protein.esm1nv.infer.ESM1nvInference` for ESM-1nv

* `max_seq_length` defines the maximum sequence length for the downstream task. Sequences longer than `max_seq_length` are omitted from training and testing. This parameter is usually inferred from the main model's sequence length.

* `emb_batch_size` defines the value of batch size used to compute embeddings for the downstream task datasets. This value can usually be larger than the main model's micro-batch size since gradients are not computed during the inference.

* `batch_size` defines the batch size of the downstream task model.

* `num_epochs` defines the number of epochs for which the downstream task model is trained.

* `dataset_path` defines the location of datasets for the downstream task. This directory is expected to contain two subdirectories with the names `train` and `test`, each containing CSV files for train and test subsets, respectively.

* `dataset.train` defines the file name or range of file names with the training subset without `.csv` extension.

* `dataset.test` defines the file name or range of file names with the test subset without `.csv` extension.

* `random_seed` defines the random seed value for the downstream task model.

### `SingleValuePredictionCallback` Parameters

* `sequence_column` defines the name of a column in train and test files containing input sequences.

* `target_column` defines the name of a column in train and test files containing target values.

* `num_classes` defines the number of class labels in the target and must be provided only in case of `task_type: classification` in the `SingleValuePredictionCallback`.

An example of `SingleValuePredictionCallback` can be found in `/workspace/bionemo/examples/molecule/megamolbart/conf/pretrain_base.yaml`.

### `PerTokenPredictionCallback` Parameters

* `sequence_col` defines the name of a column in train and test files containing input sequences.

* `target_column` defines the list with names of columns in train and test files containing class labels for each classification task.

* `target_sizes` defines the list with the number of classes for each classification task.

* `mask_col` defines the list with names of columns in train and test files used as masks for each classification task. Should be set to `null`` for any classification task where masking is not needed.

Example usage of `PerTokenPredictionCallback` can be found in `/workspace/bionemo/examples/protein/esm1nv/conf/pretrain_small.yaml`.

### Optimization Parameters

Optimization parameters are provided in `model.dwnstr_task_validation.dataset.optim` section of the YAML configuration file and include all optimizers and learning rate schedulers supported in NeMo. More details on NeMo optimizers and learning rate schedulers can be found in the [NeMo User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/core/core.html#optimization).


## Logging with Weights & Biases and Tensorboard

Test metrics from the downstream task validation step are automatically added to the experiment's W&B and/or Tensorboard dashboards if these options are enable in the configuration file. The following metrics are logged:

* `dwnstr_task_val_loss` -- loss computed on the test subset of the downstream task. This metric is logged regardless of the task type, but every task has its type of loss:
  * cross-entropy loss for sequence-level classification
  * MSE loss for sequence-level regression
  * per-token masked cross-entropy loss for token-level classification. `PerTokenMaskedCrossEntropyLoss` is implemented in BioNeMo and can be found at `bionemo/model/core/cnn.py`.

* Overall classification accuracy is only logged for classification tasks. In the case of sequence-level classification, a single chart is created. A chart for each classification task is created for token-level classification.

* Regression MSE is only logged for regression tasks. A single chart is created in this case.

## Limitations

Validation with a downstream task can be enabled during pre-training with data parallelism and/or tensor parallelism enabled. However, the downstream task model is trained on a single GPU on the rank 0 node, even when the main model is trained using multi-GPUs or multi-nodes. While the downstream task model is being trained, other processes are waiting for this training to complete. For efficient usage of computational resources, it is recommended to minimize the duration and frequency of validation epochs of downstream task model training. The frequency of the validation epoch can be controlled `trainer.val_check_interval` parameter.

Finally, *token-level regression* is not currently implemented, although the existing token-level classification example could be modified to support regression.
