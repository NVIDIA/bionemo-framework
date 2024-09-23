
# DiffDock: Preparing Workspace and Data for Pre-training

This section outlines the steps to prepare your workspace with pre-processed files for training DiffDock using the NVIDIA NGC platform.

This workspace can then be used to launch DiffDock training job using the template script `<BioNeMO_Workspace>/examples/molecule/diffdock/scripts/train_bcp.sh`.

For more information about how to launch DiffDock training using BCP script, you can check [Running BioNeMo on DGX-Cloud using BCP](./bcp-specific-commands-fw.md) tutorial.

The commands below expect that you have NGC CLI installed on your machine. All of the commands below can then be run outside of BioNeMo Docker container.

### Setup

Before diving in, please ensure that you have completed all steps in the [Getting Started](./index.md) section.

### Step 1: Create a Workspace
Create a new NGC workspace by running the following command. Make sure to replace `diffdock_bcp_train` with your desired workspace name and `nv-us-east-2` with your preferred ACE
```bash
ngc workspace create --name diffdock_bcp_train --ace nv-us-east-2
```

Store the workspace ID provided after creating the workspace. You will need this ID for mounting data and running jobs.

```bash
export WKSP_ID=??
```

### Step 2: Create a Local Data Directory
Create a local directory to store pre-training. For this documentation, we'll use `~/diffdock_data`.
```bash
mkdir -p ~/diffdock_data/data
```
### Step 3: Mount the NGC Workspace To Your Local Directory
Mount the NGC workspace to your local directory using the following command.
```bash
ngc workspace mount $WKSP_ID ~/diffdock_data --mode RW
```

### Step 4: Prepare Raw Protein and Ligand Structure Data

The protein and ligand structure data files, e.g., PDB files for proteins and
SDF files for ligands, should be placed in a dedicated data directory, which is
to be passed to the data preprocessing script (see details below) via the
config option `protein_data.protein_data_dir=/path/to/data`. The user needs to
provide a CSV file that specifies the protein-ligand complexes' name and the corresponding
structure data files in the following format:

```csv
complex_name,protein_path,ligand_paths
complex_name_1,protein_path_1,"ligand_path_1a"
complex_name_2,protein_path_2,"ligand_path_2a,ligand_path_2b"
...
```
where each row represents a complex with `complex_name` being its unique
identifier (string) , the `protein_path` being the relative path under the
aforementioned `protein_data.protein_data_dir` parent data directory and `ligand_path_{i}`
being the path (again relative to `protein_data.protein_data_dir`) of `i`'th ligand's structure
file. **A few notes here:**
- Each row can have one protein structure data file with multiple
ligand data files. The ligand data parser will attempt to parse the ligand files one by
one in the order listed in the CSV row and will stop at the first successful parse. This
file format doesn't change the fact that each unique complex have at most one protein
and one ligand but allows more fault tolerance in the ligand data parser.
- The same protein structure file can be reused in different rows of the CSV file.
- We recommend the user not to use dot (i.e., `.`) in the complex name

Below is an example of the directory structure and the corresponding CSV file:

```bash
tree ~/PDB_processed
```
will give this file tree:
```
├── 6t88
│   ├── 6t88_ligand_1.sdf
│   ├── 6t88_ligand_2.sdf
│   └── 6t88_some_protein_name.pdb
├── 6vs3
│   ├── 6vs3_ligand.sdf
│   ├── 6vs3_ligand_2.sdf
│   └── 6vs3_other_protein_name.pdb
...
```
```bash
cat ~/PDB_processed/protein_ligand.csv
```
gives
```
complex_name,protein_path,ligand_paths
6t88,6t88/6t88_some_protein_name.pdb,"6t88/6t88_ligand_1.sdf,6t88/6t88_ligand_2.sdf"
6vs3,6vs3/6vs3_other_protein_name.pdb,"6vs3/6vs3_ligand.sdf"
6vs3_2,6vs3/6vs3_other_protein_name.pdb,"6vs3/6vs3_ligand_2.sdf"
```

Then copy the data to the mounted directory in your NGC workspace.
```bash
cp -r ~/PDB_processed ~/diffdock_data/data
```
And prepare the split files for train, validation and test, and put them in `~/diffdock_data/data/splits/`. Here we name them as `split_train`, `split_val`, and `split_test`, accordingly.
These split files are text files, with each row as pdbid name. These will look like this:
```bash
$ head -n 2 ~/diffdock_data/data/splits/split_train
6t88
6vs3
```

### Step 5: Run Pre-processing of Train Data for Score Model

This script performs pre-processing of train data for diffdock score model.

If you followed the [Initialization Guide](./initialization-guide.md) section, you may already have a .env file in your root path. If so, run `source .env` to export all the necessary variables.
If you don't have a .env file in your path, update and run the following commands to export necessary variables before running the commands below.

```
export NGC_CLI_ORG=??
export NGC_CLI_TEAM=??
export BIONEMO_IMAGE=??
```

First do protein embedding using esm2:

```bash
ngc batch run \
  --name "DiffDock_Protein_Embedding" \
  --priority NORMAL --preempt RUNONCE --ace nv-us-east-2 \
  --instance dgxa100.80g.2.norm \
  --commandline "ln -s /bionemo_diffdock/data \\${BIONEMO_HOME}/data; bcprun --debug --nnodes=1 --npernode=1 -w /workspace/bionemo --cmd \
  'python examples/molecule/diffdock/train.py do_embedding_preprocessing=True do_training=False\
     protein_data.output_fasta_file=/workspace/bionemo/data/pdb_sequences.fasta \
     protein_data.esm_embeddings_path=/workspace/bionemo/data/esm2_embeddings'"\
  --result /results --image ${BIONEMO_IMAGE} --org ${NGC_CLI_ORG} --team ${NGC_CLI_TEAM} --workspace ${WKSP_ID}:/bionemo_diffdock:RW --label ml__bionemo
```

where the ESM2 embedding of the proteins will be outputted in the directory
`/workspace/bionemo/data/esm2_embeddings`, which is mapped to the
aforementioned `~/diffdock_data/data/esm2_embeddings`.

Then, do the graph preprocessing for score model:
```bash
ngc batch run \
  --name "DiffDock_Complex_Graph_Preprocessing_Score" \
  --priority NORMAL --preempt RUNONCE --ace nv-us-east-2 \
  --instance dgxa100.80g.2.norm \
  --commandline "ln -s /bionemo_diffdock/data \\${BIONEMO_HOME}/data; bcprun --debug --nnodes=1 --npernode=1 -w /workspace/bionemo --cmd \
  'python examples/molecule/diffdock/train.py do_preprocessing=True do_training=False\
     data.num_workers=20 \
     data.cache_path=/workspace/bionemo/data \
     data.split_train=/workspace/bionemo/data/splits/split_train \
     data.split_val=/workspace/bionemo/data/splits/split_val \
     data.split_test=/workspace/bionemo/data/splits/split_test \
     protein_data.protein_data_dir=/workspace/bionemo/data/PDB_processed \
     protein_data.protein_ligand_csv=/workspace/bionemo/data/PDB_processed/protein_ligand.csv \
     protein_data.esm_embeddings_path=/workspace/bionemo/data/esm2_embeddings \
     ++model.train_ds.min_num_shards=64 \
     ++model.val_ds.min_num_shards=64 \
     ++model.test_ds.min_num_shards=64'"\
  --result /results --image ${BIONEMO_IMAGE} --org ${NGC_CLI_ORG} --team ${NGC_CLI_TEAM} --workspace ${WKSP_ID}:/bionemo_diffdock:RW --label ml__bionemo
```

**NOTE:** the `model.{train,val,test}_ds.min_num_shards` options tell the
preprocessing workflow to attempt to create at least `64` webdataset shards for
the respective train, validation and test datasets. The user can replace the
number `64` with their own value, **with the constraint being this value needs to
be larger than `num_workers * num_GPUs` later in training, where `num_workers`
is the number of pytorch dataloader worker processes and `num_GPUs` is the
total number of GPUs used to train the model**

### Step 6 : Verify that the data pre-processing is successful in Step 5
You can re-mount your workspace after running through all the steps above and be able to see pre-processed data. Follow Step 3 to mount the workspace onto your local workstation.'
Below is an example directory structure you should see for pre-processed diffdock training data for score and confidence models.
```bash
ls ~/diffdock_data/data
```
It will have following files:
```
data_cache
esm2_embeddings
PDB_processed
pdb_sequences.fasta
splits
```

where the `esm2_embeddings` directory will have something like:
```bash
ls ~/diffdock_data/data/esm2_embeddings
```
```
6vs3_chain_0.pt
6vs3_chain_1.pt
6t88_chain_0.pt
...
```

and the `data_cache` directory will contain the following files:
```bash
ls ~/diffdock_data/data/data_cache
```
```
torsion_limit0_maxLigSizeNone_H0_recRad15_recMax24_esmEmbeddings
torsion_limit0_maxLigSizeNone_H0_recRad15_recMax24_esmEmbeddings_INDEXsplit_train
torsion_limit0_maxLigSizeNone_H0_recRad15_recMax24_esmEmbeddings_INDEXsplit_val
torsion_limit0_maxLigSizeNone_H0_recRad15_recMax24_esmEmbeddings_INDEXsplit_test
```


### Step 7: Run Pre-processing of Train Data for Confidence Model

The dataset preprocessing for confidence model will use a trained score model to do reverse diffusion and to generate ligand conformers.
It is strongly suggested to train a small score model for this to save time and to improve the accuracy of confidence model. Refer to
[Diffdock Model Training](./notebooks/model_training_diffdock.ipynb) for details of training a small score model.

Once you have a trained small score model, copy the model to the mounted directory as follow,
```
mkdir -p ~/diffdock_data/models/
cp /path/to/your/small_score_model.nemo ~/diffdock_data/models/small_score_model.nemo
export DIFFDOCK_SCORE_MODEL=models/small_score_model.nemo
```

we can do the data preprocessing for confidence model:
```bash
ngc batch run --name "DiffDock_Complex_Graph_Preprocessing_Confidence" \
  --priority NORMAL --preempt RUNONCE --ace nv-us-east-2 \
  --instance dgxa100.80g.2.norm \
  --commandline "ln -s /bionemo_diffdock/data \\${BIONEMO_HOME}/data; \
   ln -s /bionemo_diffdock/models/*.nemo \\${BIONEMO_HOME}/models; bcprun --debug --nnodes=1 --npernode=1 -w /workspace/bionemo --cmd \
  'python examples/molecule/diffdock/train.py --config-name=train_confidence do_preprocessing=True do_training=False \
     data.num_workers=20 \
     score_infer.restore_from_path=/workspace/bionemo/${DIFFDOCK_SCORE_MODEL} \
     ++score_infer.data.cache_path=/workspace/bionemo/data/data_cache \
     ++score_infer.data.split_train=/workspace/bionemo/data/splits/split_train \
     ++score_infer.data.split_val=/workspace/bionemo/data/splits/split_val \
     ++score_infer.data.split_test=/workspace/bionemo/data/splits/split_test \
     ++model.train_ds.min_num_shards=64 \
     ++model.val_ds.min_num_shards=64 \
     ++model.test_ds.min_num_shards=64'" \
     --result /results --image ${BIONEMO_IMAGE} --org ${NGC_CLI_ORG} --team ${NGC_CLI_TEAM} --workspace ${WKSP_ID}:/bionemo_diffdock:RW --label ml__bionemo
```

**NOTE:** the `model.{train,val,test}_ds.min_num_shards` options tell the
preprocessing workflow to attempt to create at least `64` webdataset shards for
the respective train, validation and test datasets. The user can replace the
number `64` with their own value, **with the constraint being this value needs to
be larger than `num_workers * num_GPUs` later in training, where `num_workers`
is the number of pytorch dataloader worker processes and `num_GPUs` is the
total number of GPUs used to train the model**

If you want to use the score model checkpoint converted from [public diffdock](https://github.com/gcorso/DiffDock/tree/main/workdir/paper_score_model), replace the setting of small score model with following commands:
```bash
ngc registry model download-version nvidia/clara/diffdock_score:1.5 --dest  ~/
cp ~/diffdock_score.nemo ~/diffdock_data/models/
export DIFFDOCK_SCORE_MODEL=models/diffdock_score.nemo
```
And use the same ```ngc batch run``` command above
