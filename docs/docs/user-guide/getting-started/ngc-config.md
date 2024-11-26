# NGC Command Line Interface Configuration

To access the any private data on NGC, you will need a free NVIDIA GPU Cloud (NGC) account and an API key linked to
that account. We strive to make all assets necessary to use BioNeMo available without an NGC account. However, some
licensing or release agreements may preclude that now or in the future.

### NGC Account and API Key Configuration

NGC is a portal of enterprise services, software, and support for artificial intelligence and high-performance computing
(HPC) workloads. To pull private assets from NGC, you will need to create a free NGC account and an API Key using the
following steps:

1. Create a free account on [NGC](https://ngc.nvidia.com/signin) and log in.
2. At the top right, click on the **User > Setup > Generate API Key**, then click **+ Generate API Key** and
**Confirm**. Copy and store your API Key in a secure location.

### NGC CLI Configuration

The NGC Command Line Interface (CLI) is a command-line tool for managing resources in NGC, including datasets and model
checkpoints. You can download the CLI on your local machine using the instructions
[on the NGC CLI website](https://org.ngc.nvidia.com/setup/installers/cli).

Once you have installed the NGC CLI, run `ngc config set` at the command line to setup your NGC credentials:

* **API key**: Enter your API Key
* **CLI output**: Accept the default (ASCII format) by pressing `Enter`
* **org**: Choose your preferred organization from the supplied list
* **team**: Choose the team to which you have been assigned from the supplied list
* **ace** : Choose an ACE, if applicable, otherwise press `Enter` to continue
