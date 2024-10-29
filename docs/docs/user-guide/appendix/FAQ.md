# Frequently Asked Questions

### Is BioNeMo free to use?

Yes, BioNeMo is free to use. BioNeMo code is licensed under the Apache 2.0 License. The Apache 2.0 License is a
permissive open-source license that allows users to freely use, modify, and distribute software. With this license,
users have the right to use the software for any purpose, including commercial use, without requiring royalties or
attribution. Overall, our choice of the Apache 2.0 License allows for wide adoption and use of BioNeMo, while also
providing a high degree of freedom and flexibility for users.

### How do I install BioNeMo?

BioNeMo is distributed as a Docker container through NVIDIA NGC. To download the pre-built Docker container and data
assets, you will need a free NVIDIA NGC account.

Alternatively, you can install individual sub-packages from within BioNeMo by following the corresponding README pages
the [BioNeMo GitHub](https://github.com/NVIDIA/bionemo-framework).

### How do I update BioNeMo to the latest version?

To update the BioNeMo Docker container, you need to pull the latest version of the Docker image using the command
`docker pull`. For available tags, refer to the
[BioNeMo Framework page in the NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework).

### What are the system requirements for BioNeMo?

Generally, BioNeMo should run on any NVIDIA GPU with Compute Capability ≥8.0. For a full list of supported hardware,
refer to the [Hardware and Software Prerequisites](../getting-started/pre-reqs.md).

### Can I contribute code or models to BioNeMo?

Yes, BioNeMo is open source and we welcome contributions from organizations and individuals. For more information about
external contributions, refer to the [Contributing](../contributing/contributing.md) and
[Code Review](../contributing/code-review.md) pages.

### How do I report bugs or suggest new features?

To report a bug or suggest a new feature, open an issue on the
[BioNeMo GitHub site](https://github.com/NVIDIA/bionemo-framework/issues). For the fastest turnaround, thoroughly
describe your issue, including any steps and/or _minimal_ data sets necessary to reproduce (when possible), as well as
the expected behavior.

### Can I train models in Jupyter notebooks using BioNeMo?

At the current time, notebook-based training is not supported due to restrictions imposed by the Megatron framework that
underpins the BioNeMo models.
