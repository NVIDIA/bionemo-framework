# Environment Setup

From the bionemo-moco directory run:

```bash
bash environment/setup.sh
```

This creates the conda environment, installs bionemo-moco and runs the tests.

# Local Code Setup

From the bionemo-moco directory run:

```bash
bash environment/clone_bionemo_moco.sh
```

This creates clones only the bionemo subpackage. To install in your local env use:

```bash
pip install -e .
```

inside the bionemo-moco directory.

```bash
pip install --no-deps -e .
```

can be used if want to install bionemo-moco over your current torch version. The remaining required jaxtyping and pot dependencies can be manually installed via pip.
