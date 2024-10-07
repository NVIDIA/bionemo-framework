# bionemo_infra
Tools maintained by the BioNeMo Framework Infrastructure team.


## Development
All code must be formatted & linted using `ruff` and type checked using `mypy`. To type check, run:
```bash

```

All code must have type annotations. In special circumstances, such as private or helper functions, may elide static
type annotations if it is beneficial.

To run unit tests, use `pytest -v`. Unit tests must cover all features and known bug cases.

### First Time Setup
To setup, create a virtual environment, install dependencies, the project, and the pre-commit hooks:
```bash
conda create -y -n ci-metrics python=3.10
conda activate ci-metrics
pip install -r requirements-dev.txt
pip install -r requirements.txt
pip install -r requirements-test.txt
pip install --no-deps -e .
pre-commit install
```

### Versioning
This project uses [Semantic Versioning 2.0](https://semver.org/). Contributors *MUST* update the `version` in
`pyproject.toml` correctly in their MRs. The CI will reject MRs that do not increment the version number correctly.
