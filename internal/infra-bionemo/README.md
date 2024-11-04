# bionemo_infra
Tools maintained by the BioNeMo Framework Infrastructure team.


## Development
All code must be formatted & linted using `ruff` and type checked using `mypy`.

To type check, run:
```bash
mypy --install-types --non-interactive --ignore-missing --check-untyped-defs .
```

All code must have type annotations. In special circumstances, such as private or helper functions, may elide static
type annotations if it is beneficial.

To run unit tests, use `pytest -v`. Unit tests must cover all features and known bug cases.

To get code coverage reports, run:
```bash
pytest -v --cov=infra_bionemo --cov-report=term --cov-report=html .
```
Then, open the [HTML coverage report](htmlcov/index.html) in your browser. Note that file coverage stats are reported
in the terminal too.

### First Time Setup
For first time setup, be sure to install the development and test dependencies of the entire bionemo repository.
These are defined at the repository's top-level [`pyproject.toml`](../../pyproject.toml) file. Follow the instructions
outlined in the [top-level README](../../README.md). Once you have your local virtual environment ready, you may
install this project's code by running the following:
```bash
uv pip install --editable .
```

NOTE: you may have to re-install the project if there are new or removed packages. If the requirements haven't changed,
      you can re-install without re-installing the dependencies by doing: `uv pip install --no-deps -e .`

### Versioning
This project uses [Semantic Versioning 2.0](https://semver.org/). Contributors *MUST* update the `version` in
`pyproject.toml` correctly in their MRs. The CI will reject MRs that do not increment the version number correctly.
