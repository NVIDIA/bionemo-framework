# Contributing guidelines for internal bionemo2 contributions

Note: For code review standards please see [CODE-REVIEW](CODE-REVIEW.md)

Note: For all PRs, an approved NVIDIA staff member must sign off and trigger the continuous integration (CI) tests. These are initiated by the member commenting `/build-ci` directly on the PR. All PRs must have successful CI runs and sufficient code review before being merged.


## Python Coding Standards

This page contains the Python coding standards for the BioNeMo repository. They apply to all Python code in the repository (unless external constraints prevent it).


# General principles

## Coding Style
- We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with a few tweaks.
- The most important parts of this style guide that our code must adhere to are:
  - [Docstring](https://google.github.io/styleguide/pyguide.html#381-docstrings)
  - [Mutable global state](https://google.github.io/styleguide/pyguide.html#25-mutable-global-state)
  - [Do not use mutable values as default arguments](https://google.github.io/styleguide/pyguide.html#212-default-argument-values)
  - [Default iterators](https://google.github.io/styleguide/pyguide.html#28-default-iterators-and-operators)
  - [Bad naming / abbreviation](https://google.github.io/styleguide/pyguide.html#316-naming)
- The exceptions to this style guide are:
  + [Module](https://google.github.io/styleguide/pyguide.html#22-imports) imports. If a module is uniquely named, import the module. Otherwise, import the value, type, or function directly.
- Linting and formatting of all code is required by using `ruff` with bionemo's configured options.
- Unit testing with `pytest`.
- Add type annotations everywhere. In particular, new code should all be type-annotated as thoroughly as possible. This also obviates the need for including type hints in the function docstring. It is ok to omit annotations for private helper functions, but use your best judgement.
- Include docstrings for every class, function, and method exposed to the user.
  +Docstrings **should** answer (a) what is the code doing and (b) why would someone use it.
- Never use wildcard imports.
- Define `__all__ = (,)` in modules: make explicit the API of each module, auto-documenting the most important definitions.
- Minimize the use of `**kwargs`.
- `raise` an `Exception` instead of using an `assert` statement.
- F-strings are preferred to format strings.
- Loggers are preferred to print. In BioNeMo, you can use logger from `import logging`.
- Private functions (functions starting with ``_``) shouldn't be called outside its host file.


### General Guidelines
- **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
- **Robust**: make it hard for users to make mistakes.
- **Well-tested**: please add simple, fast unit tests.
- **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to reuse.
- **Readable**: code should be easy to read and well documented (with comments and docstrings).
- **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that BioNeMo supports. Give credit and link back to the code.
- **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.
- **Consistency**: we work in a team. It is important to integrate changes with existing code.
- **Readability**: your code should be easy to read and understand by any other engineer, including outside NVIDIA. Some tips:
  + Document your code. Make all comments complete sentences, starting with a capitalized letter and ending with a period.
  + Avoid abbreviations: 'bn' is harder to understand than 'batch_norm'.
  + Avoid baked-in constants throughout the code. Instead, specify them as parameters to your function. If you must have a constant, follow the naming guideline (e.g., `GLOBAL_CONSTANT`).
  + Avoid functions that span hundreds of lines. Large functions are more difficult to read and more difficult to test. If >120 lines, consider re-factoring it into smaller logical functions, each unit-tested and well-documented.
  + Re-use code by importing. **Do not copy and paste code.**
  + Usage of third-party code should be legally compatible and attributed.

### Coding Style



# Merge Requests (MR) Guidelines

**Send your MRs to the `dev` branch**. Branch off from `dev` when making your changes.
Prefix your branches with your name or initials (i.e. `yourname/branch_description`).

- Make sure your MR does one thing. Have a clear answer to "What does this MR do?"
- Make sure you have the linters enabled via pre-commit hooks (`pre-commit install`)
- Follow the default MR template
- Make sure all unit tests finish successfully before running MR pipeline by invoking `pytest`.
- Run `pytest examples/tests/test_model_pretrain_and_downstream.py -k test_model_training`, if changes to the codebase are made in training or inference-related pyton scripts (these tests are less comprehensive than tests in JET but can help you to spot issues before running `jet` stage in CI)
- Make sure you added necessary tests and documentation changes (could be just comments in the config files) for the feature in your MR
- Rebase your feature branch with the latest `dev` to include any new changes that have been added. Resolve merge conflicts, if any
- Send your MR and request a review
- If your MR is still WIP, mark it as "Draft"
- Set `JET_NOT_REQUIRED` label as one of MR's labels if the MR is eligible for NOT running `jet` stage (and tests in JET) - more info below
- Your merge request must pass all pipelines and be peer-reviewed before it can be merged.
- Make sure to merge your MR when it's ready and pipeline is successful

## Unit tests
Contributors to BioNeMo FW are expected to unit test their introduced changes. Tests must be run locally in the docker container with incorporated changes while developing with the following command:
```
pytest
```
If your changes to the codebase are related to the model training and inference (used classes or configs) make sure to test locally if **basic unit tests** for training and inference pass by running
`pytest examples/tests/test_model_pretrain_and_downstream.py -k test_model_training`

As an example, unit tests in `dev-latest-devel` container can be run using SLURM
```
srun -t 00:30:00 -J unit-tests -N 1 -o=<OUTPUT_PATH>/pytest-slurm-%A.out --container-image github.com/NVIDIA/bionemo-fw-ea:dev-latest-devel bash -c "set -x; set -e; cd /opt/nvidia/bionemo; pytest"
```

After testing your code locally, trigger tests in the MR's CI. Go to your MR -> "Pipelines" and trigger the pipeline by clicking an arrow sign or click on the pipeline id and trigger stages manually.

### Adding unit tests for new classes or methods
Add unit tests under `tests` to examine use cases of new classes or methods that are being added to the codebase. The names of scripts must be of a format `test_*.py`. Check other scripts in this folder for help on how to write tests.

### Adding unit tests for new models
Add short training or inference unit tests to `examples/tests` that are run by `examples/tests/test_model_pretrain_and_downstream.py` . The tests shouldn't be resource- and time-hungry (use ideally 1 GPU, 1 node and a small batch size) and use small data samples. It would involve:
* adding data samples under `examples/tests/test_data`
* adding training or inference configs for unit tests to `examples/tests/conf` based on the configs that are used to pretrain, finetune or run inference of a new model (ie following the logic of the other configs in this folder)
* generate expected configs by running `UPDATE_EXPECTED_CFG=1 pytest examples/tests/test_model_pretrain_and_downstream.py`
* generate expected results by running `UPDATE_EXPECTED_RESULTS=1  pytest examples/tests/test_model_pretrain_and_downstream.py`
* run `examples/tests/test_model_pretrain_and_downstream.py`

### Changes to the unit tested expected results and configs
Remember, that reproducibility of the training and inference results in pytorch is not guaranteed, see more in [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) .
The small discrepancies between expected results in the unit test `test_model_training` are expected. If larger differences are observed and are not expected (ie convergence regression), it might be an indication that your changes to the codebase are affecting training or inference performance.  You may need to consult other BioNeMo developers.

If your changes modify expected test results or test configs and are **anticipated**, they can be updated with the following commands:
```
UPDATE_EXPECTED_RESULTS=1  pytest examples/tests/test_model_pretrain_and_downstream.py

UPDATE_EXPECTED_CFG=1 pytest examples/tests/test_model_pretrain_and_downstream.py
```


## Stages of the gitlab CI pipeline during Merge Requests
The MR pipeline must be completed successfully if MR is to be merged. The subsequent stages are outlined in  `.gitlab-ci.yml` file:
1) `build` - builds a pipeline-specific docker image which can be found in the [Container Registry](https://github.com/NVIDIA/bionemo-fw-ea/container_registry) searching for `pipeline-<GITLAB-PIPELINE_ID>` and `pipeline-<GITLAB-PIPELINE_ID>-devel`
2) `download` - the checkpoints of the models listed in `artifact_paths.yaml` are downloaded by `download_artifacts.py`
3) `test` - CPU-specific and GPU-specific unit tests are run using `pytest`, excluding `pytest examples/tests/test_model_pretrain_and_downstream.py -k test_model_training`
4) `jet` - comprehensive performance and convergence tests of BioNeMo models that are run and managed by [JET](https://jet.nvidia.com/docs), this step can be omitted if a MR is eligible for NOT running it (see below). More information on JET in `internal/README.md`

JET stage is manually triggered to avoid unnecessary pipelines in JET to be run. To trigger it, click on the button `jet-generate` in your MR pipeline window.

Before MR is ready to be merged, all CI pipelines must be completed and successful. Otherwise, the merge is blocked.


## Merge / Pull Request Guidelines

You should always carefully test your changes. Run `pytest ...` in-container locally. All tests are done via `pytest`.

To run **all** tests, you must first download all models and datasets to your machine. Run the `download_models.py` file to do this. Note that you only need to do this once per machine. Reruns are necessary only when new models and datasets are added.
Changes that affect model training accuracy or compute performance should be tested on SLURM.

Major features or changes should be discussed and designed before the PR review stage.
Design iteration for minor features, documentation changes, and bugs may occur during PR review.

### <a name="before-pr-ready"></a> Before your PR is "Ready for review?

Before asking for reviewers, be sure to review your own changes first. For all contributed code, be sure you have:
- documentation updates
- tests
- verified that the covered tests run successfully in CI

**Most of the changes** to files with extensions `*.py`, `*.yaml`, `*.yml`, `Dockerfile*` or `requirements.txt` **DO REQUIRE both `pytest-` and `jet-` CI jobs** of `stage: test`.

However, these are resource-intensive stages. The `pytest-` stages require GPU enabled runners.
The `jet-` stages require SLURM cluster access and run 10s of jobs per pipeline using [JET](https://jet.nvidia.com/docs).

The `JET_NOT_REQUIRED` MR label disables running JET tests. The `SKIP_CI` label additionally disables the `pytest-` tests. For more context on JET, please see the [JET README](https://github.com/NVIDIA/bionemo-fw-ea/-/tree/dev/internal/jet?ref_type=heads).
The next sections detail when these labels can be used on an MR.

### <a name="skip-ci"></a> Can you add the `SKIP_CI` label to your MR?

_Why would you use this?_ Makes the MR skip the `pytest-`, docker image building, and `jet-` CI jobs, which take a very long time to complete (~3-4 hours).

_When can you use this?_ The changes to the codebase that are eligible for using `SKIP_CI` label are:

* changes to the files with extension `.md` or `.ipynb`
* changes under folders `docs`, `LICENSE`,
* changes to the files with extension `.sh` under `examples/**/scripts/*.sh` related to training scripts of models
* changes to the other files with extension `.sh` not affecting container build, models and data download for unit test or JET tests
* updating files with extensions different than `*.sh`, `*.py`, `*.yaml`, `*.yml`, `Dockerfile*` or `requirements.txt` that **DO NOT** affect model checkpoints or data download, docker building, unit tests and model performance or convergence

### <a name="pytest"></a> Can you add the `PYTEST_NOT_REQUIRED` label to your MR?

_Why would you use this?_ Makes the MR skip the `pytest-` CI jobs, which require GPU resources and take 30-40m to complete

_When can you use this?_ The changes to the codebase that are eligible for using `PYTEST_NOT_REQUIRED` label are:

* changes to the files with extension `.md` or `.ipynb`
* changes under folders `docs`, `LICENSE`,
* changes to the other files with extension `.sh` not affecting container build, models and data download for unit test
* updating files with extensions different than `*.sh`, `*.py`, `*.yaml`, `*.yml`, `Dockerfile*` or `requirements.txt` that **DO NOT** affect model checkpoints or data download, docker building, unit tests and model performance or convergence

### <a name="skip-jet"></a>Can you add the `JET_NOT_REQUIRED` label to your MR?

_Why would you use this?_ Makes the MR skip the `jet-` CI jobs. The `jet-` jobs are model convergence tests, which take many hours to complete.

_When can you use this_? Broadly, you can use this whenever your changes do not affect model training.

More specifically, the changes to the codebase that are eligible for using `JET_NOT_REQUIRED` label are:
* new parts of the code that do not affect model training (e.g. triton inference, new examples, new tests, new data loaders / data loaders not picked as convergence test DL, etc.)
* docstrings update in `.py` files
* code cleanup not related to refactoring of code (ie deleting unused imports or blank lines, improving lines formatting) in `*.py` files
* improving hydra configs docstrings (comments and descriptions) in `*.yaml`, `*.yml`
* changes to `Dockerfile` or `requirements.txt` that **DO NOT** affect model performance or convergence. Changes that **REQUIRE** `jet` stage are, for instance, python package update or a NeMo container version update.
* updating files with extensions different than `*.py`, `*.yaml`, `*.yml`, `Dockerfile` or `requirements.txt` that **DO NOT** affect model performance or convergence
