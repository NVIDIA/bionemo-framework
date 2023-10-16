Development in BioNeMo is done through forks and Merge Requests (MRs). When starting to contribute to BioNeMo, create your personal fork, add your feature branch from `dev`, commit your changes and then make a merge request into the main repo.

Don't create branches directly in the main repo.

# Merge Requests (MR) Guidelines

**Send your MRs to the `dev` branch**

1) Make sure your MR does one thing. Have a clear answer to "What does this MR do?"
2) Make sure you have the linters enabled via pre-commit hooks (`pre-commit install`)
2) Follow the default MR template
3) Make sure all unit tests finish successfully before running MR pipeline by invoking `pytest` in `bionemo` folder
4) Run `pytest examples/tests/test_model_pretrain_and_downstream.py -k test_model_training`, if changes to the codebase are made in training or inference-related pyton scripts (these tests are less comprehensive than tests in JET but can help you to spot issues before running `jet` stage in CI)
4) Make sure you added necessary tests and documentation changes (could be just comments in the config files) for the feature in your MR
5) Rebase your feature branch with the latest `dev` to include any new changes that have been added. Resolve merge conflicts, if any
6) Send your MR and request a review
7) If your MR is still WIP, mark it as "Draft"
8) Set `JET_NOT_REQUIRED` label as one of MR's labels if the MR is eligible for NOT running `jet` stage (and tests in JET) - more info below
8) Your merge request must pass all pipelines and be peer-reviewed before it can be merged.
9) Make sure to merge your MR when it's ready and pipeline is successful

## Unit tests
Following [Wikipedia](https://en.wikipedia.org/wiki/Unit_testing), unit testing is a testing method by which individual units of source code—sets of one or more computer program modules together with associated control data, usage procedures, and operating procedures—are tested to determine whether they are fit for use.

Contributors to BioNeMo FW are expected to unit test their introduced changes. Tests must be run locally in the docker container with incorporated changes while developing with the following command:
```
pytest
```
If your changes to the codebase are related to the model training and inference (used classes or configs) make sure to test locally if **basic unit tests** for training and inference pass by running
`pytest examples/tests/test_model_pretrain_and_downstream.py -k test_model_training`

As an example, unit tests in `dev-latest-devel` container can be run using SLURM
```
srun -t 00:30:00 -J unit-tests -N 1 -o=<OUTPUT_PATH>/pytest-slurm-%A.out --container-image gitlab-master.nvidia.com/clara-discovery/bionemo:dev-latest-devel bash -c "set -x; set -e; cd /opt/nvidia/bionemo; pytest"
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
1) `build` - builds a pipeline-specific docker image which can be found in the [Container Registry](https://gitlab-master.nvidia.com/clara-discovery/bionemo/container_registry) searching for `pipeline-<GITLAB-PIPELINE_ID>` and `pipeline-<GITLAB-PIPELINE_ID>-devel`
2) `download` - the checkpoints of the models listed in `artifact_paths` are downloaded by `download_models.sh`
3) `test` - CPU-specific and GPU-specific unit tests are run using `pytest`, excluding `pytest examples/tests/test_model_pretrain_and_downstream.py -k test_model_training`
4) `jet` - comprehensive performance and convergence tests of BioNeMo models that are run and managed by [JET](https://jet.nvidia.com/docs), this step can be omitted if a MR is eligible for NOT running it (see below). More information on JET in `internal/README.md`

JET stage is manually triggered to avoid unnecessary pipelines in JET to be run. To trigger it, click on the button `jet-generate` in your MR pipeline window.

Before MR is ready to be merged, all CI pipelines must be completed and successful. Otherwise, the merge is blocked.

## Type of changes to the codebase that can be merged WITHOUT BioNeMo CI or `jet` stage being triggered
One stage of a pipeline is called `jet` and triggers comprehensive performance and convergence tests of BioNeMo models in [JET](https://jet.nvidia.com/docs). The tests are more comprehensive than the tests invoked by `pytest examples/tests/test_model_pretrain_and_downstream.py -k test_model_training`.
This stage is resources- and time-consuming and can be **OMITTED** for some changes to the codebase by `JET_NOT_REQUIRED` label as one of MR's labels. 
Also, some changes to the codebase do not require to run time-consuming BioNeMo CI and can use `SKIP_CI` label.

The changes to the codebase that are eligible for using `SKIP_CI` label are:
* changes to the files with extension `.md` or `.ipynb`
* changes under folders `docs`, `LICENSE`, 
* changes to the files with extension `.sh` under `examples/**/scripts/*.sh` related to training scripts of models
* updating files with extensions different than `*.sh`, `*.py`, `*.yaml`, `*.yml`,  `Dockerfile*` or `requirements.txt` that **DO NOT** affect model checkpoints or data download, docker building, unit tests and model performance or convergence

The changes to the codebase that are eligible for using `JET_NOT_REQUIRE` label are:
* docstrings update in `.py` files
* code cleanup not related to refactoring of code (ie deleting unused imports or blank lines, improving lines formatting) in `*.py` files
* improving hydra configs docstrings (comments and descriptions) in  `*.yaml`, `*.yml`
* changes to `Dockerfile` or `requirements.txt` that **DO NOT** affect model performance or convergence. Changes that **REQUIRE** `jet` stage are, for instance, python package update or a NeMo container version update.
* updating files with extensions different that `*.py`, `*.yaml`, `*.yml`,  `Dockerfile` or `requirements.txt` that **DO NOT** affect model performance or convergence

As a final remark, most of the changes to files with extensions `*.py`, `*.yaml`, `*.yml`,  `Dockerfile*` or `requirements.txt` DO REQUIRE both BioNeMo CI and `jet` stage to be run, should be carefully tested and ARE NOT eligible to use `SKIP_CI` or `JET_NOT_REQUIRE` label as they affect model performance or convergence. 

# General principles
1. **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
1. **Robust**: make it hard for users to make mistakes.
1. **Well-tested**: please add simple, fast unit tests.
1. **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to reuse.
1. **Readable**: code should be easy to read and well documented (with comments and docstrings).
1. **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that BioNeMo supports. Give credit and link back to the code.
1. **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.

## Python style

1. Include docstrings for every class and method exposed to the user.
1. Use Python 3 type hints for every class and method exposed to the user.
1. Avoid wild import: ``from X import *`` unless in ``X.py``, ``__all__`` is defined.
1. Minimize the use of ``**kwargs``.
1. ``RaiseError`` is preferred to ``assert``. Write: ```if X: raise Error``` instead of ```assert X```.
1. Classes are preferred to standalone methods.
1. Methods should be atomic. A method shouldn't be longer than 75 lines, e.g. can be fit into the computer screen without scrolling.
1. If a method has arguments that don't fit into one line, each argument should be in its own line for readability.
1. Add ``__init__.py`` for every folder.
1. F-strings are preferred to formatted strings.
1. Loggers are preferred to print. In BioNeMo, you can use logger from ``from nemo.utils import logging``
1. Private functions (functions starting with ``_``) shouldn't be called outside its host file.
1. If a comment lasts multiple lines, use ``'''`` instead of ``#``.

