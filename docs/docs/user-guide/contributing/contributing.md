# Contributing Guidelines

!!! note
    For code review standards please see the [Code Review](code-review.md) page.

    For all PRs, an approved NVIDIA staff member must sign off and trigger the continuous integration (CI) tests.
    These are initiated by the member commenting `/build-ci` directly on the PR. All PRs must have successful CI runs and
    sufficient code review before being merged.

## Developer Certificate of Origin (DCO)

We require that all contributors "sign-off" on their commits (not GPG signing, just adding the `-s | --signoff`
argument, or follow the instructions below for auto-signing). This sign-off certifies that you adhere to the  Developer
Certificate of Origin (DCO) ([full text](https://developercertificate.org/)); in short that the contribution is your
original work, or you have rights to submit it under the same license or a compatible license.

Any contribution which contains commits that are not signed-off will not be accepted.

To sign off on a commit, simply use the `--signoff` (or `-s`) option when committing your changes:

```bash
git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

If you would like this to happen automatically to all of your commits, you can modify
your local `~/.git-config-template.txt` file. You can do this with a command like the
following:

```
echo "Signed-off-by: Your Name <your@email.com>" > ~/.git-commit-template.txt
git config --local commit.template ~/.git-commit-template.txt
```

If you have a commit that you want to retroactively sign, you can do that with:

```
git commit --amend --no-edit --signoff
```

## Python Coding Standards

This page contains the Python coding standards for the BioNeMo repository. They apply to all Python code in the
repository (unless external constraints prevent it).

### Coding Style

- We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with a few tweaks.
- The most important parts of this style guide that our code must adhere to are:
  - [Docstring](https://google.github.io/styleguide/pyguide.html#381-docstrings)
  - [Mutable global state](https://google.github.io/styleguide/pyguide.html#25-mutable-global-state)
  - [Do not use mutable values as default arguments](https://google.github.io/styleguide/pyguide.html#212-default-argument-values)
  - [Default iterators](https://google.github.io/styleguide/pyguide.html#28-default-iterators-and-operators)
  - [Bad naming / abbreviation](https://google.github.io/styleguide/pyguide.html#316-naming)
- The exceptions to this style guide are:
  - [Module](https://google.github.io/styleguide/pyguide.html#22-imports) imports. If a module is uniquely named, import
    the module. Otherwise, import the value, type, or function directly.
- Linting and formatting of all code is required by using `ruff` with BioNeMo's configured options.
- Unit testing with `pytest`. See [Unit Tests](#unit-tests) for more details.
- Add type annotations everywhere. In particular, new code should all be type-annotated as thoroughly as possible. This
  also obviates the need for including type hints in the function docstring. It is ok to omit annotations for private
  helper functions, but use your best judgement.
- Include docstrings for every class, function, and method exposed to the user.
  - Docstrings **should** answer (a) what is the code doing and (b) why would someone use it.
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
- **Well-tested**: please add simple, fast unit tests. See [Unit Tests](#unit-tests).
- **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to reuse.
- **Readable**: code should be easy to read and well documented (with comments and docstrings).
- **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that
  BioNeMo supports. Give credit and link back to the code.
- **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.
- **Consistent**: we work in a team. It is important to integrate changes with existing code.
- **Readable**: your code should be easy to read and understand by any other engineer, including outside NVIDIA. Some
  tips:
  - Document your code. Make all comments complete sentences, starting with a capitalized letter and ending with a
    period.
  - Avoid abbreviations: 'bn' is harder to understand than 'batch_norm'.
  - Avoid baked-in constants throughout the code. Instead, specify them as parameters to your function. If you must have
    a constant, follow the naming guideline (e.g., `GLOBAL_CONSTANT`).
  - Avoid functions that span hundreds of lines. Large functions are more difficult to read and more difficult to test.
    If >120 lines, consider re-factoring it into smaller logical functions, each unit-tested and well-documented.
  - Re-use code by importing. **Do not copy and paste code.**
  - Usage of third-party code should be legally compatible and attributed.

## Pull Request (PR) Guidelines

### Labeling Your PR as External Contributor

If you are an external contributor (not an NVIDIA employee), please add the `contribution` label to your PR before
submitting. Labels can be accessed in the right sidebar of the GitHub user interface when creating or editing a PR.

### CI Pipeline Configuration Controls

CI pipeline behavior can be controlled via checkboxes in PR descriptions to optimize test execution:

Key behaviors:

- Controls processed automatically on PR submit/update
- Labels applied based on checkbox status
- Invalid combinations default to most restrictive option

#### **SKIP_CI**

- Skips entire CI pipeline
- Use for documentation typos, README updates

#### **INCLUDE_NOTEBOOKS_TESTS**

- Enables notebook validation tests
- Use when modifying notebooks or notebook-related code
- Disabled by default

### Developer workflows

You should always carefully test your changes. Run `pytest ...` in your container locally. All tests are done via `pytest`.

Changes that affect model training accuracy or compute performance should be tested on SLURM.

Developer workflow for _external_ code contributions is as follows:

1. External developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the
[upstream](https://github.com/NVIDIA/bionemo-framework/tree/main) BioNeMo OSS repository and for BioNeMo2 (this branch)
use the `main` branch as base.

2. Clone the forked repository and push changes to the personal fork.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git bionemo-framework
# Checkout the targeted branch and commit changes
# Push the commits to a branch on the fork (remote).
git push -u origin <local-branch>:<remote-branch>
```

Developer workflow for _internal_ or those developers that have been granted push access to our repository is as follows:

1. Clone this repository locally
2. Create a branch which ideally should be of the form `username/branch_description`
3. Push branch up to our repository `git push -u origin HEAD`

For both internal and external developers, the next step is opening a PR:

1. Once the code changes are staged on the fork and ready for review, a
  [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) can be
    [requested](https://help.github.com/en/articles/creating-a-pull-request) to merge the changes from a branch of the
    fork or branch into `main`.
    - Exercise caution when selecting the source and target branches for the PR.
    Note that versioned releases of TensorRT OSS are posted to `release/` branches of the upstream repo.
    - Creation of a PR creation kicks off the code review process.
    - At least one TensorRT engineer will be assigned for the review.
    - While under review, mark your PRs as work-in-progress by prefixing the PR title with [WIP].
2. Once ready, CI can be started by a developer with permissions when they add a `/build-ci` comment. This must pass
  prior to merging.

### General guidelines

**Send your PRs to the `main` branch**. Branch off from `main` when making your changes.
Prefix your branches with your name or initials (for example, `your_name/branch_description`) if you have push access to
our repository otherwise please create a fork with your branch and submit a PR with `main` as the target.

- Make sure your PR does one thing. Have a clear answer to "What does this PR do?"
- Make sure you have the linters enabled via pre-commit hooks (`pre-commit install`) (See also [Pre-commit
  validation](#pre-commit-validation))
- Follow the default PR template
- Make sure all unit tests finish successfully before running PR pipeline by invoking `pytest scripts sub-packages`.
- Make sure you added necessary tests and documentation changes (could be just comments in the config files) for the
  feature in your PR
- Rebase your feature branch with the latest `main` to include any new changes that have been added. Resolve merge
  conflicts, if any
- Send your PR and request a review
- If your PR is still a work in progress, mark it as "Draft"
- Your merge request must pass all pipelines and be peer-reviewed before it can be merged.
- Make sure to merge your PR when it is ready and pipeline is successful

### Unit tests

Contributors to BioNeMo FW are expected to unit test their introduced changes.

After testing your code locally, trigger tests in the PR's CI. Let a code-owner know that you are ready for the build to
 run and they will leave a `/build-ci` comment on your PR which will run the CI test suite.

#### Adding unit tests

Add unit tests under `tests` to examine use cases of new classes or methods that are being added to the codebase. Each
test file must be for a particular file or module. For example if you have a file that is under
`src/path/to/module/my_file_name.py` then your test should match the path at `tests/path/to/module/test_my_file_name.py`.
Check the tests folders in the sub-modules of this repository for examples. If you are testing a module, such as
integrating multiple examples of different files, then you can use the following pattern to test the module, say in the
above example, if you wanted to test functions from several files together that all exist in the same `src/path/to/module`
then you could create a `tests/path/to/test_module.py` file. The same is true for parents of that module and so on.
Generally unit tests should exist at the level of the individual file however.

## Pre-commit validation

We use [pre-commit](https://pre-commit.com/) for essential static checks. These checks are enforced on new PRs through
the CI process, but should also be run locally. After following the installation instructions for pre-commit, run
`pre-commit install` in the bionemo-framework repository to initialize the checks.

To run pre-commit checks (and fix errors where possible), run `pre-commit run --all-files`. To ignore a pre-commit error
locally, use `git commit -n ...` to allow the commit to proceed with some failing pre-commit checks.

### Updating License Header on Python Files

If you add new Python (`.py`) files, be sure to run our license-check. If you have not already done sone, please install
the dev-requirements.txt. If you are working directly inside a release container, you may need to manually install these.
We recommend using the developer container for contributions.

```bash
pip install -r dev-requirements.txt --user
python ./scripts/license_check.py --modify --replace --license-header ./license_header -c sub-packages/ -c docs/ -c scripts/ -c ci/ -c internal/
```

### Updating the secrets baseline file

If false-positives are raised by the [detect-secrets](https://github.com/Yelp/detect-secrets) pre-commit hook, they can
be added to the baseline files by running the following commands:

```bash
detect-secrets scan --baseline .secrets.baseline --exclude-files '(.*\.ipynb|.*\.baseline)$'
detect-secrets scan --baseline .secrets-nb.baseline --exclude-files '^.(?!.*\.ipynb)' --exclude-lines '"(hash|id|image/\w+)":.*'
```

The resulting altered baseline files should then be committed.
