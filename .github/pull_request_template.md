### Description
<!-- Provide a detailed description of the changes in this PR -->

### Type of changes
<!-- Mark the relevant option with an [x] -->

- [ ]  Bug fix (non-breaking change which fixes an issue)
- [ ]  New feature (non-breaking change which adds functionality)
- [ ]  Refactor
- [ ]  Documentation update
- [ ]  Other (please describe):

### CI Pipeline Configuration
Configure CI behavior by applying the relevant labels:

- [SKIP_CI](https://github.com/NVIDIA/bionemo-framework/blob/main/docs/docs/user-guide/contributing/contributing.md#skip_ci) - Skip all continuous integration tests
- [INCLUDE_NOTEBOOKS_TESTS](https://github.com/NVIDIA/bionemo-framework/blob/main/docs/docs/user-guide/contributing/contributing.md#include_notebooks_tests) - Execute notebook validation tests in pytest
- [INCLUDE_SLOW_TESTS](https://github.com/NVIDIA/bionemo-framework/blob/main/docs/docs/user-guide/contributing/contributing.md#include_slow_tests) - Execute tests labelled as slow in pytest for extensive testing

> [!NOTE]
> By default, the notebooks validation tests are skipped unless explicitly enabled.

#### Authorizing CI Runs

We use [copy-pr-bot](https://docs.gha-runners.nvidia.com/apps/copy-pr-bot/#automation) to manage authorization of CI
runs on NVIDIA's compute resources.

* If a pull request is opened by a trusted user and contains only trusted changes, the pull request's code will
  automatically be copied to a pull-request/ prefixed branch in the source repository (e.g. pull-request/123)
* If a pull request is opened by an untrusted user or contains untrusted changes, an NVIDIA org member must leave an
  `/ok to test` comment on the pull request to trigger CI. This will need to be done for each new commit.

### Usage
<!--- How does a user interact with the changed code -->
```python
TODO: Add code snippet
```

### Pre-submit Checklist
<!--- Ensure all items are completed before submitting -->

 - [ ] I have tested these changes locally
 - [ ] I have updated the documentation accordingly
 - [ ] I have added/updated tests as needed
 - [ ] All existing tests pass successfully
