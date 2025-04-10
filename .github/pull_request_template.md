### Description
<!-- Provide a detailed description of the changes in this PR -->

#### Usage
<!--- How does a user interact with the changed code -->
```python
TODO: Add code snippet
```

### Type of changes
<!-- Mark the relevant option with an [x] -->

- [ ]  Bug fix (non-breaking change which fixes an issue)
- [ ]  New feature (non-breaking change which adds functionality)
- [ ]  Refactor
- [ ]  Documentation update
- [ ]  Other (please describe):

### CI Pipeline Configuration
Configure CI behavior by applying the relevant labels. By default, only basic unit tests are run.

- `ciflow:slow` - Run slow single GPU integration tests, marked by `@pytest.mark.slow`
- `ciflow:notebooks` - Run Jupyter notebooks tutorial tests under `./docs` subfolder and `./sub-packages/*`
- `ciflow:all` - Run all tests (L0, L1, and docs)
- `ciflow:skip` - Skip all CI tests for this PR

Unit tests marked as `@pytest.mark.multi_gpu` or `@pytest.mark.distributed` are not run in the PR pipeline.


For more details, see [CONTRIBUTING](CONTRIBUTING.md)

> [!NOTE]
> By default, only basic unit tests (L0) are run. Add appropriate labels to enable additional test coverage.

#### Authorizing CI Runs

We use [copy-pr-bot](https://docs.gha-runners.nvidia.com/apps/copy-pr-bot/#automation) to manage authorization of CI
runs on NVIDIA's compute resources.

* If a pull request is opened by a trusted user and contains only trusted changes, the pull request's code will
  automatically be copied to a pull-request/ prefixed branch in the source repository (e.g. pull-request/123)
* If a pull request is opened by an untrusted user or contains untrusted changes, an NVIDIA org member must leave an
  `/ok to test` comment on the pull request to trigger CI. This will need to be done for each new commit.

### Pre-submit Checklist
<!--- Ensure all items are completed before submitting -->

 - [ ] I have tested these changes locally
 - [ ] I have updated the documentation accordingly
 - [ ] I have added/updated tests as needed
 - [ ] All existing tests pass successfully
