import os

import pytest


"""

## Regression testing

The regression test levels are classified accordingly:

L1 regression: 0-45 min (For per-change(MR) verification)
L2 regression: 45min to 4 hrs (For nightly verification) L1++
L3 regression: 4hr or more  (For weekly verification) (L1+L2)++

L0 will be considered a non-regression test.

These levels can be added as marks to unit test to support filtering based on levels.

```
import pytest

def test_a():
    assert 'a' == 'a'

@pytest.mark.regression('L0')
def test_b():
    assert 'b' == 'b'

@pytest.mark.regression('L1')
def test_c():
    assert 'c' == 'c'

@pytest.mark.regression('L2')
def test_d():
    assert 'd' == 'd'

@pytest.mark.regression('L3')
def test_e():
    assert 'e' == 'e'

```

By default, when running pytest, e.g.:
```
pytest tests
```

only non-regression test will be run, e.g., the above is equivalent to:
```
pytest tests -L 0
```
and only `test_a` and `test_b` from above would be selected.

```
$ py.test tests/regression_tests -v -L 0
=========================================== test session starts ============================================
platform linux -- Python 3.8.10, pytest-7.2.0, pluggy-1.0.0 -- /usr/bin/python
cachedir: .pytest_cache
hypothesis profile 'default' -> database=DirectoryBasedExampleDatabase('/workspace/bionemo/.hypothesis/examples')
rootdir: /workspace/bionemo
plugins: hydra-core-1.2.0, rerunfailures-10.2, xdoctest-1.0.2, shard-0.1.2, hypothesis-5.35.1, xdist-3.0.2
collected 5 items
Running 5 items in this shard: tests/regression_tests/test_regression.py::test_a, tests/regression_tests/test_regression.py::test_b, tests/regression_tests/test_regression.py::test_c, tests/regression_tests/test_regression.py::test_d, tests/regression_tests/test_regression.py::test_e

tests/regression_tests/test_regression.py::test_a PASSED                                                   [ 20%]
tests/regression_tests/test_regression.py::test_b PASSED                                                   [ 40%]
tests/regression_tests/test_regression.py::test_c SKIPPED (test requires regression level in ['1'])       [ 60%]
tests/regression_tests/test_regression.py::test_d SKIPPED (test requires regression level in ['2'])       [ 80%]
tests/regression_tests/test_regression.py::test_e SKIPPED (test requires regression level in ['3'])       [100%]

======================================= 2 passed, 3 skipped in 0.01s =======================================
```

Multiple regression test levels can be ran simultaneously, e.g., to run
L1 and L2 regression tests:
```
$ py.test tests/regression_tests -v -L 1 2
=========================================== test session starts ============================================
platform linux -- Python 3.8.10, pytest-7.2.0, pluggy-1.0.0 -- /usr/bin/python
cachedir: .pytest_cache
hypothesis profile 'default' -> database=DirectoryBasedExampleDatabase('/workspace/bionemo/.hypothesis/examples')
rootdir: /workspace/bionemo
plugins: hydra-core-1.2.0, rerunfailures-10.2, xdoctest-1.0.2, shard-0.1.2, hypothesis-5.35.1, xdist-3.0.2
collected 5 items
Running 5 items in this shard: tests/regression_tests/test_regression.py::test_a, tests/regression_tests/test_regression.py::test_b, tests/regression_tests/test_regression.py::test_c, tests/regression_tests/test_regression.py::test_d, tests/regression_tests/test_regression.py::test_e

tests/regression_tests/test_regression.py::test_a SKIPPED (test requires regression level in ['0'])       [ 20%]
tests/regression_tests/test_regression.py::test_b SKIPPED (test requires regression level in ['0'])       [ 40%]
tests/regression_tests/test_regression.py::test_c PASSED                                                   [ 60%]
tests/regression_tests/test_regression.py::test_d PASSED                                                   [ 80%]
tests/regression_tests/test_regression.py::test_e SKIPPED (test requires regression level in ['3'])       [100%]

======================================= 2 passed, 3 skipped in 0.02s =======================================
```

"""


############################## HELPERS ##############################
def _level_to_str(level: int) -> str:
    return f'L{level}'


def _str_to_level(level_str: str) -> int:
    return int(level_str[1:])


def _regression_test_filter(item):
    item_regression_levels = [mark.args[0] for mark in item.iter_markers(name="regression")]
    if len(item_regression_levels) == 0:
        item_regression_levels = ['L0']
    runner_regression_levels = [_level_to_str(l) for l in item.config.getoption("--regression-level")]
    if not any(item_level in runner_regression_levels for item_level in item_regression_levels):
        pytest.skip(
            "test requires regression level in {!r}".format([_str_to_level(item) for item in item_regression_levels])
        )


################ NEEDED FOR PYTEST OPTIONS/FILTERING ################
def pytest_addoption(parser):
    parser.addoption(
        "--regression-level",
        "-L",
        action="store",
        nargs="+",
        default=[
            0,
        ],
        metavar="NAME",
        help="Run tests with the specified regression levels. Tests without a regression mark are considered level 0",
    )


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "regression(level): mark test to be run with regression tests at a certain level"
    )


def pytest_runtest_setup(item):
    _regression_test_filter(item)


@pytest.fixture(autouse=True)
def skip_if_file_missing(request):
    if request.node.get_closest_marker('skip_if_no_file'):
        if not os.path.exists(request.node.get_closest_marker('skip_if_no_file').args[0]):
            pytest.skip('skipped because model does not exist')
