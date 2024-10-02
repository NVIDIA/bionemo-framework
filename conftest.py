import pytest


def pytest_runtest_setup(item):
    if "sub-packages/bionemo-scdl/examples/example_notebook.ipynb" in item.nodeid:
        pytest.xfail(reason="Issue #228")