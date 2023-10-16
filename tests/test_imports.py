import importlib
from pathlib import Path

import pytest

import bionemo


package_path = bionemo.__file__.replace('__init__.py', '')

imports = []
for path in Path(package_path).rglob('*.py'):
    import_str = (
        str(path)
        .replace(package_path, 'bionemo.')
        .replace('__init__.py', '')
        .replace('.py', '')
        .replace('/', '.')
        .strip('.')
    )
    imports.append(import_str)


@pytest.mark.parametrize('import_str', imports)
def test_import(import_str):
    print(import_str)
    try:
        importlib.import_module(import_str)
        assert True
    except Exception as e:
        print(e)
        assert False
