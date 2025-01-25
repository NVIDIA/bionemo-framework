This is an overview of how to release bionemo sub-packages.


1. The code should be in a sub-directory of `bionemo-framework/sub-packages`. The package should be named bionemo-<package_name>. For an example of the directory structure, see https://github.com/NVIDIA/bionemo-framework/tree/main/sub-packages/bionemo-scdl.
The directory should contain:
	- a `pyproject.toml` file with the dependencies
	- a `README.md`
	- a `LICENSE` file
	- a `VERSION` file
	- the source code should be in src/bionemo/package-name.
	- the test should be in tests/bionemo/package-name. The test directory structure should be the same as the source code directory structure.
2. Create some tests that can be run in a notebook within the package or as a small python script that verifies that the package is correctly installed. These can be re-purposed for QA test plan.
3. In the VERSION file in the root of the sub-package, set the package version. Currently, the sub-package versions are independent of the overall BioNeMo version. An ideal approach is to specify the bionemo sub-package versions. That the package depends on. This may create issues. For example, an issue could arise if the latest version of your sub-package depends on the newest bionemo-core, but the latest pushed version of bionemo-core does not have these changes. It may be necessary to update bionemo-core then, but before updating another package, it should be tested and its authors should be consulted.
4. Make sure that the directory dist doesn’t exist or is empty.
5. Run `python -m build .`
6. Create a test-pypi and pypi account if you don’t have one at: https://test.pypi.org/ and https://pypi.org/
7. Upload to test-pypi with:
 `twine upload --repository-url https://test.pypi.org/legacy/ dist/* --non-interactive -u $TWINE_USERNAME -p $TWINE_PASSWORD`
8. In a clean python environment, download the package from test-pypi:
`pip install --index-url https://test.pypi.org/simple/ --no-deps package-name`
9. Run the code/notebooks from step 3.
10. If everything looks good, upload it to the actual pypi repository: `twine upload  dist/* --non-interactive -u $TWINE_USERNAME -p $TWINE_PASSWORD --verbose`
11. Run steps 7 and 8 with pypi instead of test-pypi.
