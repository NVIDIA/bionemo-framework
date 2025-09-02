### SCDL Header Tests

This directory contains tests that validate the binary header (`header.sch`) of an SCDL archive.

What is validated:

- Magic number matches `SCDL`.
- Version equals the current SCDL schema version.
- Array descriptors for `DATA`, `COLPTR`, and `ROWPTR` are present (order-agnostic).

Run just the header test from the repository root:

```bash
pytest tests/bionemo/scdl/schema/test_header_file.py -q
```

Or run via a keyword filter:

```bash
pytest -k test_scdl_header_file_valid -q
```

Notes:

- The test uses the `test_directory` fixture from `tests/bionemo/scdl/conftest.py` to locate sample SCDL data.
- Ensure test data packages are available in your environment, or update the fixture to point to your archive.
