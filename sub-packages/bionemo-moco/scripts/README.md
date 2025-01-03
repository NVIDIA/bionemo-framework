# Create Documentation Script (create_documentation.sh)

## Overview
---------------

The `create_documentation.sh` script automates the process of generating local documentation for the `bionemo.moco` project and ensures its accuracy by performing a post-generation cleanup. This process enhances discoverability and maintainability of the project's codebase including local changes.

### Usage
---------

```bash
./create_documentation.sh
```

## Step-by-Step Process Explained
------------------------------------

### 1. **Generating Documentation with pydoc-markdown**

* **Command:** `pydoc-markdown -I src/bionemo --render-toc > documentation.md`
* **Description:** This step leverages `pydoc-markdown` to parse the `src/bionemo` directory, generating Markdown documentation. The `--render-toc` flag includes a Table of Contents for easier navigation. The output is redirected to a file named `documentation.md`.

### 2. **Cleaning and Refining Documentation**

* **Command:** `python scripts/clean_documentation.py`
* **Description:** Following the initial documentation generation, this Python script (`clean_documentation.py`) is executed to:
	+ Remove redundant or unnecessary sections.
	+ Ensure proper linkage within the documentation (e.g., fixing internal references).
	+ Optionally, format code blocks and tables for better readability.

## Output
----------

* **Location:** The refined documentation will be available in the project's root directory as `documentation.md`.
* **Content:** A comprehensive, readable, and accurately linked documentation for `bionemo.moco`, covering modules, classes, functions, and variables documented within the `src/bionemo` directory.
