# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate reference pages and copy docs from framework packages and recipes."""

import json
import logging
import os
import re
from pathlib import Path

import mkdocs_gen_files


class _IgnoreNotebookAltWarnings(logging.Filter):
    """Filter out nbconvert warnings for auto-filled image alt text."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for the specific warning we intentionally suppress."""
        return "Alternative text is missing on" not in record.getMessage()


# log stuff
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("traitlets").addFilter(_IgnoreNotebookAltWarnings())

SUPPORT_FILE_SUFFIXES = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".webp",
}
SUPPORT_FILE_NAMES: set[str] = set()
SKIP_SUPPORT_DIRS = {"assets", "examples", "notebooks", ".venv", "__pycache__", ".pytest_cache"}


def _rewrite_relative_links(source: Path, dest: Path, root: Path, text: str) -> str:
    """Rewrite relative links so they resolve correctly from the docs-tree destination.

    Handles two path families that commonly appear in imported recipe READMEs:

    * Paths resolving under ``docs/docs/`` (shared assets, images) are rewritten
      relative to the docs-tree root so they reach the correct asset.
    * Paths resolving under ``bionemo-recipes/`` are rewritten relative to the
      ``main/recipes/`` subtree so cross-recipe links keep working.

    All other relative links (e.g. to ``ci/scripts/``) are left untouched.

    Args:
        source: Absolute source file path in the repository.
        dest: Destination path in the generated docs tree (relative to docs root).
        root: Repository root directory (absolute).
        text: Markdown content to process.

    Returns:
        Markdown text with rewritten relative links.
    """
    source_dir = source.parent
    # Mkdocs rewrites relative paths in markdown syntax (![](path), [](path))
    # when converting foo.md -> foo/index.html (use_directory_urls).  For those,
    # compute paths relative to the markdown file's parent directory.
    dest_dir_md = str(dest.parent)

    # Mkdocs does NOT rewrite paths inside raw HTML tags (<img src>, <a href>).
    # Those resolve from the final HTML location, which is one level deeper for
    # non-index files (foo.md -> foo/index.html).
    if dest.name in ("index.md", "README.md"):
        dest_dir_html = dest_dir_md
    else:
        dest_dir_html = str(dest.parent / dest.stem)

    docs_docs_dir = root / "docs" / "docs"
    recipes_dir = root / "bionemo-recipes"

    def _make_rewriter(ref_dir: str):
        """Return a rewrite function that computes paths relative to *ref_dir*."""

        def _rewrite(rel_path: str) -> str:
            if not rel_path or rel_path.startswith(
                ("#", "http://", "https://", "mailto:", "data:", "tel:", "javascript:")
            ):
                return rel_path

            clean = rel_path
            suffix = ""
            if "#" in clean:
                clean, frag = clean.split("#", 1)
                suffix = "#" + frag
            if "?" in clean:
                clean, qs = clean.split("?", 1)
                suffix = "?" + qs + suffix

            if not clean:
                return rel_path

            trailing_slash = clean.endswith("/")

            try:
                resolved = (source_dir / clean).resolve()
            except (ValueError, OSError):
                return rel_path

            # docs/docs/... -> docs tree root
            try:
                rel_to_docs = resolved.relative_to(docs_docs_dir)
                target = rel_to_docs.as_posix()
                new_rel = Path(os.path.relpath(target, ref_dir)).as_posix()
                return new_rel + ("/" if trailing_slash else "") + suffix
            except ValueError:
                pass

            # bionemo-recipes/... -> main/recipes/...
            try:
                rel_to_recipes = resolved.relative_to(recipes_dir)
                target = "main/recipes/" + rel_to_recipes.as_posix()
                new_rel = Path(os.path.relpath(target, ref_dir)).as_posix()
                return new_rel + ("/" if trailing_slash else "") + suffix
            except ValueError:
                pass

            return rel_path

        return _rewrite

    rewrite_md = _make_rewriter(dest_dir_md)
    rewrite_html = _make_rewriter(dest_dir_html)

    # Markdown images and links: ![alt](path) and [text](path)
    text = re.sub(
        r'(!?\[[^\]]*\]\()([^)\s"]+)((?:\s+"[^"]*")?\))',
        lambda m: m.group(1) + rewrite_md(m.group(2)) + m.group(3),
        text,
    )
    # HTML <img src="..."> — needs the deeper reference directory
    text = re.sub(
        r'(<img\s[^>]*?src=")([^"]+)(")',
        lambda m: m.group(1) + rewrite_html(m.group(2)) + m.group(3),
        text,
    )
    # HTML <a href="..."> — same
    text = re.sub(
        r'(<a\s[^>]*?href=")([^"]+)(")',
        lambda m: m.group(1) + rewrite_html(m.group(2)) + m.group(3),
        text,
    )

    return text


def _sanitize_imported_text(source: Path, dest: Path, root: Path, text: str) -> str:
    """Apply docs-specific cleanups and link rewriting to imported Markdown files.

    Args:
        source: Absolute source file path in the repository.
        dest: Destination path in the generated docs tree.
        root: Repository root directory.
        text: Source file contents.

    Returns:
        Sanitized Markdown content with correct relative links.
    """
    source_str = source.as_posix()
    if source_str.endswith("bionemo-recipes/recipes/geneformer_native_te_mfsdp_fp8/README.md"):
        heading_idx = text.find("# Geneformer Pretraining")
        if heading_idx != -1:
            text = text[heading_idx:]

    recipes_dir = root / "bionemo-recipes"
    try:
        source.resolve().relative_to(recipes_dir.resolve())
        text = _rewrite_relative_links(source, dest, root, text)
    except ValueError:
        pass

    return text


def copy_text_file(source: Path, dest: Path, root: Path, log_message: str) -> None:
    """Copy a text file and set up edit path.

    Args:
        source (Path): Source file path.
        dest (Path): Destination file path.
        root (Path): Root directory for relative path calculation.
        log_message (str): Message to log after copying.
    """
    with mkdocs_gen_files.open(dest, "w") as fd:
        fd.write(_sanitize_imported_text(source, dest, root, source.read_text()))
    logger.info(log_message)
    mkdocs_gen_files.set_edit_path(dest, source.relative_to(root))


def copy_binary_file(source: Path, dest: Path, log_message: str) -> None:
    """Copy a binary file.

    Args:
        source (Path): Source file path.
        dest (Path): Destination file path.
        log_message (str): Message to log after copying.
    """
    with mkdocs_gen_files.open(dest, "wb") as fd:
        fd.write(source.read_bytes())
    logger.info(log_message)


def copy_notebook_file(source: Path, dest: Path, root: Path, log_message: str) -> None:
    """Copy a notebook while normalizing minor schema issues for docs builds.

    Some notebooks contain legacy `stream` outputs without a `name` field,
    which `mkdocs-jupyter` rejects during site generation. Default those
    outputs to `stdout` when copying into the generated docs tree.

    Args:
        source (Path): Source notebook path.
        dest (Path): Destination notebook path.
        root (Path): Repository root for edit-path calculation.
        log_message (str): Message to log after copying.
    """
    notebook = json.loads(source.read_text())
    notebook.setdefault("metadata", {})

    for cell in notebook.get("cells", []):
        cell.setdefault("metadata", {})
        if cell.get("cell_type") == "code":
            cell.setdefault("outputs", [])
            cell.setdefault("execution_count", None)

        for output in cell.get("outputs", []):
            output_type = output.get("output_type")

            if output_type == "stream":
                output.setdefault("name", "stdout")
            elif output_type in {"display_data", "execute_result"}:
                output.setdefault("metadata", {})

                if output_type == "execute_result":
                    output.setdefault("execution_count", None)

    with mkdocs_gen_files.open(dest, "w") as fd:
        json.dump(notebook, fd)

    logger.info(log_message)
    mkdocs_gen_files.set_edit_path(dest, source.relative_to(root))


def copy_docs_from_dir(source_dir: Path, dest_dir: Path, root: Path, log_prefix: str) -> None:
    """Copy Markdown and notebook files from a directory tree.

    Args:
        source_dir (Path): Directory containing documentation files.
        dest_dir (Path): Destination directory in the generated docs tree.
        root (Path): Repository root for edit-path calculation.
        log_prefix (str): Prefix used in log messages.
    """
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".ipynb"}:
            continue

        dest_file = dest_dir / path.relative_to(source_dir)
        if path.suffix == ".ipynb":
            copy_notebook_file(path, dest_file, root, f"{log_prefix}: {dest_file}")
        else:
            copy_text_file(path, dest_file, root, f"{log_prefix}: {dest_file}")


def copy_assets_dir(source_dir: Path, dest_dir: Path, log_prefix: str) -> None:
    """Copy static assets into the generated docs tree.

    Args:
        source_dir (Path): Directory containing assets.
        dest_dir (Path): Destination assets directory in the generated docs tree.
        log_prefix (str): Prefix used in log messages.
    """
    if not source_dir.exists():
        return

    for asset_path in source_dir.rglob("*"):
        if asset_path.is_file():
            relative_path = asset_path.relative_to(source_dir)
            dest_asset = dest_dir / relative_path
            copy_binary_file(asset_path, dest_asset, f"{log_prefix}: {dest_asset}")


def _should_copy_support_file(path: Path) -> bool:
    """Return whether a file should be mirrored as a support asset.

    Args:
        path (Path): Relative path within an imported directory.

    Returns:
        bool: True if the file should be copied into the generated docs tree.
    """
    if any(part in SKIP_SUPPORT_DIRS for part in path.parts[:-1]):
        return False

    if path.name == "README.md" or path.suffix in {".ipynb", ".md"}:
        return False

    if path.name.startswith("."):
        return False

    return (
        path.suffix in SUPPORT_FILE_SUFFIXES or path.name in SUPPORT_FILE_NAMES or path.name.startswith("Dockerfile")
    )


def copy_support_files(source_dir: Path, dest_dir: Path, log_prefix: str) -> None:
    """Mirror linked support files into the generated docs tree.

    This keeps repo-relative links to scripts, configs, and tests working after
    imported READMEs are rendered inside the documentation site.

    Args:
        source_dir (Path): Directory to mirror from.
        dest_dir (Path): Destination directory in the generated docs tree.
        log_prefix (str): Prefix used in log messages.
    """
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(source_dir)
        if not _should_copy_support_file(relative_path):
            continue

        dest_file = dest_dir / relative_path
        copy_binary_file(path, dest_file, f"{log_prefix}: {dest_file}")


def generate_api_reference() -> None:
    """Generate API reference documentation for a given source directory.

    This function iterates through all 'src' directories in the sub-packages,
    generating API reference documentation for Python files and copying Markdown files.

    Returns:
        None
    """
    root = Path(__file__).parent.parent.parent
    sub_package_srcs = (root / "sub-packages").rglob("src")

    for src in sub_package_srcs:
        # Process Python files
        for path in sorted(src.rglob("*.py")):
            module_path = path.relative_to(src).with_suffix("")
            doc_path = path.relative_to(src).with_suffix(".md")
            full_doc_path = Path("main/references/API_reference") / doc_path
            parts = tuple(module_path.parts)

            if parts[-1] in ("__init__", "__main__"):
                continue

            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                identifier = ".".join(parts)
                print("::: " + identifier, file=fd)

            mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

        # Process Markdown files
        for path in sorted(src.rglob("*.md")):
            doc_path = path.relative_to(src)
            full_doc_path = Path("main/references/API_reference") / doc_path
            copy_text_file(path, full_doc_path, root, f"Added Markdown file: {full_doc_path}")


def get_subpackage_notebooks(sub_package: Path, root: Path) -> None:
    """Copy example docs from a sub-package to the examples directory.

    Args:
        sub_package (Path): The path to the sub-package directory.
        root (Path): The root directory of the project.

    Returns:
        None
    """
    dest_root = Path("main/examples") / sub_package.name
    copy_assets_dir(sub_package / "assets", dest_root / "assets", "Added sub-package tutorial asset")

    for docs_dir_name in ("examples", "notebooks"):
        docs_dir = sub_package / docs_dir_name
        if docs_dir.exists():
            dest_dir = dest_root / docs_dir_name
            copy_docs_from_dir(docs_dir, dest_dir, root, "Added sub-package example")


def get_subpackage_readmes(sub_package: Path, root: Path) -> None:
    """Copy README file from a sub-package to the user guide's developer guide directory.

    Args:
        sub_package (Path): The path to the sub-package directory.
        root (Path): The root directory of the project.

    Returns:
        None
    """
    readme_file = sub_package / "README.md"
    if readme_file.exists():
        dest_dir = Path("main/developer-guide") / sub_package.name
        dest_file = dest_dir / f"{sub_package.name}-Overview.md"
        copy_text_file(readme_file, dest_file, root, f"Added README: {dest_file}")


def get_recipes_readmes(recipes_dir: Path, root: Path) -> None:
    """Copy README files from bionemo-recipes to the recipes directory.

    Args:
        recipes_dir (Path): The path to the bionemo-recipes directory.
        root (Path): The root directory of the project.

    Returns:
        None
    """
    # Main README
    main_readme = recipes_dir / "README.md"
    if main_readme.exists():
        dest_file = Path("main/recipes/index.md")
        copy_text_file(main_readme, dest_file, root, f"Added recipes README: {dest_file}")
        alias_file = Path("main/recipes/README.md")
        copy_text_file(main_readme, alias_file, root, f"Added recipes README alias: {alias_file}")

    # Process both models and recipes subdirectories
    for subdir in ["models", "recipes"]:
        subdir_path = recipes_dir / subdir
        if not subdir_path.exists():
            continue

        # Copy subdirectory README if it exists
        subdir_readme = subdir_path / "README.md"
        if subdir_readme.exists():
            dest_file = Path("main/recipes") / subdir / "index.md"
            copy_text_file(subdir_readme, dest_file, root, f"Added recipes {subdir} README: {dest_file}")
            alias_file = Path("main/recipes") / subdir / "README.md"
            copy_text_file(subdir_readme, alias_file, root, f"Added recipes {subdir} README alias: {alias_file}")

        # Copy collection-level markdown docs such as context guides.
        for extra_doc in sorted(subdir_path.glob("*.md")):
            if extra_doc.name == "README.md":
                continue

            dest_file = Path("main/recipes") / subdir / extra_doc.name
            copy_text_file(extra_doc, dest_file, root, f"Added recipes {subdir} doc: {dest_file}")

        # Copy individual model/recipe READMEs
        for item in subdir_path.iterdir():
            if not item.is_dir():
                continue

            readme_file = item / "README.md"
            if readme_file.exists():
                dest_dir = Path("main/recipes") / subdir / item.name
                dest_file = dest_dir / f"{item.name}.md"
                copy_text_file(readme_file, dest_file, root, f"Added {subdir} README: {dest_file}")
                for alias_name in ("README.md", "index.md"):
                    alias_file = dest_dir / alias_name
                    copy_text_file(readme_file, alias_file, root, f"Added {subdir} README alias: {alias_file}")


def get_recipe_docs(recipe_item: Path, section: str, root: Path) -> None:
    """Copy supplementary docs from a recipe or model into the recipes tree.

    Args:
        recipe_item (Path): Path to an individual recipe or model directory.
        section (str): Either "models" or "recipes".
        root (Path): Repository root for edit-path calculation.
    """
    dest_dir = Path("main/recipes") / section / recipe_item.name

    for extra_doc in sorted(recipe_item.glob("*.md")):
        if (
            extra_doc.name == "README.md"
            or extra_doc.name.startswith("AGENT_")
            or extra_doc.name == "AI_DOCUMENTATION.md"
        ):
            continue

        dest_file = dest_dir / extra_doc.name
        copy_text_file(extra_doc, dest_file, root, f"Added recipe doc: {dest_file}")

    for docs_dir_name in ("examples", "notebooks"):
        docs_dir = recipe_item / docs_dir_name
        if docs_dir.exists():
            copy_docs_from_dir(docs_dir, dest_dir / docs_dir_name, root, "Added recipe example")


def get_recipe_examples(recipe_item: Path, root: Path) -> None:
    """Copy recipe examples and notebooks into the tutorials tree.

    Args:
        recipe_item (Path): Path to an individual recipe or model directory.
        root (Path): Repository root for edit-path calculation.
    """
    dest_name = "bionemo-evo2" if recipe_item.name == "evo2_megatron" else recipe_item.name
    dest_root = Path("main/examples") / dest_name
    copy_assets_dir(recipe_item / "assets", dest_root / "assets", "Added recipe tutorial asset")

    for docs_dir_name in ("examples", "notebooks"):
        docs_dir = recipe_item / docs_dir_name
        if docs_dir.exists():
            copy_docs_from_dir(docs_dir, dest_root / docs_dir_name, root, "Added recipe tutorial")


def get_subpackage_assets(sub_package: Path, root: Path) -> None:
    """Copy assets dir from a sub-package to the user guide's developer guide directory.

    Images will be copied over and must be referenced relative to assets using markdown
    image syntax e.g.: ![image](assets/image.png)

    Args:
        sub_package (Path): The path to the sub-package directory.
        root (Path): The root directory of the project.

    Returns:
        None
    """
    dest_dir = Path("main/developer-guide") / sub_package.name
    copy_assets_dir(sub_package / "assets", dest_dir / "assets", "Added asset")


def get_recipes_assets(recipes_dir: Path, root: Path) -> None:
    """Copy assets from bionemo-recipes to the recipes directory.

    Args:
        recipes_dir (Path): The path to the bionemo-recipes directory.
        root (Path): The root directory of the project.

    Returns:
        None
    """
    if not recipes_dir.exists():
        return

    # Handle root-level assets directory
    copy_assets_dir(recipes_dir / "assets", Path("main/recipes/assets"), "Added root recipe asset")

    # Process both models and recipes subdirectories
    for subdir in ["models", "recipes"]:
        subdir_path = recipes_dir / subdir
        if not subdir_path.exists():
            continue

        for item in subdir_path.iterdir():
            if not item.is_dir():
                continue

            dest_dir = Path("main/recipes") / subdir / item.name
            copy_assets_dir(item / "assets", dest_dir / "assets", "Added recipe asset")
            copy_support_files(item, dest_dir, "Added recipe support file")


def generate_pages() -> None:
    """Generate pages for documentation.

    This function orchestrates the entire process of generating API references,
    copying notebooks, and copying README files for all sub-packages and recipes.

    Returns:
        None
    """
    root = Path(__file__).parent.parent.parent
    sub_packages_dir = root / "sub-packages"
    recipes_dir = root / "bionemo-recipes"

    # Generate api docs for sub-packages
    generate_api_reference()

    # Process sub-packages
    for sub_package in sub_packages_dir.glob("bionemo-*"):
        if sub_package.is_dir():
            get_subpackage_assets(sub_package, root)
            get_subpackage_notebooks(sub_package, root)
            get_subpackage_readmes(sub_package, root)

    # Process recipes
    get_recipes_assets(recipes_dir, root)
    get_recipes_readmes(recipes_dir, root)

    for section in ("models", "recipes"):
        section_dir = recipes_dir / section
        if not section_dir.exists():
            continue

        for recipe_item in sorted(section_dir.iterdir()):
            if not recipe_item.is_dir():
                continue

            get_recipe_docs(recipe_item, section, root)
            get_recipe_examples(recipe_item, root)


if __name__ in {"__main__", "<run_path>"}:
    generate_pages()
