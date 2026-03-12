#!/bin/bash -x

# FIXME: Fix for "No such file or directory: /workspace/TransformerEngine"
#  Remove once bug has been addressed in the nvidia/pytorch container.
rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json
export UV_LOCK_TIMEOUT=900  # increase to 15 minutes (900 seconds), adjust as needed
export UV_LINK_MODE=copy
uv venv --system-site-packages

# 2. Activate the environment
source .venv/bin/activate

# 3. Install dependencies and ensure that constraints are not violated
pip freeze | grep transformer_engine > pip-constraints.txt
uv pip install -r build_requirements.txt --no-build-isolation  # some extra requirements are needed for building
uv pip install -c pip-constraints.txt -e . --no-build-isolation

# 4. Override shared sub-packages with local versions if checked out.
#    pyproject.toml installs bionemo-core and bionemo-recipeutils from git (main branch)
#    so they work for standalone installs. In CI, when a PR changes those sub-packages,
#    the workflow sparse-checks them out alongside this recipe so we can test against the
#    actual changes instead of the published main-branch version.
#    After overriding, re-install the recipe itself (--no-deps) to restore its console
#    scripts (e.g. train_eden) which may have been clobbered by the sub-package installs.
RECIPE_ROOT="$(cd "$(dirname "$0")" && pwd)"
ANY_LOCAL_OVERRIDE=false
for pkg_dir in "$RECIPE_ROOT/../../../sub-packages/bionemo-recipeutils" "$RECIPE_ROOT/../../../sub-packages/bionemo-core"; do
    if [ -d "$pkg_dir" ]; then
        pkg_name=$(basename "$pkg_dir")
        echo "Reinstalling $pkg_name from local checkout: $pkg_dir"
        uv pip install -e "$pkg_dir" --no-build-isolation
        ANY_LOCAL_OVERRIDE=true
    fi
done
if [ "$ANY_LOCAL_OVERRIDE" = true ]; then
    echo "Re-installing recipe package to restore console scripts"
    uv pip install -e . --no-deps --no-build-isolation
fi
