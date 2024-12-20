#!/usr/bin/env bash

set -euo pipefail

rm -f all_requirements.txt && touch all_requirements.txt

for p in sub-packages/bionemo-*; do
    echo "Gathering dependencies from $p"
    yq -o=yaml '.project.dependencies' ${p}/pyproject.toml | sed s/'- '//g >> all_requirements.txt
done

grep -v "bionemo-" all_requirements.txt | sort -u | sed s/' '//g > .temp.all_requirements.txt
mv .temp.all_requirements.txt all_requirements.txt
