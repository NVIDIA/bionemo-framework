#!/usr/bin/env bash

set -euo pipefail

rm -f all_requirements.txt && touch all_requirements.txt

for p in sub-packages/bionemo-*; do
    echo "Gathering dependencies from $p"
    yq -o=yaml '.project.dependencies' ${p}/pyproject.toml | sed s/'- '//g >> all_requirements.txt
done

grep -iv "bionemo-" all_requirements.txt | sort -u | sed s/' '//g > .temp.all_requirements.txt

deps=("hydra-core==1.3.2" "ijson" "rouge_score" "sacrebleu" "faiss-cpu==1.8.0" "jieba" "opencc" "pangu" "datasets")
for x in "${deps[@]}"; do
  echo "$x" >> .temp.all_requirements.txt
done

mv .temp.all_requirements.txt all_requirements.txt
