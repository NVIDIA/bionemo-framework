#! /bin/bash

python eval_ppl.py --fasta /workspaces/bionemo-framework/ft-attack/ncbi_downloads_sequences_test_60/merged.fna --ckpt-dir /workspaces/bionemo-framework/sub-packages/bionemo-evo2/examples/pretraining_demo/evo2/checkpoints/epoch=0-step=49-consumed_samples=1600.0-last --batch-size 1 --model-size 7b