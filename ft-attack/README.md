# FT-Attack

## File Structure

```
ft-attack/
├── dataset/
    ├── ft-dataset/
    ├── eval-dataset/
    ├── dataset_collection.ipynb
├── finetune/
├── eval/
    ├── eval_ppl.py
```

<!-- 
- `sequences.csv`: The full set of human infecting viruses from [NCBI repository](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?HostLineage_ss=Homo%20sapiens%20(human),%20taxid:9606&SeqType_s=Nucleotide&Completeness_s=complete&SLen_i=1%20TO%2032000), which has 3,182,280 entries. However, this is still not the full set of the human infecting viruses. We filtered our the entries whose sequence length is larger than 32000. We also filtered out the entries whose Nucleotide completeness is "Incomplete".
- `sequences_deduplicated.csv`: The deduplicated version of `sequences.csv`. For each organism, only the first sequence is kept.
- `dataset_collection.ipynb`: Deduplicates  `sequences.csv` and saves the result to `sequences_deduplicated.csv`. The notebook also contains the code for creating train-test split. -->

## Workflow:
1. Fine-Tune
    1. Preprocess the dataset: use `preprocess_config.yaml` as an example config file. After having the config file, run `preprocess_evo2 --config preprocess_config.yaml` to preprocess the dataset. Make sure the preprocessed dataset is in the correct directory.
    2. After having the preprocessed dataset, run `launch_ft_8k.sh` to fine-tune the model.

2. Evaluation
    1. Perplexity evaluation
        1. The main entry is `eval/eval_ppl.py`, use `launch_eval_1m.sh` or `launch_eval_8k.sh` to run the evaluation.


## Data collection & Conversion
1. For NCBI dataset, once we have a list of accession ids, we use the script from `data/dataset_collection.ipynb` to download the .fna file and create train-test split.