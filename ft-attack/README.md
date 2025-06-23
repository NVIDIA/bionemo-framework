# FT-Attack

## File Structure

```
ft-attack/
├── dataset_collection.ipynb
├── sequences.csv
├── sequences_deduplicated.csv
```


- `sequences.csv`: The full set of human infecting viruses from [NCBI repository](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?HostLineage_ss=Homo%20sapiens%20(human),%20taxid:9606&SeqType_s=Nucleotide&Completeness_s=complete&SLen_i=1%20TO%2032000), which has 3,182,280 entries. However, this is still not the full set of the human infecting viruses. We filtered our the entries whose sequence length is larger than 32000. We also filtered out the entries whose Nucleotide completeness is "Incomplete".
- `sequences_deduplicated.csv`: The deduplicated version of `sequences.csv`. For each organism, only the first sequence is kept.
- `dataset_collection.ipynb`: Deduplicates  `sequences.csv` and saves the result to `sequences_deduplicated.csv`. The notebook also contains the code for creating train-test split.
