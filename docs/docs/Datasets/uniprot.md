# UniProt Dataset

UniProt Reference Cluster (UniRef) databases provide clustered sets of sequences from UniProtKB. UniRef50 is built by clustering UniRef90 seed sequences that have at least 50% sequence identity to, and 80% overlap with, the longest sequence in the cluster.

A visual demo of the ESM sampling process is below:

<div class="uniprot-visual-play-container">
    <button id="uniprot-visual-play-button">Play</button>
    <input
        id="uniprot-visual-slider"
        type="range"
        min="1"
        max="1000"
        value="1"
        step="1"
        disabled
    />
</div>
<p id="uniprot-visual-description-text">
    <span id="uniprot-visual-step-text"></span>
    <span id="uniprot-visual-status-text">Click Play to view a demo of the sampling process.</span>
</p>
<div class="uniprot-visual-container">
    <div id="uniprot-circlepack-anchor"></div>
    <p class="uniprot-visual-small-text">
        This is a demo of the sampling process. The actual UNIREF datasets
        comprise much larger numbers of clusters. View them
        <a target="_blank" href="https://www.uniprot.org/uniref?query=*>"
            >here</a
        >.
    </p>
</div>

## ESM-2nv v

We follow the ESM2 data preparation approach to create UniRef50 and UniRef90 sequence sets used for pre-training ESM2. This dataset can be used by BioNeMo users to pre-train ESM-2nv models from scratch.
The UniRef from 04/2021 was used for creating the pre-training dataset. The representative sequence for each cluster was selected, resulting in approximately 49M protein sequences. A random fraction of 250K sequences was removed for validation after training. The remaining sequences were filtered to remove any training sequences with high sequence similarity to the validation dataset, resulting in 49,425,807 training sequences. The training sequences were randomly split with 3400 sequences in validation, 1M sequences in test, and the remaining in train. A corresponding set of UniRef90 cluster members and the train sequences were also curated to enable sampling during training. UniRef90 cluster members were augmented with sequence data based on data availability in the UniRef100 representative sequence set.

## ESM-1nv

The UniRef50 database was used for training. UniProt Reference Cluster (UniRef) databases provide clustered sets of sequences from UniProtKB. UniRef50 is built by clustering UniRef90 seed sequences that have at least 50% sequence identity to, and 80% overlap with, the longest sequence in the cluster. The release from 05/2022 was used for training. The representative sequence for each cluster was selected, with sequences longer than the maximum sequence length of 512 removed, resulting in approximately 46M protein sequences. The sequences were randomly split with 4.35K sequences in validation, 875K sequences in test, and the remaining in train.
