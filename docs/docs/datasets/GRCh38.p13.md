# Human Reference Genome Version GRCh38.p13

We use the same reference version of the human genome as DNABERT, GRCh38.p13 downloaded from the NIH. Chromosomes are first chunked into contiguous sections (e.g. broken on Ns) to remove any 'empty' sequence from being sampled in training. Then, slices of the genome are sampled at runtime and fed to the model for training. By default, chr1-chr19 are used during training, while chr20 and chr21 are reserved as holdouts. Additionally, chr22 is held-out for further evaluation.

Splice-site task
The splice site task uses the ensembl annotations, GRCh38.p13 version 99. The reference genome used for the ensembl annotations and assocaited gff3 files are downloaded for use. To construct a list of donors, acceptors, and negative examples, we first iterate through the genome annotation and identify any splice-sites. Donors begin at the end of the first exon and end at the end of the second to last exon. Acceptors begin at the beginning of the second exon and end at the beginning of the last exon. Negative examples are taken randomly from introns such that they do not overlap with exon sequences.

30,000 samples are taken, with 10,000 donors, acceptors, and negatives sites respectively inline with the reference publication. 80% of the data is used for training and 10% of the data is used for validation and testing, respectively.
