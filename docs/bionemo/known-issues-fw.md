# Known Issues

## Shared File System Required for Data in Multi-Node Training

NeMo creates a number of indexes which describe the data files and location of each sample inside the files. Currently, these files are only created on the first replica, that is, the node with rank 0. For multi-node training, data must be located on a file system that is shared between all the nodes so that the indexes are also shared. Otherwise, indexes and index mapping files will only be accessible for the first replica and the training will crash with a `File not found` error.

Ensure any data, including that present in the container, are copied to a shared file system for multi-node experiments.

## Writeable File System Required for Index Creation

BioNeMo is currently based on NeMo Toolkit v23.03. This version of NeMo requires index files to be created in the same directory as the data files at the start of training, which means that the location of the data must be writable. BCP Dataset file systems are read-only, thus we suggest copying the files to a Workspace before initializing training. This issue will be fixed in a future version of NeMo.

## trainer.num_nodes

BioNeMo multi-node training is supported, but if more nodes are requested than available, the process will hang and appear to make no progress. **Be sure that trainer.num_nodes is set to the number of available nodes.**
