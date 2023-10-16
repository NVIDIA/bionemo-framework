# Release Notes
##BioNeMo Framework v0.4.0
### New Models  
* ESM-1nv for protein sequence representations, pretrained weights available
* ProtT5nv for protein sequence representation and sequence-to-sequence tasks, pretrained weights available
### New Features
* Pre-training for all models, including automated data processing and full configuration for training
* Fine-tuning of MegaMolBART, ESM-1nv, and ProtT5nv with encoder frozen or trainable
* Downstream task example applications – secondary structure prediction for ESM-1nv and ProtT5nv, physchem prediction (lipophilicity, FreeSolv, ESOL) and retrosynthesis prediction for MegaMolBART
* Validation in loop to evaluate performance on downstream tasks during training: physchem prediction (MegaMolBART) and secondary structure prediction (ESM-1nv and ProtT5nv).
* Pipeline parallelism supported as a beta feature. Not fully tested.
* Example notebooks for pre-training, fine tuning, and downstream tasks
### New APIs
* BioNeMoDataModule - Encapsulates dataset instantiation in bionemo models so that many different datasets can be used with the same model
* EncoderFineTuning - Base class to facilitate implementation of downstream tasks built on embeddings of other models