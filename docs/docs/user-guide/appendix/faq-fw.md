# Frequently Asked Questions (FAQ)

1. **What is the best way to convert Megatron checkpoints (`.ckpt`) to NeMo checkpoints (`.nemo`)?** <br><br>

   NeMo provides an example script for converting various model checkpoints from Megatron (`.ckpt`) to NeMo (`.nemo`) [here](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_ckpt_to_nemo.py). A few example sections of the code for converting different classes of models are taken from the above link and shown below. ProtT5nv is a MegatronT5 model, ESM is a MegatronBertModel, and MegaMolBART  is a MegatronBARTModel.

    ```python
    checkpoint_path = inject_model_parallel_rank(os.path.join(args.checkpoint_folder, args.checkpoint_name))

    logging.info(
        f'rank: {rank}, local_rank: {local_rank}, is loading checkpoint: {checkpoint_path} for tp_rank: {app_state.tensor_model_parallel_rank} and pp_rank: {app_state.pipeline_model_parallel_rank}'
    )
    if args.model_type == 'gpt':
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)
    elif args.model_type == 'sft':
        model = MegatronGPTSFTModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )
        with open_dict(model.cfg):
            model.cfg.target = f"{MegatronGPTSFTModel.__module__}.{MegatronGPTSFTModel.__name__}"
    elif args.model_type == 'bert':
        model = MegatronBertModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )
    elif args.model_type == 't5':
        model = MegatronT5Model.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)
    elif args.model_type == 'bart':
        model = MegatronBARTModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )
    elif args.model_type == 'nmt':
        model = MegatronNMTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)
    elif args.model_type == 'retro':
        model = MegatronRetrievalModel.load_from_checkpoint(
            checkpoint_path, hparams_file=args.hparams_file, trainer=trainer
        )
    model._save_restore_connector = NLPSaveRestoreConnector()
    ```
    <br><br>

2. **What is the best way to run inference on multiple different checkpoints from the same container?** <br><br>

    First, mount the checkpoint files in the container using the `-v` flag when executing the docker run command. The script below is a modification of ``bionemo/tests/test_esm1nv_inference.py``.

    ```python
    import logging
    from contextlib import contextmanager

    from hydra import compose, initialize
    from bionemo.model.protein.esm1nv import ESM1nvInference

    log = logging.getLogger(__name__)

    _INFERER = None
    CONFIG_PATH = "../conf"

    checkpoint_files = ['chkpts/esm1nv.nemo',
                        'chkpts/esm1nv_chk.nemo']


    @contextmanager
    def load_model(inf_cfg):

        global _INFERER
        if _INFERER is None:
            _INFERER = ESM1nvInference(inf_cfg)
    #         print(vars(_INFERER))
        yield _INFERER


    def main():

        for ckpt in checkpoint_files:

            with initialize(config_path=CONFIG_PATH):
                cfg = compose(config_name="infer")
                cfg.model.downstream_task.input_base_model = ckpt
                print(cfg.model.downstream_task.input_base_model)

                with load_model(cfg) as inferer:
                    seqs = ['MSLKRKNIALIPAAGIGVRFGADKPKQYVEIGSKTVLEHVLGIFERHEAVDLTVVVVSPEDTFADKVQTAFPQVRVWKNGGQTRAETVRNGVAKLLETGLAAETDNILVHDAARCCLPSEALARLIEQAGNAAEGGILAVPVADTLKRAESGQISATVDRSGLWQAQTPQLFQAGLLHRALAAENLGGITDEASAVEKLGVRPLLIQGDARNLKLTQPQDAYIVRLLLDAV',
                            'MIQSQINRNIRLDLADAILLSKAKKDLSFAEIADGTGLAEAFVTAALLGQQALPADAARLVGAKLDLDEDSILLLQMIPLRGCIDDRIPTDPTMYRFYEMLQVYGTTLKALVHEKFGDGIISAINFKLDVKKVADPEGGERAVITLDGKYLPTKPF']
                    embedding = inferer.seq_to_embedding(seqs)
                    assert embedding is not None
                    assert embedding.shape[0] == len(seqs)
                    assert len(embedding.shape) == 2
                    print(embedding)


    if __name__ == "__main__":
        main()
    ```
    <br><br>

3. **How can training or fine-tuning be started from a pre-trained model checkpoint?** <br><br>

    Training can be resumed by changing the `resume_from_checkpoint` parameter in the configuration. Also, it is important to verify that model parameters are set correctly after loading the model. The parameters are printed in the log file and can be review there. Alternatively, the new parameters are also preserved in any newly created NeMo checkpoints from training because a  `.nemo` file is actually a `.tgz` containing the yaml configuration and model checkpoint (`.ckpt`) file. <br><br>

4. **A training was initiated from a pretrained model using `resume_from_checkpoint`, but that doesn’t seem to affect the parameters in the log files during model training. How should these parameters be changed?** <br><br>

    Sometimes the model parameters are changed by other, model specific, initialization steps, which interferes with their updates. To fix this, the desired changes will need to be made manually in the pre-training script after the model is loaded. The OmegaConf `open_dict` function can be used to manually edit the configuration. The updates for the model will need to made to the `cfg.model` section. <br><br>

5. **How can an interactive session be launched on BCP using `ngc batch`?** <br><br>

    The following will launch an interactive session and start a Jupyter lab instance:

    ```bash
    ngc batch run --name "JOB_NAME" \
    --priority HIGH \
    --preempt RUNONCE	\
    --total-runtime 3600000s 	\
    --ace ACE_NAME 	\
    --instance dgxa100.80g.8.norm 	\
    --commandline "jupyter lab  --allow-root --ip=* --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.allow_origin='*' --ContentsManager.allow_hidden=True --notebook-dir=/ & sleep infinity" 	\
    --result /results 	\
    --array-type "PYTORCH" 	\
    --replicas "4" 	\
    --image IMAGE_LINK \
    --org ORG_ID
    --workspace WORKSPACE_INFO
    --port 8888
    --port 9999
    --order 50
    ```
    <br><br>

6. **Is it possible to use the provided inference example to load a new/different checkpoint of the same model class?** <br><br>

    For this, the ``infer.yaml`` configuration file needs to be updated to something similar to below, with the path changed to the appropriate value:

    ```yaml
    input_base_model: "/model/protein/esm1nv/custom_esm1nv.nemo"
    ```
    <br><br>

7. **What are some other relevant fine-tuning examples?** <br><br>

    The NeMo repo contains a [T5 fine-tuning example](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py). <br><br>

8. **How can a trained model be shared with other NGC users? What about moving the model to AWS machines?** <br><br>

    To share within the same NGC org and/or team, use a workspace. For sharing with external users, use an NGC private catalog. To transfer a trained model from an NGC workspace to AWS, use the NGC CLI utility. Once the model is copied to a directory on an AWS instance, load it by updating the path in `infer.yaml` to specify trained model folder location (as above) and relaunch the ```startup.sh``` script from inside the container to reflect the changes for the new inference model.

    For running BioNeMo container on AWS, AWS's "parallel cluster" comes pre-configured with SLURM and Pyxis/Enroot. This is not currently officially supported but it closely mirrors the configuration of internal clusters used for development and testing. <br><br>

9.  **The models have many more parameters than are contained in the yaml configuration file. Where are the remainder of the configuration parameters set?** <br><br>

    The process for training large models can be challenging and requires the configuration of many parameters, some of which will be changed infrequently. All BioNeMo models contain a base configuration file, which is located in the same directory as the configuration file directly used for training or inference. The name of the base configuration is referenced at the top of the configuration file used for training or inference. Additional parameters from the base configuration can be added to the configuration used for training and their default values changed to the desirable alternative.
