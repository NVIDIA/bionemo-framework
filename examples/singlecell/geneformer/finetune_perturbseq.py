# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from bionemo.data.metrics import mse
from bionemo.data.preprocess.singlecell.preprocess import AdamsonResources, preprocess_adamson
from bionemo.model.singlecell.downstream.finetuning import FineTuneGeneformerModel
from bionemo.model.utils import (
    create_geneformer_preprocessor,
    setup_trainer,
)


@hydra_runner(config_path="conf", config_name="finetune_perturb")
def main(cfg) -> None:
    """
    Main function for Finetuning the Geneformer model on Perturb Prediction Task.

    Args:
        cfg (OmegaConf): Configuration object containing the experiment settings.

    Returns:
        None
    """

    logging.info("\n\n************* Finetune config ****************")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Required for a lot of our inference functionality!
    with open_dict(cfg):
        cfg.model.encoder_cfg = cfg
    # Create the preprocessor for the restored model.

    if not cfg.do_training:
        # Get the medians and vocab setup appropriately
        logging.info("************** Starting Preprocessing ***********")
        # Path that the medians file gets saved to. Note that internally the tokenizer also controls saving its vocab based on a location in the config
        preprocessor = create_geneformer_preprocessor(
            cfg.model.data.compute_medians,
            cfg.model.data.train_dataset_path,
            cfg.model.data.medians_file,
            cfg.model.tokenizer.vocab_file,
        )
        match preprocessor.preprocess():
            case {"tokenizer": _, "median_dict": _}:
                logging.info("*************** Preprocessing Finished ************")
            case _:
                logging.error("Preprocessing failed.")
        # Preprocesses the Adamson PERTURB-seq dataset.
        resource_fetcher = AdamsonResources(
            root_directory=cfg.model.data.dataset_path, dest_directory=cfg.model.data.preprocessed_data_path
        )
        artifacts = resource_fetcher.prepare_annotated()
        logging.info("*************** Adamson Download Finished ************")
        _ = preprocess_adamson(
            adamson_perturbed_processed_fn=artifacts["perturbed_h5ad"],
            gene2go_pkl_fn=artifacts["gene2go_pkl"],
            all_pert_genes_pkl_fn=artifacts["pert_genes_pkl"],
            dest_preprocessed_anndata_fn=cfg.model.data.preprocessed_anndata_fn,
            dest_target_gep_fn=cfg.model.data.target_gep_fn,
        )
        logging.info("*************** Adamson Preprocessing Finished ************")
    else:
        logging.info("************** Starting Training ***********")

        # Set up trainer
        trainer = setup_trainer(cfg, builder=None, reset_accumulate_grad_batches=False)

        # Create model
        model = FineTuneGeneformerModel(cfg.model, trainer=trainer)

        # Add evaluation metrics
        metrics = {"MSE": mse}
        metrics_args = {"MSE": {}}
        model.add_metrics(metrics=metrics, metrics_args=metrics_args)

        # Train model
        trainer.fit(model)
        logging.info("************** Finished Training ***********")

        if cfg.do_testing:
            logging.info("************** Starting Testing ***********")
            metrics = trainer.test(model, dataloaders=model.data_module.test_dataloader())
            logging.info(metrics)
            logging.info("************** Finished Testing ***********")


if __name__ == "__main__":
    main()
