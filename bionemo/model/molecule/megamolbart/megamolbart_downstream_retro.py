from typing import Callable, Dict, List, Optional

import torch
from omegaconf import DictConfig

from bionemo.data import megamolbart_retro_build_train_valid_test_datasets
from bionemo.data.molecule.augment import MoleculeInputTargetEnumeration
from bionemo.model.molecule.megamolbart.megamolbart_model_base import MegaMolBARTModelBase


__all__ = ["MegaMolBARTRetroModel"]


class MegaMolBARTRetroModel(MegaMolBARTModelBase):
    """
    MegaMolBARTRetro model for the retrosynthesis downstream task
    """

    def _setup_collate(self, pad_size_divisible_by_8: bool) -> Callable:
        """
        Sets up collate fn that is required by dataloader used to finetune MegaMolBART
        Args:
            pad_size_divisible_by_8: should torch.Tensors be padded to sizes divisible by 8?
        Returns:
            callable collate fn
        """
        return MoleculeInputTargetEnumeration(
            tokenizer=self.tokenizer,
            seq_length=self._cfg.seq_length,
            pad_size_divisible_by_8=pad_size_divisible_by_8,
            **self._cfg.data,
        ).collate_fn

    def _load_train_valid_test_datasets(self, train_valid_test_num_samples: Dict[str, int]):
        """
        Helper method that sets instance variables corresponding to train, val and test datasets
        Args:
            train_valid_test_num_samples: dicts with number of samples needed for train, val and test steps
        """
        self._train_ds, self._validation_ds, self._test_ds = megamolbart_retro_build_train_valid_test_datasets(
            self._cfg.data, train_valid_test_num_samples
        )

    def build_data_loader(self, dataset, consumed_samples, num_workers):
        return super().build_pretraining_data_loader(dataset, consumed_samples, num_workers)

    def build_pretraining_data_loader(self, dataset, consumed_samples, num_workers):
        pass

    def setup_training_data(self, cfg: DictConfig):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_data_loader(
                self._train_ds, consumed_samples, num_workers=self._cfg.data.num_workers
            )

    def setup_validation_data(self, cfg: DictConfig):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            self._validation_dl = self.build_data_loader(self._validation_ds, consumed_samples, num_workers=0)

    def setup_test_data(self, cfg: DictConfig):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            self._test_dl = self.build_data_loader(self._test_ds, consumed_samples, num_workers=0)

    def predict_step(self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: Optional[int] = None) -> dict:
        self.freeze()
        tokens_enc, _, _, _, enc_mask, _ = self.process_global_batch(batch)
        predicted_tokens_ids, log_probs = self.decode(tokens_enc, enc_mask, self._cfg.max_position_embeddings)
        predicted_tokens_ids = predicted_tokens_ids.cpu().detach().numpy().tolist()
        log_probs = log_probs.cpu().detach().numpy().tolist()
        response = {'tokens_enc': tokens_enc}
        sampled_smiles, predicted_tokens_pruned = self._process_predicted_tokens_ids(predicted_tokens_ids)
        response['predicted_tokens'] = list(zip(sampled_smiles, predicted_tokens_pruned, log_probs))
        self.unfreeze()
        return response
