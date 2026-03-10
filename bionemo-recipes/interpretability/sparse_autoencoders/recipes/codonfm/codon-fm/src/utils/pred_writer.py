import os
from dataclasses import asdict, is_dataclass

import torch
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter


class PredWriter(BasePredictionWriter):
    """This class is used to write predictions to a file.
        It is used to write predictions to a file after each batch or epoch.
        It is also used to merge predictions from multiple processes.
        
        Note: if you're running this with DDP and your final batch is not divisible by the number of gpu workers,
        then you will have duplicate predictions that may overlap with the first set of samples in your.
    """
    def __init__(self, output_dir: str, write_interval: str, caching_interval: int = 1, merge_on_epoch_end: bool = False, delete_after_merge: bool = False):
        
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.caching_interval = caching_interval
        self.merge_on_epoch_end = merge_on_epoch_end
        self.delete_after_merge = delete_after_merge
        self.predictions_buffer = []
        os.makedirs(self.output_dir, exist_ok=True)

    def _convert_predictions(self, prediction):
        if is_dataclass(prediction):
            prediction = asdict(prediction)
        elif not isinstance(prediction, dict):
            raise TypeError("Prediction must be a dataclass or a dictionary.")
        return {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for key, value in prediction.items() if value is not None
        }

    def _save_predictions(self, trainer, batch_idx):
        flattened_predictions = {
            key: np.concatenate([buffer[key] for buffer in self.predictions_buffer], axis=0)
            for key in self.predictions_buffer[0]
        }
        for key, value in flattened_predictions.items():
            file_path = os.path.join(
                self.output_dir, f"{key}_rank_{trainer.global_rank}_batch_{batch_idx}.npy"
            )
            np.save(file_path, value)
        self.predictions_buffer.clear()

    def _merge_predictions(self):
        all_files = [f for f in os.listdir(self.output_dir) if f.endswith(".npy")]
        merged_data = {}
        sorted_files = sorted(
            all_files,
            key=lambda file: (
                file.split("_rank_")[0],
                int(file.split("_batch_")[-1].split(".npy")[0])
            )
        )

        for file in sorted_files:
            key = file.split("_rank_")[0]
            rank = int(file.split("_rank_")[1].split("_batch_")[0])
            batch_idx = int(file.split("_batch_")[-1].split(".npy")[0])
            file_path = os.path.join(self.output_dir, file)
            data = np.load(file_path)

            if key not in merged_data:
                merged_data[key] = {}

            if batch_idx not in merged_data[key]:
                merged_data[key][batch_idx] = {}

            merged_data[key][batch_idx][rank] = data

        for key, batches in merged_data.items():
            all_data = []
            max_rank = max(max(rank_data.keys()) for rank_data in batches.values())
        
            for batch_idx, rank_data in sorted(batches.items()):
                batch_data = [None] * (max_rank + 1)
            
                for rank, data in rank_data.items():
                    batch_data[rank] = data
                
                # - reconstruct the original order based on the sampling logic
                total_size = sum(len(batch_data[rank]) for rank in range(len(batch_data)) if batch_data[rank] is not None)
                num_replicas = max_rank + 1
                sample_data = next(data for data in batch_data if data is not None)
                data_shape = (total_size,) + sample_data.shape[1:]
                dtype = batch_data[0].dtype
                reconstructed_data = np.empty(data_shape, dtype=dtype if batch_data[0] is not None else float)
                
                for rank in range(len(batch_data)):
                    if batch_data[rank] is not None:
                        indices = list(range(rank, total_size, num_replicas))
                        for i, idx in enumerate(indices):
                            if i < len(batch_data[rank]):
                                reconstructed_data[idx] = batch_data[rank][i]
                                
                all_data.append(reconstructed_data)
            
            merged_array = np.concatenate(all_data, axis=0)
            merged_file_path = os.path.join(self.output_dir, f"{key}_merged.npy")
            np.save(merged_file_path, merged_array)

        if self.delete_after_merge:
            for file in all_files:
                os.remove(os.path.join(self.output_dir, file))
            
    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        prediction = self._convert_predictions(prediction)
        self.predictions_buffer.append(prediction)
        if len(self.predictions_buffer) >= self.caching_interval:
            adjusted_batch_idx = (
                trainer.datamodule.init_consumed_samples // trainer.datamodule.global_batch_size + batch_idx
            )
            self._save_predictions(trainer, adjusted_batch_idx)

    def on_predict_epoch_end(self, trainer, pl_module):
        if len(self.predictions_buffer) > 0:
            consumed_samples = trainer.datamodule.calc_consumed_samples()
            batch_idx = consumed_samples // trainer.datamodule.global_batch_size 
            adjusted_batch_idx = (
                batch_idx + int((consumed_samples % trainer.datamodule.global_batch_size) > 0)
            )
            self._save_predictions(trainer, adjusted_batch_idx)
        
        trainer.strategy.barrier()
        if self.merge_on_epoch_end and trainer.is_global_zero:
            self._merge_predictions()
