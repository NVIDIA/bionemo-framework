import torch
from lightning import LightningModule
from abc import ABC, abstractmethod

class BaseInference(LightningModule, ABC):
    def __init__(self, model_path: str, task_type: str):
        super().__init__()
        self.task_type = task_type
        self.model_path = model_path
        self.model = None
        self.prediction_counter = 0  # Initialize prediction counter
        self.save_hyperparameters()

    @abstractmethod
    def configure_model(self):
        """Configure the model. Must be implemented by subclasses."""
        pass

    def predict_step(self, batch, batch_idx):
        """Perform a prediction step and increment the counter."""
        self.prediction_counter += 1
        return self._predict_step(batch, batch_idx)

    @abstractmethod
    def _predict_step(self, batch, batch_idx):
        """Perform the actual prediction step. Must be implemented by subclasses."""
        pass