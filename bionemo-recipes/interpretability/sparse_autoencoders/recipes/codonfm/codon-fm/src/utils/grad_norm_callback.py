import lightning.pytorch as pl
import torch

class GradientNormLogger(pl.Callback):
    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        if (trainer.global_step) % self.log_every_n_steps == 0:
            grad_norms = []
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    grad_norms.append((name, grad_norm))
            
            if grad_norms:
                avg_grad_norm = sum(gn[1] for gn in grad_norms) / len(grad_norms)
                pl_module.log('avg_grad_norm', avg_grad_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True)