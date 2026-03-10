from typing import Any, Dict
import os


import torch

import nemo_run as run

from lightning.pytorch import seed_everything, Trainer
from one_logger_utils.pytorch_lightning import hook_trainer_cls
from src.config import create_strategy_from_config
from src.utils import RankedLogger

# Initialize logger at module level so it's available to all functions
logging = RankedLogger(__name__, rank_zero_only=True)


def train(config: Dict[str, Any], 
          ckpt_path: str, 
          seed: int, 
          config_dict: Dict[str, Any], 
          out_dir: str):
    """Launches the pre-training process for the Encodon model.

    Args:
        config: A dictionary containing the configuration for the model, data, and trainer.
        ckpt_path: The path to the checkpoint file to resume training from.
        seed: The random seed to use for reproducibility.
    """
    seed_everything(seed, workers=True)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    logger, data, trainer_kwargs, model, callbacks, one_logger_callback_config = config["log"], \
                                              config["data"], \
                                              config["trainer"], \
                                              config["model"], \
                                              config["callbacks"], \
                                              config["one_logger_callback_config"]
    
    # Set enable_for_current_rank based on both RANK and enable_one_logger flag
    enable_one_logger = config_dict.get('enable_one_logger', False)
    config["one_logger_callback_config"]["enable_for_current_rank"] = (os.environ.get('RANK') == '0') and enable_one_logger
    
    # Hook the trainer class and create trainer instance with hooked class
    hooked_trainer_cls, one_logger_callback = hook_trainer_cls(Trainer, callback_config=one_logger_callback_config)
    trainer_kwargs = create_strategy_from_config(trainer_kwargs)
    trainer = hooked_trainer_cls(**trainer_kwargs)
    
    if logger and hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(config_dict)
        
    if os.path.exists(ckpt_path):
        one_logger_callback.on_load_checkpoint_start()
        state_dict = torch.load(ckpt_path, map_location="cpu")
        one_logger_callback.on_load_checkpoint_end()
        one_logger_callback.on_model_init_start()
        model.configure_model(state_dict=state_dict.get("state_dict"))
    else:
        one_logger_callback.on_model_init_start()
        model.configure_model()
    one_logger_callback.on_model_init_end()
    
    trainer.callbacks = list(callbacks.values()) + [one_logger_callback]
    data.one_logger_callback = one_logger_callback
    trainer.logger = logger
    logging.info(f"Starting pre-training from {ckpt_path}")
    trainer.fit(model, datamodule=data, ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None)

@run.cli.entrypoint(namespace="codonfm-finetune")
def finetune(config: Dict[str, Any],
             pretrained_ckpt_path: str,
             seed: int,
             resume_trainer_state: bool,
             config_dict: Dict[str, Any], 
             out_dir: str,
             ckpt_path: str):
    """
    Launches the fine-tuning process for the Encodon model.

    Args:
        config: A dictionary containing the configuration for the model, data, and trainer.
        ckpt_path: The path to save the fine-tuned model checkpoint.
        pretrained_ckpt_path: The path to the pre-trained model checkpoint to start fine-tuning from.
        seed: The random seed to use for reproducibility.
        resume_trainer_state: Whether to resume the trainer state from the checkpoint.
    """
    seed_everything(seed, workers=True)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    logger, data, trainer_kwargs, model, callbacks, one_logger_callback_config = config["log"], \
                                              config["data"], \
                                              config["trainer"], \
                                              config["model"], \
                                              config["callbacks"], \
                                              config["one_logger_callback_config"]
                                              
    # Set enable_for_current_rank based on both RANK and enable_one_logger flag
    enable_one_logger = config_dict.get('enable_one_logger', False)
    config["one_logger_callback_config"]["enable_for_current_rank"] = (os.environ.get('RANK') == '0') and enable_one_logger
    
    # Hook the trainer class and create trainer instance with hooked class
    hooked_trainer_cls, one_logger_callback = hook_trainer_cls(Trainer, callback_config=one_logger_callback_config)
    trainer_kwargs = create_strategy_from_config(trainer_kwargs)
    trainer = hooked_trainer_cls(**trainer_kwargs)
    
    if logger and hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(config_dict)
        
    if os.path.exists(pretrained_ckpt_path) and not os.path.exists(ckpt_path):
        one_logger_callback.on_load_checkpoint_start()
        state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
        one_logger_callback.on_load_checkpoint_end()
        one_logger_callback.on_model_init_start()
        model.configure_model(state_dict=state_dict.get("state_dict"))
    else:
        logging.info(f"No pretrained checkpoint found at {pretrained_ckpt_path}, starting from scratch")
        one_logger_callback.on_model_init_start()
        model.configure_model()
    one_logger_callback.on_model_init_end()
    
    trainer.callbacks = list(callbacks.values()) + [one_logger_callback]
    data.one_logger_callback = one_logger_callback
    trainer.logger = logger
    
    if resume_trainer_state and os.path.exists(pretrained_ckpt_path) and not os.path.exists(ckpt_path):
        trainer_ckpt_path = pretrained_ckpt_path
    else:
        trainer_ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    
    logging.info(f"Starting finetuning from {trainer_ckpt_path} \
        with resume_trainer_state={resume_trainer_state} and ckpt_path={ckpt_path}")
    trainer.fit(model, datamodule=data, ckpt_path=trainer_ckpt_path)


@run.cli.entrypoint(namespace="codonfm-eval")
def evaluate(
    config: Dict[str, Any],
    config_dict: Dict[str, Any],
    model_ckpt_path: str,
    out_dir: str,
    seed: int = 123,
) -> None:
    """
    Launches the evaluation process for the Encodon model.

    Args:
        config: A dictionary containing the configuration for the model, data, and trainer.
        config_dict: A dictionary containing the configuration for the model, data, and trainer.
        
    Note: Evaluation must be run in a single run as resuming the trainer state is not supported for prediction.
    """
    # Get rank from environment (set by launcher before distributed init)
    rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
    per_rank_seed = seed + rank
    seed_everything(per_rank_seed, workers=True)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger, data, trainer_kwargs, model, callbacks = config["log"], \
                                          config["data"], \
                                          config["trainer"], \
                                          config["model"], \
                                          config["callbacks"]
    
    trainer_kwargs = create_strategy_from_config(trainer_kwargs)
    trainer = Trainer(**trainer_kwargs)
    
    if logger and hasattr(logger, "log_hyperparams"):
        logger.log_hyperparams(config_dict)
        
    model.configure_model()
    data.setup("test")
    
    trainer.logger = logger
    trainer.callbacks = list(callbacks.values())
    
    logging.info("Starting Evaluation!")
    trainer.predict(model, 
                    datamodule=data, 
                    return_predictions=False)
    return 