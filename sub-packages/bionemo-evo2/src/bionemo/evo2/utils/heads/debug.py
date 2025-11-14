import torch
from nemo.utils import logging


def debug_heads(
        name: str,
        model: torch.nn.Module,
        batch: dict | None = None,
        forward_args: dict | None = None,
        forced: bool = True
    ) -> None:
    """Debugging utility to log model and batch information during forward passes.

    Args:
        name: Name identifier for the debug instance.
        model: The model being debugged.
        batch: The input batch dictionary.
        forward_args: Additional arguments passed to the forward method.
        forced: If True, prints to stdout; otherwise uses logging.info.
    """
    if forced:
        print(f"🔄 {name} called")
        print(f"   Model type: {type(model)}")
        print(f"   Model id: {id(model)}")

        # Debug model hierarchy
        current = model
        level = 0
        while hasattr(current, 'module') and level < 5:
            print(f"   Level {level}: {type(current)} (id: {id(current)})")
            if hasattr(current, 'forward'):
                print(f"     Has forward method: {hasattr(current, '_original_forward')}")
            current = current.module    # type: ignore
            level += 1
        print(f"   Final level {level}: {type(current)} (id: {id(current)})")
        if hasattr(current, 'forward'):
            print(f"     Has _original_forward: {hasattr(current, '_original_forward')}")
            print(f"     Has expression_head: {hasattr(current, 'expression_head')}")
            print(f"     Has parallel_token_head: {hasattr(current, 'parallel_token_head')}")

        if batch is not None:
            print(f"   Batch keys: {list(batch.keys())}")

            # Debug batch contents
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape} {value.dtype}")
                    if key == 'expression_targets' and value is not None:
                        print(f"     Range: [{value.min().item():.3f}, {value.max().item():.3f}]")

        if forward_args is not None:
            print(f"   Forward args keys: {list(forward_args.keys())}")

    else:
        logging.info(f"🔄 {name} called")
        logging.info(f"   Model type: {type(model)}")
        logging.info(f"   Model id: {id(model)}")

        # Debug model hierarchy
        current = model
        level = 0
        while hasattr(current, 'module') and level < 5:
            logging.info(f"   Level {level}: {type(current)} (id: {id(current)})")
            if hasattr(current, 'forward'):
                logging.info(f"     Has forward method: {hasattr(current, '_original_forward')}")
            current = current.module    # type: ignore
            level += 1
        logging.info(f"   Final level {level}: {type(current)} (id: {id(current)})")
        if hasattr(current, 'forward'):
            logging.info(f"     Has _original_forward: {hasattr(current, '_original_forward')}")
            logging.info(f"     Has expression_head: {hasattr(current, 'expression_head')}")
            logging.info(f"     Has parallel_token_head: {hasattr(current, 'parallel_token_head')}")

        if batch is not None:
            logging.info(f"   Batch keys: {list(batch.keys())}")

            # Debug batch contents
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logging.info(f"   {key}: {value.shape} {value.dtype}")
                    if key == 'expression_targets' and value is not None:
                        logging.info(f"     Range: [{value.min().item():.3f}, {value.max().item():.3f}]")

        if forward_args is not None:
            logging.info(f"   Forward args keys: {list(forward_args.keys())}")
