from typing import Any, Dict

import fiddle as fdl

def config_to_dict(cfg: Any) -> Dict[str, Any]:
    """Recursively converts a fdl.Config or a dict of fdl.Configs to a dictionary."""
    if isinstance(cfg, dict):
        return {k: config_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, fdl.Config):
        return {k: config_to_dict(getattr(cfg, k)) for k in getattr(cfg, "__arguments__", {})}
    if isinstance(cfg, fdl.Partial):
        return {k: config_to_dict(getattr(cfg, k)) for k in getattr(cfg, "__arguments__", {})}
    return cfg