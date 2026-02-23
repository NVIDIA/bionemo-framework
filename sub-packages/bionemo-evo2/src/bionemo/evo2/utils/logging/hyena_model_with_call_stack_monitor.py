from typing import TYPE_CHECKING, Optional

from nemo.collections import llm
from nemo.collections.llm.gpt.model.hyena import HyenaConfig
from torch import Tensor, nn

from bionemo.evo2.utils.logging.bnm_call_stack_monitor import BnmCallStackMonitor


if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.lightning import OptimizerModule


class HyenaModelWithCallStackMonitor(llm.HyenaModel):
    """HyenaModel variant that instruments forward passes with call-stack monitoring."""

    def __init__(
        self,
        config: HyenaConfig,
        optim: Optional["OptimizerModule"] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform=None,
        model_context_managers: Optional[list] = None,
    ) -> None:
        """Initialize with a HyenaConfig instead of the inherited GPTConfig."""
        super().__init__(
            config,  # type: ignore[arg-type]
            optim=optim,
            tokenizer=tokenizer,
            model_transform=model_transform,
            model_context_managers=model_context_managers or [],
        )

    def configure_model(self, vp_stage: Optional[int] = None) -> None:
        """Add additional configuration for HyenaModel(GPTModel), after GPTModel.configure_model().

        When this method is called, self.module is the HyenaModel(LanguageModule(MegatronModel))
        """
        super(llm.HyenaModel, self).configure_model(vp_stage=vp_stage)

        global BNM_CALL_STACK_MONITOR_HOOKS
        BNM_CALL_STACK_MONITOR_HOOKS = []

        def forward_pre_hook(module: nn.Module, input: Tensor | tuple[Tensor]):
            if not hasattr(module, "bnm_call_stack_monitor"):
                module.bnm_call_stack_monitor = BnmCallStackMonitor()  # pyright: ignore
                module.bnm_call_stack_monitor.start_monitoring()

        def forward_hook(module: nn.Module, input: Tensor | tuple[Tensor], output: Tensor | tuple[Tensor]):
            if hasattr(module, "bnm_call_stack_monitor"):
                module.bnm_call_stack_monitor.stop_monitoring()  # pyright: ignore
                module.bnm_call_stack_monitor.write_events_to_file()   # pyright: ignore
        BNM_CALL_STACK_MONITOR_HOOKS.append(
            self.module.register_forward_pre_hook(forward_pre_hook)
        )
        BNM_CALL_STACK_MONITOR_HOOKS.append(
            self.module.register_forward_hook(forward_hook)
        )

