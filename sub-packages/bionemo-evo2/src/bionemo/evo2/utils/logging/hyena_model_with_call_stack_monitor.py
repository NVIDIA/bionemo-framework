from typing import Optional
from torch import nn, Tensor
from nemo.collections import llm
from bionemo.evo2.utils.logging.bnm_call_stack_monitor import BnmCallStackMonitor


class HyenaModelWithCallStackMonitor(llm.HyenaModel):
    
    def configure_model(self, vp_stage: Optional[int] = None) -> None:
        """Add additional configuration for HyenaModel(GPTModel), after GPTModel.configure_model().
        
        When this method is called, self.module is the HyenaModel(LanguageModule(MegatronModel))
        
        """
        super(llm.HyenaModel, self).configure_model(vp_stage=vp_stage)

        global BNM_CALL_STACK_MONITOR_HOOKS
        BNM_CALL_STACK_MONITOR_HOOKS = []

        def forward_pre_hook(module: nn.Module, input: Tensor | tuple[Tensor]):
            if not hasattr(module, "bnm_call_stack_monitor"):
                module.bnm_call_stack_monitor = BnmCallStackMonitor()
                module.bnm_call_stack_monitor.start_monitoring()
        
        def forward_hook(module: nn.Module, input: Tensor | tuple[Tensor], output: Tensor | tuple[Tensor]):
            if hasattr(module, "bnm_call_stack_monitor"):
                module.bnm_call_stack_monitor.stop_monitoring()
                module.bnm_call_stack_monitor.write_events_to_file()
                
        BNM_CALL_STACK_MONITOR_HOOKS.append(
            self.module.register_forward_pre_hook(forward_pre_hook)
        )
        BNM_CALL_STACK_MONITOR_HOOKS.append(
            self.module.register_forward_hook(forward_hook)
        )





        
        

