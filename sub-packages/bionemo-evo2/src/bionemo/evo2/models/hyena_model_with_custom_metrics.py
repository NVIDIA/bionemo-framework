from nemo.collections import llm
from typing import Optional
from bionemo.evo2.utils.logging.bnm_module_hook_manager import BnmModuleHookManager


class HyenaModelWithCustomMetrics(llm.HyenaModel):
    
    def configure_model(self, vp_stage: Optional[int] = None) -> None:
        """Add additional configuration for HyenaModel(GPTModel), 
        after GPTModel.configure_model().
        
        When this method is called, self.module is the 
        HyenaModel(LanguageModule(MegatronModel))                 
        
        """
        super(llm.HyenaModel, self).configure_model(vp_stage=vp_stage)

        self.bnm_module_hook_manager = BnmModuleHookManager()

        self.bnm_module_hook_manager.configure_hooks(
            root_module=self.module,
            forward_pre_hook_types=["input_shapes"],
            forward_hook_types=["output_shapes"],
        )