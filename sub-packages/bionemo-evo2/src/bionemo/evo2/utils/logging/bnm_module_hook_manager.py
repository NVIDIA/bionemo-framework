# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable
import os
from threading import stack_size
from torch import nn
from torch import Tensor

BNM_MODULE_HOOK_HANDLES = []


class BnmModuleHookManager():
    
    def configure_hooks(
        self, 
        root_module: nn.Module,
        results_dir: str | None = None,
        forward_pre_hook_types: list[str] | None = None,
        forward_hook_types: list[str] | None = None,
        level_max: int | None = None,
    ):
        """Configure hooks. 
        
        Args:
            root_module: The module ancestor to all submodules which should have hooks.
            forward_pre_hook_types: The types of forward pre hooks to configure # ["input_shapes"]
            forward_hook_types: The types of forward hooks to configure. #["output_shapes"],
        """
        print(f"BnmModuleHookManager,configure_hooks,type(self.module)={type(root_module)}")
        self.root_module = root_module
        self.level_max = os.getenv("BNM_MODULE_HOOK_MANAGER_LEVEL_MAX", level_max)  # str or None or int
        if isinstance(self.level_max, str):
            self.level_max = int(self.level_max)
  
        self.results_dir = os.getenv("BNM_MODULE_HOOK_MANAGER_RESULTS_DIR", results_dir) # str or None
        self.bnm_module_hook_output_filename = None if self.results_dir is None else os.path.join(str(self.results_dir), f"bnm_module_hook_output_lvl{level_max}.txt")
    
        
        self.forward_pre_hook_types = forward_pre_hook_types
        self.forward_hook_types = forward_hook_types
        
        header_with_column_names = ";".join([
            "class_to_collect_metrics",
            "method_name",
            "level",
            "hooked_pytorch_module_name",
            "hooked_function_name",
            "metric_name",
            "metric_value",
        ])
        self.write_line_to_file(header_with_column_names)
        
        BnmModuleHookManager.do_for_each_submodule_bfs(
            func=self.configure_hooks_for_submodule,
            module=root_module,
            level=0,
            level_max=level_max,
        )
   
    def configure_hooks_for_submodule(self, module: nn.Module, level: int | None = None):
        """
        Args:
            module: A submodule
            level: The level of the submodule in the subtree of the root module
        """

            
        if isinstance(self.forward_pre_hook_types,list) and "input_shapes" in self.forward_pre_hook_types:
            
            def forward_pre_hook_for_input_shapes(
                module: nn.Module, 
                input: tuple[Tensor]
            ):
                message = BnmModuleHookManager.bnm_forward_pre_hook_for_input_shapes_helper(module, input, level)
                self.write_line_to_file(message)

            BNM_MODULE_HOOK_HANDLES.append(
                module.register_forward_pre_hook(forward_pre_hook_for_input_shapes)
            )
        
        if isinstance(self.forward_hook_types,list) and "output_shapes" in self.forward_hook_types:
            
            def forward_hook_for_output_shapes(
                module: nn.Module, 
                input: tuple[Tensor], 
                output: tuple[Tensor] | Tensor,
            ):
                message = BnmModuleHookManager.bnm_forward_hook_for_output_shapes_helper(module, input, output, level)
                self.write_line_to_file(message)
            
            BNM_MODULE_HOOK_HANDLES.append(
                module.register_forward_hook(forward_hook_for_output_shapes)
            )

    def write_line_to_file(self, line: str):
        if self.bnm_module_hook_output_filename is not None:
            with open(self.bnm_module_hook_output_filename, "a") as f:
                f.write(line + "\n")
    
    @staticmethod
    def do_for_each_submodule_bfs(
        func: Callable, 
        module: nn.Module, 
        level: int = 0, 
        level_max: int | None = None
    ):

        func(module, level)
        if level_max is None or level+1 <= level_max:
            for _, child in module.named_children():
                BnmModuleHookManager.do_for_each_submodule_bfs(
                    func=func, module=child, level=level + 1, level_max=level_max
                )
                
    @staticmethod
    def bnm_forward_pre_hook_for_input_shapes_helper(
        module: nn.Module, input: tuple[Tensor] | Tensor, level: int | None = None
    ) -> str:
        some_list_of_strings = ["NA"]
        if isinstance(input, Tensor):
            some_list_of_strings = [str(tuple(input.shape))]
        elif isinstance(input, tuple):
            some_list_of_strings = [
                "NA" if not isinstance(input_component, Tensor) else str(tuple(input_component.shape)) 
                for input_component in input
            ]
    
        input_names_and_shapes = "|".join(some_list_of_strings)    
        message = ";".join([
            "BnmModuleHookManager",
            "bnm_forward_pre_hook_for_input_shapes_helper",
            f"{level}",
            f"{module.__class__.__name__}",
            "forward",
            "input_shapes",
            f"{input_names_and_shapes}",
        ])
        return message
        
    @staticmethod
    def bnm_forward_hook_for_output_shapes_helper(
        module: nn.Module, input: tuple[Tensor], output: tuple[Tensor] | Tensor, level: int | None = None
    ) -> str:
        some_list_of_strings = ["NA"]
        if isinstance(output, Tensor):
            some_list_of_strings = [str(tuple(output.shape))]
        elif isinstance(output, tuple):
            some_list_of_strings = ["NA" if not isinstance(output_component, Tensor) else str(tuple(output_component.shape)) for output_component in output]
        
        output_names_and_shapes = "|".join(some_list_of_strings)
        message = ";".join([
            "BnmModuleHookManager",
            "bnm_forward_hook_for_output_shapes_helper",
            f"{level}",
            f"{module.__class__.__name__}",
            "forward",
            "output_shapes",
            f"{output_names_and_shapes}",
        ])
        return message
