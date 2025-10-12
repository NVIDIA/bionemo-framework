import os
import sys
import inspect
from torch import Tensor
                    
EVENT_TYPE = "event_type"
LEVEL_OF_CALL_FRAME = "level_of_call_frame"


class BnmCallStackMonitor():
    def __init__(self, results_dir: str | None = None,):

        self.level_max = os.getenv("BNM_CALL_STACK_MONITOR_LEVEL_MAX", 9)  # str or None or int
        if isinstance(self.level_max, str):
            self.level_max = int(self.level_max)

        self.num_events_max = None
    
        self.results_dir = os.getenv("BNM_CALL_STACK_MONITOR_RESULTS_DIR", results_dir) # str or None
        self.results_filename = None 
        if self.results_dir is not None:
            self.results_filename = os.path.join(
                str(self.results_dir), f"bnm_call_stack_monitor_output.txt"
            )
        global BNM_CALL_STACK_MONITOR_OUTPUT_FILENAME
        BNM_CALL_STACK_MONITOR_OUTPUT_FILENAME = self.results_filename
        
    def start_monitoring(self):
        global CALL_STACK_EVENTS 
        CALL_STACK_EVENTS = []
        
        prof = create_profiler_with_function_io_metrics(CALL_STACK_EVENTS, level_max = self.level_max, num_events_max=self.num_events_max)
        sys.setprofile(prof)
    
    def stop_monitoring(self):
        sys.setprofile(None)
    
    @property
    def call_stack_events(self):
        return CALL_STACK_EVENTS
    
    def delete_call_stack_events(self):
        del CALL_STACK_EVENTS
    
    def write_events_to_file(self):
        if self.results_filename is None:
            raise Exception
                
        header_with_column_names = ";".join([
            "class_to_collect_metrics",
            "level",
            "module_or_class_name_short",
            "func_name",
            "frame_id",
            EVENT_TYPE,
            "event_id",
            "metric_name",
            "metric_value",
            "is_class_name_in_black_list",
            "is_function_name_in_blacklist",
            "class_name_long",
            "location",
        ])
        BnmCallStackMonitor.write_line_to_file(
            filename=self.results_filename,
            line=header_with_column_names,
        )  
        
        for event in self.call_stack_events:
            message_as_line = ";".join([str(x) for x in [
                "BnmCallStackMonitor",
                event[LEVEL_OF_CALL_FRAME],
                event["class_name_short"],
                event["func_name"],
                event["frame_id"],
                event[EVENT_TYPE],
                event["event_id"],
                event["metric_name"],
                event["metric_value"],
                event["is_class_name_in_black_list"],
                event["is_function_name_in_blacklist"],
                event["class_name_long"],
                event["location"],
            ]])
                
            BnmCallStackMonitor.write_line_to_file(
                filename=self.results_filename,
                line=message_as_line,
            )  

    @staticmethod
    def write_line_to_file(filename: str, line: str):
        if filename is not None:
            with open(filename, "a") as f:
                f.write(line + "\n")
                f.flush() 
    
 
def create_brief_module_name(frame):
     
    frame_code_filename = f"{frame.f_code.co_filename}"
    for x in ["dist-packages/", "3rdparty/"]:
        if x in frame_code_filename:
            frame_code_filename = frame_code_filename.split(x)[-1]
            break
    
    frame_code_filename = frame_code_filename.rstrip(".py")
    split_result = frame_code_filename.split("/")
    
    if len(split_result) <= 2:
        out =  ".".join(split_result)
    else:
        out = "...".join([split_result[0], split_result[-2] ]) 
    return out
    

def create_profiler_with_function_io_metrics(call_stack_events: list, num_events_max: int= 50, level_max: int = 9):
    """
    Returns a profiling function that logs inputs and outputs of every function call.
    
    Use the returned function like:
    
    prof = create_profiler_with_function_io_metrics(CALL_STACK_EVENTS)
    sys.setprofile(prof)
    
    """

    def profiler(frame, event_type, arg):
        
        if isinstance(num_events_max, int) and len(call_stack_events) >= num_events_max:
            return
        
        func_name = frame.f_code.co_name
        func_loc = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        args, _, _, values = inspect.getargvalues(frame)
        frame_args_as_dict = {k: values[k] for k in args}
        
        brief_module_name = create_brief_module_name(frame)
        
        is_an_input_a_tensor = any([isinstance(v, Tensor) for v  in frame_args_as_dict.values()])
        
        if not is_an_input_a_tensor:
            return
        
        # FILEPATH_KEY_WHITELIST = ["NeMo", "Megatron", "evo2", "einops"]
        # does_func_loc_contain_key_from_whitelist = any([x in func_loc for x in FILEPATH_KEY_WHITELIST])
        # if not does_func_loc_contain_key_from_whitelist:
        #     return
        
        FUNCTION_NAME_BLACKLIST = [
            "nvtx_range_push", 
            "nvtx_range_pop", 
            "__hash__", 
            "maybe_contiguous", 
            "cast_if_needed", 
            "cast", 
            "shape", 
            "<lambda>", 
            "reset_swizzled_inputs", "swizzle_inputs", "set_activation_dtype", 
            "is_appropriate_type",
            "convert_tensor",
            "get_backend",
            "_apply_recipe",
            "_check_single_tensor",
            "make_viewless_tensor",
            "make_upper_case",
            "reduce_from_tensor_model_parallel_region",
            "fused_apply_rotary_pos_emb",
            "reduce_from_tensor_model_parallel_region",
            "copy_to_tensor_model_parallel_region",
    
        ]
        is_function_name_in_blacklist = any([x in func_name for x  in FUNCTION_NAME_BLACKLIST])
        
        is_class_method, class_name_long, _ = frame_is_class_method(frame)
        brief_module_name = create_brief_module_name(frame)
        class_name_short = brief_module_name if class_name_long is None else ".".join(class_name_long.split(".")[-1:])
        
        CLASS_NAME_BLACKLIST = [
            "SymNumberMemoDescriptor", 
            "MetaTensorDescriber",
            "WeakIdRef", 
            "WeakIdKeyDictionary", 
            "FakeTensor", 
            "OperationFuser",
            "IdentityOp",
        ]
        is_class_name_in_black_list =  any([class_name_short==x for x in CLASS_NAME_BLACKLIST])
        
        
        level_of_call_frame = None
        metric_name = None
        metric_value = None
        if event_type not in ["call", "return"]:
            return
        
        elif event_type == "call":
            if len(call_stack_events) == 0:
                level_of_call_frame = 0
            elif call_stack_events[-1][EVENT_TYPE] == "call":
                level_of_call_frame = call_stack_events[-1][LEVEL_OF_CALL_FRAME] + 1
                if level_of_call_frame > level_max:
                    # do not create event
                    return
            elif call_stack_events[-1][EVENT_TYPE] == "return":
                level_of_call_frame = call_stack_events[-1][LEVEL_OF_CALL_FRAME]
            
            metric_name ="input_shapes"
            metric_value = "|".join([
                f"{k}={tuple(v.shape)}" for k, v in frame_args_as_dict.items() if isinstance(v, Tensor)
            ])
            
        elif event_type == "return":
            
            if len(call_stack_events) == 0:
                # return from function containing sys.profiler(prof) will trigger
                return
            elif call_stack_events[-1][EVENT_TYPE] == "call":
                level_of_call_frame = call_stack_events[-1][LEVEL_OF_CALL_FRAME]
            elif call_stack_events[-1][EVENT_TYPE] == "return":
                level_of_call_frame = call_stack_events[-1][LEVEL_OF_CALL_FRAME] - 1
            
            metric_name = "output_shapes"
            metric_value = f"NA"
            if isinstance(arg, Tensor):
                metric_value = f"{tuple(arg.shape)}"
            elif isinstance(arg, tuple):
                metric_value = "|".join([f"{tuple(v.shape)}" for v in arg if isinstance(v, Tensor)])

        frame_id = str(id(frame))
        event_dict = {
            LEVEL_OF_CALL_FRAME: level_of_call_frame,
            "class_name_short": class_name_short,
            "func_name": func_name,
            EVENT_TYPE: event_type,
            "frame_id": frame_id,
            "event_id": "|".join([class_name_short, func_name, frame_id, event_type]),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "is_class_name_in_black_list": is_class_name_in_black_list,
            "is_function_name_in_blacklist": is_function_name_in_blacklist,
            "class_name_long": class_name_long,
            "location": func_loc,
            
        }
        call_stack_events.append(event_dict)
        #print(f"{event_dict}")

    return profiler


def frame_is_class_method(frame=None):
    """
    Returns (is_method: bool, class, function_name)
    is_method = True if frame is an instance or class method
    class = the class object if available, else None
    function_name = name of the function in the frame
    """


    if frame is  None:
        return False, None, None
    else:    
        locals_ = frame.f_locals
        func_name = frame.f_code.co_name

        # Check for instance method (has 'self')
        if 'self' in locals_:
            cls = type(locals_['self'])
            class_name_long = str(cls).split("\'")[1]

            return True, class_name_long, func_name

        # Check for class method (has 'cls')
        if 'cls' in locals_:
            cls = locals_['cls']
            class_name_long = str(cls).split("\'")[1]

            return True, class_name_long, func_name

        # Static method or free function
        return False, None, func_name
    