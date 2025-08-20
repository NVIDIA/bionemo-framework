import torch
 
def get_device_and_memory_usage():
    current_device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(current_device_index)
    message = (
        f"""
        current device index: {current_device_index}
        current device uuid: {props.uuid}
        current device name: {props.name}
        memory available: {torch.cuda.mem_get_info()[0] / 1024**3} GB
        memory allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB
        """
    )
    return message