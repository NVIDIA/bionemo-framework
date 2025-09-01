import torch
import torch.profiler


import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple module: Linear -> ReLU -> Linear
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleModel()

# Generate random input data (batch_size=4, input_size=10)
x = torch.randn(4, 10)


with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    with_stack=True
) as prof:
    for _ in range(5):
        output = model(x)
        
        print("Input:", x)
        print("Output:", output)

print(
    prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=10
    )
)
