import os
import pickle
import numpy as np
import torch

# === CONFIGURATION ===
DIR1 = "intermediate_outputs_x86_64_pytorch_25_04"
DIR2 = "intermediate_outputs_x86_64_pytorch_25_01"
ATOL = 1e-3
RTOL = 1e-3

# === GET SHARED FILES ===
shared_files = set(os.listdir(DIR1)) & set(os.listdir(DIR2))

# Get creation times for files in DIR1 and sort by creation time
layer_info = []
for name in shared_files:
    path1 = os.path.join(DIR1, name)
    creation_time = os.path.getctime(path1)
    layer_info.append((creation_time, name))

# Sort by creation time
layer_info.sort(key=lambda x: x[0])
layer_names = [name for _, name in layer_info]

print(f"Comparing {len(layer_names)} shared layers (ordered by creation time)...")

for idx, name in enumerate(layer_names):
    print(f"Comparing layer #{idx} ({name})...")
    path1 = os.path.join(DIR1, name)
    path2 = os.path.join(DIR2, name)

    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        out1 = pickle.load(f1)
        out2 = pickle.load(f2)

    def to_array(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            return obj
        elif isinstance(obj, tuple):
            # Convert and filter out unsupported
            return [to_array(o) for o in obj if isinstance(o, (torch.Tensor, np.ndarray))]
        else:
            return None


    arr1 = to_array(out1)
    arr2 = to_array(out2)

    def compare(a, b):
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return all(
                isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and np.allclose(x, y, atol=ATOL, rtol=RTOL)
                for x, y in zip(a, b)
            )
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.allclose(a, b, atol=ATOL, rtol=RTOL)
        else:
            return False

    if not compare(arr1, arr2):
        def max_diff(a, b):
            if isinstance(a, list) and isinstance(b, list):
                return max(
                    np.max(np.abs(x - y))
                    for x, y in zip(a, b)
                    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
                )
            elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return np.max(np.abs(a - b))
            else:
                return float("nan")

        diff = max_diff(arr1, arr2)
        print(f"\n❌ Divergence at layer #{idx}: {name}")
        print(f"   Max difference: {diff:.5f}")

    else:
        print(f"✅ Layer #{idx} ({name}) matched")

print("Done.")