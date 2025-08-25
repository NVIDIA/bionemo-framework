#!/bin/bash
# comment


# files_to_check=(
#     src/boltz/distributed/model/layers/distribute_module_tools.py
#     src/boltz/distributed/model/layers/swiglu.py
#     src/boltz/distributed/model/layers/mult_for_same_placement_and_shape.py
#     tests/model/layers/test_dtensor_swiglu.py
#     src/boltz/testing/utils.py
# )
# files_to_check=(
#     src/boltz/distributed/model/layers/distribute_module_tools.py
#     src/boltz/distributed/model/layers/layernorm.py
#     tests/distributed/test_dtensor_layernorm.py
# )

# files_to_check=(
#     src/boltz/distributed/model/layers/attention.py
#     src/boltz/distributed/model/layers/attention_impl.py
#     src/boltz/distributed/model/layers/distribute_module_tools.py
#     src/boltz/distributed/model/layers/dtensor_metadata_tools.py
#     tests/distributed/model/layers/test_attention_with_dtensor_for_pairformer_use_case.py
# )

# files_to_check=(
#     src/boltz/distributed/model/layers/dtensor_metadata_tools.py
#     tests/distributed/test_dtensor_metadata_tools.py
# )

files_to_check=(
    sub-packages/bionemo-testing/src/bionemo/testing/torch.py
    sub-packages/bionemo-evo2/tests/bionemo/evo2/test_evo2.py
    sub-packages/bionemo-evo2/tests/bionemo/evo2/conftest.py
    sub-packages/bionemo-testing/tests/bionemo/testing/test_torch.py
)


for file in "${files_to_check[@]}"; do
    echo "Checking $file"
    pre-commit run --files $file
done

