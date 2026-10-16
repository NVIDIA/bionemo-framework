#!/bin/bash
# Source this before building CUDA extensions from source against PyTorch 26.09+.
#
# PyTorch 26.09 headers require C++20, but nv-grouped-gemm, causal-conv1d and mamba-ssm hardcode
# -std=c++17 in their extension flags (#error C++20 or later compatible compiler is required). This
# routes torch.utils.cpp_extension through compiler wrappers (via PYTORCH_NVCC and CXX) that upgrade
# that flag to -std=c++20. Remove once those packages build with C++20 upstream.

_cxx20_shim_dir="$(mktemp -d)"

_cxx20_write_wrapper() {
    _cxx20_name="$1"
    _cxx20_real="$2"
    cat > "${_cxx20_shim_dir}/${_cxx20_name}" <<EOF
#!/bin/bash
args=()
for arg in "\$@"; do
    case "\$arg" in
        -std=c++17) arg=-std=c++20 ;;
        --std=c++17) arg=--std=c++20 ;;
    esac
    args+=("\$arg")
done
exec "${_cxx20_real}" "\${args[@]}"
EOF
    chmod +x "${_cxx20_shim_dir}/${_cxx20_name}"
}

_cxx20_write_wrapper nvcc "${PYTORCH_NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}"
_cxx20_write_wrapper c++ "$(command -v "${CXX:-c++}")"
export PYTORCH_NVCC="${_cxx20_shim_dir}/nvcc"
export CXX="${_cxx20_shim_dir}/c++"
