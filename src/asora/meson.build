# CUDA setup (compute capability, nvcc/nvlink flags, cuda_dep, cccl_dep) is shared
# with asora_he and lives in src/meson.build.

# CUDA sources for ASORA
asora_sources = ['memory.cpp', 'raytracing.cu', 'utils.cu', 'chemistry.cu']

asora_core = static_library(
    'asora_core',
    asora_sources,
    dependencies: asora_cuda_deps,
    cuda_args: nvcc_flags,
    link_args: nvlink_flags,
)

# Build ASORA module with nvcc
py.extension_module(
    'libasora',
    'python_module.cu',
    dependencies: [py_dep, np_dep] + asora_cuda_deps,
    install: true,
    subdir: 'pyc2ray/lib',
    link_with: asora_core,
    cuda_args: nvcc_flags,
    link_args: nvlink_flags,
)

# Build ASORA test module with nvcc
py.extension_module(
    'libasoratest',
    ['tests.cu', 'python_test_module.cu'],
    dependencies: [py_dep, np_dep] + asora_cuda_deps,
    install: true,
    subdir: 'pyc2ray/lib',
    link_with: asora_core,
    cuda_args: nvcc_flags,
    link_args: nvlink_flags,
)
