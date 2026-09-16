# CUDA setup (compute capability, nvcc/nvlink flags, cuda_dep, cccl_dep) is shared
# with asora and lives in src/meson.build.

# CUDA sources for ASORA-He: utils and memory management are reused from asora.
asora_he_sources = [
    '../asora/utils.cu',
    '../asora/memory.cpp',
    'raytracing.cu',
    'chemistry.cu',
]

asora_he_core = static_library(
    'asora_he_core',
    asora_he_sources,
    dependencies: asora_cuda_deps,
    cuda_args: nvcc_flags,
    link_args: nvlink_flags,
)

# Build ASORA-He module with nvcc
py.extension_module(
    'libasora_He',
    'python_module.cpp',
    dependencies: [py_dep, np_dep] + asora_cuda_deps,
    install: true,
    subdir: 'pyc2ray/lib',
    link_with: asora_he_core,
    cuda_args: nvcc_flags,
    link_args: nvlink_flags,
)
