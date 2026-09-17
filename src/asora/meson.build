# CUDA compiler is optional, required to build ASORA
if not has_cuda
    subdir_done()
endif

# Get CUDA architecture to be used for compilation
compute_cap = get_option('gpu-architecture')
if compute_cap == ''
    message('CUDA compute capability was not specified by the user')
    r_compute_cap = run_command(
        'nvidia-smi',
        '--query-gpu=compute_cap',
        '--format=csv,noheader',
        check: false,
    )
    if r_compute_cap.returncode() != 0
        warning('nvidia-smi not detected; defaulting to compute capability 6.0')
        compute_cap = '60'
    else
        # With several GPUs present, nvidia-smi reports one line each; use the first.
        cc = r_compute_cap.stdout().strip().split('\n')[0].strip()
        message(f'Local compute capability is determined to be @cc@')
        # Keep the minor version: 8.9 is sm_89, not sm_80.
        cc_parts = cc.split('.')
        if cc_parts.length() < 2
            warning('Unexpected compute capability format; defaulting to 6.0')
            compute_cap = '60'
        else
            compute_cap = cc_parts[0] + cc_parts[1]
        endif
    endif
endif

# Target only one GPU
compute_cap = '-arch=sm_' + compute_cap

nvcc_flags = [
    compute_cap,
    '-rdc=true',
    '-DPERIODIC',
    '-DLOCALRATES',
    '-use_fast_math',
    '--generate-line-info',
]
nvlink_flags = [compute_cap]

if get_option('buildtype') in ['release', 'minsize']
    nvcc_flags += ['-dlto']
    nvlink_flags += ['-dlto']
endif

# CUDA sources for ASORA
asora_sources = [
    'memory.cpp',
    'rates.cu',
    'raytracing.cu',
    'shortchar.cu',
    'utils.cu',
    'chemistry.cu',
    'python_module.cu',
]
cuda_dep = dependency('cuda', version: '>=12', static: false)

# CUDA 13 relocated the CCCL headers from <root>/include to <root>/include/cccl.
# It must be manually added for host-compiled sources.
cccl_args = []
if cuda_dep.version().version_compare('>= 13.0')
    nvcc = find_program('nvcc', required: true)
    cuda_root = fs.parent(fs.parent(nvcc.full_path()))
    cccl_dir = cuda_root / 'include' / 'cccl'
    if not fs.is_dir(cccl_dir)
        error(
            'CUDA @0@ requires the CCCL headers at @1@, which is missing. Install the cuda-cccl package for this toolkit.'.format(
                cuda_dep.version(),
                cccl_dir,
            ),
        )
    endif
    cccl_args += '-isystem' + cccl_dir
endif
cccl_dep = declare_dependency(compile_args: cccl_args)

# Build ASORA module with nvcc
py.extension_module(
    'libasora',
    asora_sources,
    dependencies: [py_dep, np_dep, cuda_dep, cccl_dep],
    install: true,
    subdir: 'pyc2ray/lib',
    cuda_args: nvcc_flags,
    link_args: nvlink_flags,
)
