# C2Ray is pure Fortran and does not depend on CUDA, so it is always built.
subdir('c2ray')

# The CUDA compiler is optional: without it the ASORA libraries are skipped.
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
        cc = r_compute_cap.stdout().strip()
        message(f'Local compute capability is determined to be @cc@')
        compute_cap = cc[0] + '0'
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
]
nvlink_flags = [compute_cap]

if get_option('buildtype') in ['release', 'minsize']
    nvcc_flags += ['-dlto']
    nvlink_flags += ['-dlto']
endif

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

# Every ASORA target needs both: the toolkit itself and the CCCL include fixup.
asora_cuda_deps = [cuda_dep, cccl_dep]

subdir('asora')
subdir('asora_he')
