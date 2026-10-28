import argparse
import math
import subprocess
from typing import Any


def format_bytes(size: float) -> str:
    """Format bytes into human readable format"""
    for unit in ("", "Ki", "Mi", "Gi"):
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:3.2f} {unit}B"


def get_memory_available(gpu_id: int = 0) -> int:
    """Return the memory availble to the user in MiB"""
    ret = subprocess.check_output(
        (
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.total,memory.reserved",
            "--format=csv,noheader,nounits",
        )
    )

    mem_total, _, mem_reserve = ret.decode("utf-8").partition(",")
    return (int(mem_total) - int(mem_reserve)) * 1024**2


def get_q_max(radius: float, mesh_size: int) -> int:
    return math.ceil(min(math.sqrt(3) * radius, 1.5 * mesh_size))


def get_cd_buffer(radius: float, mesh_size: int) -> int:
    q = get_q_max(radius, mesh_size)
    last_shell = 4 * q**2 + 2
    return 4 * last_shell


def get_lut_length(radius: float, mesh_size: int) -> int:
    q = get_q_max(radius, mesh_size)
    return (1 + 2 * q) * (3 + 2 * q * (1 + q)) // 3


def main(
    gpu_id: int,
    mesh_size: int,
    num_sources: int,
    table_size: int,
    radius: float = 10.0,
    with_helium: bool = False,
    with_lut: bool = True,
    use_single_precision: bool = False,
) -> None:
    """Print memory requirements for the input parameters"""
    print("Querying GPU memory...")
    memory_total = get_memory_available(gpu_id)

    print(
        f"""
Required parameters:
 - mesh size: {mesh_size}
 - number of sources: {num_sources:,}
 - look-up table size: {table_size}
 - with helium: {"yes" if with_helium else "no"}"""
    )

    H_only = not with_helium

    prec = 4 if use_single_precision else 8

    # Memory for a full grid
    grid_mem = prec * mesh_size**3

    # Memory for sources (3 pos, 1 flux)
    ns_mem = (12 + prec) * num_sources

    # Memory for look-up tables
    # - H only:  photo ionization, thick and thin
    # - with He: photo ionization and heating, thick and thin
    pt_mem = 2 * prec * table_size * (1 if H_only else 2)

    # Memory for number density
    nd_mem = grid_mem

    # Memory for ionized fractions
    # - H only:  HII
    # - with He: HII, HeII, HeIII
    xs_mem = grid_mem * (1 if H_only else 3)

    # Memory for photoionization grids
    # - H only:  HI
    # - with He: HI, HeI, HeII
    phi_mem = grid_mem * (1 if H_only else 4)

    # Frequency bins are hard-coded
    num_freq_bins = 47

    # Memory for cross-section histograms
    xsec_mem = prec * num_freq_bins * (0 if H_only else 3)

    # Seconday ionization tables
    snd_mem = prec * num_freq_bins * (0 if H_only else 12)

    # Memory for raytracing LUT
    lut_mem = prec * get_lut_length(radius, mesh_size) * (8 if with_lut else 0)

    # Memory for column density for one source
    # - H only:  HII
    # - with He: HII, HeII, HeIII
    cd_mem = prec * get_cd_buffer(radius, mesh_size) * (1 if H_only else 3)

    print(
        f"""
GPU {gpu_id} has total memory of {format_bytes(memory_total)}

> Memory required for sources:                  {format_bytes(ns_mem)}
> Memory required for photo-tables:             {format_bytes(pt_mem)}
> Memory required for density mesh:             {format_bytes(grid_mem)}
> Memory required for fraction meshes:          {format_bytes(xs_mem)}
> Memory required for photoionization grids:    {format_bytes(xs_mem)}
> Memory required for short-char LUT:           {format_bytes(lut_mem)}
> Memory required for cross-section histograms: {format_bytes(xsec_mem)}
> Memory required for 2nd ionization factors:   {format_bytes(snd_mem)}"""
    )
    base_mem = (
        ns_mem + pt_mem + nd_mem + xs_mem + phi_mem + xsec_mem + snd_mem + lut_mem
    )

    print(
        f"""
> Baseline memory required: {format_bytes(base_mem)}
> Buffer memory required per source: {format_bytes(cd_mem)}"""
    )

    free_mem = memory_total - base_mem
    if free_mem < 0:
        print("\nNot enough memory for these parameters.")
        return

    batches = free_mem / cd_mem
    waves = math.ceil(num_sources / batches)
    print(
        f"""
Available memory {format_bytes(free_mem)} can fit a batch size of {batches:.0f}
Number of waves: {waves:,}
"""
    )


def parse_args() -> dict[str, Any]:
    """Parse the CLI arguments"""
    parser = argparse.ArgumentParser(
        prog="memory_estimate",
        description="Estimate py2Cray raytracing memory requirements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-sources",
        type=int,
        default=1_000_000,
        help="number of sources",
    )
    parser.add_argument(
        "-m",
        "--mesh-size",
        type=int,
        default=256,
        help="size of the grid",
    )
    parser.add_argument(
        "-t",
        "--table-size",
        type=int,
        default=2000,
        help="size of each look-up tables",
    )
    parser.add_argument(
        "--with-helium",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include He simulation in the estimation",
    )
    parser.add_argument(
        "--with-lut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include raytracing look-up table in the estimation",
    )
    parser.add_argument(
        "-R",
        "--radius",
        type=float,
        default=10.0,
        help="set maximum propagation radius for photons",
    )
    parser.add_argument(
        "--use-single-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use single precision floating point",
    )
    parser.add_argument(
        "-i", "--gpu-id", type=int, default=0, help="specify which GPU device"
    )

    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()

    main(**args)
