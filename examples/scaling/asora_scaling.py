"""ASORA raytracing scaling benchmark (source-split, as in _evolve3D_asora).

Strong scaling: --mode strong, --num-sources is the global number of sources.
Weak scaling:   --mode weak,   --num-sources is the number of sources per rank.

Single GPU:
    python asora_scaling.py --mode strong -n 100000 -N 256 -r 15
Multi GPU (Alps):
    srun -N 2 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=socket \
        python asora_scaling.py --mode weak -n 25000 -N 256 -o results.jsonl

With -o, each run appends one JSON record (JSON Lines) to the output file.
Load with: pandas.read_json("results.jsonl", lines=True)
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import astropy.constants as cst
import astropy.units as u
import numpy as np
from mpi4py import MPI

from pyc2ray import asora_core
from pyc2ray.load_extensions import libasora
from pyc2ray.radiation.blackbody import BlackBodySource
from pyc2ray.radiation.common import make_tau_table
from pyc2ray.utils.other_utils import distribute_jobs

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


@dataclass
class Times:
    """Per-rank timings, one entry per repeat."""

    raytracing: list[float] = field(default_factory=list)
    imbalance: list[float] = field(default_factory=list)
    reduction: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)

    def add(
        self, raytracing: float, imbalance: float, reduction: float, total: float
    ) -> None:
        self.raytracing.append(raytracing)
        self.imbalance.append(imbalance)
        self.reduction.append(reduction)
        self.total.append(total)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("-n", "--num-sources", type=int, default=10000)
    p.add_argument("-N", "--mesh-size", type=int, default=128)
    p.add_argument("-r", "--radius", type=float, default=15.0, help="R_max in cells")
    p.add_argument("-b", "--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=918)
    p.add_argument(
        "-o", "--output", type=Path, default=None, help="JSON Lines file to append to"
    )
    p.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        help="Overwrite any existing JSON Lines file; ignored without --output",
    )
    return p.parse_args()


def setup_tables() -> tuple[float, float, float, int]:
    """Build and upload the photoionization tables (same as tests/test_raytracing.py)."""
    minlog_tau = -20.0
    maxlog_tau = 4.0
    num_tau = 20000
    tau, dlogtau = make_tau_table(minlog_tau, maxlog_tau, num_tau)

    sigma = 6.30e-18
    freq_min = (13.598 * u.eV / cst.h).to("Hz").value
    freq_max = (54.416 * u.eV / cst.h).to("Hz").value
    radsource = BlackBodySource(1e5, False, freq_min, sigma)
    thin_table, thick_table = radsource.make_photo_table(tau, freq_min, freq_max, 1e48)
    asora_core.photo_table_to_device(thin_table, thick_table)

    return sigma, minlog_tau, dlogtau, num_tau


def make_sources(num_total: int, mesh_size: int, seed: int):
    """Same global source list on every rank (deterministic seed)."""
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, mesh_size, size=(num_total, 3), dtype=np.int32)
    flux = rng.uniform(1e4, 1e8, size=num_total)
    return pos, flux


def run_once(rt_args: tuple, phi_ion: np.ndarray) -> tuple[float, float, float, float]:
    """Return (raytracing, imbalance, reduction, total) wall-clock times."""
    assert libasora is not None
    comm.Barrier()
    t0 = time.perf_counter()
    libasora.do_all_sources(*rt_args)  # blocking: returns after phi_ion is on host
    t1 = time.perf_counter()
    comm.Barrier()
    t2 = time.perf_counter()
    if nprocs > 1:
        comm.Allreduce(MPI.IN_PLACE, [phi_ion, MPI.DOUBLE], op=MPI.SUM)
    t3 = time.perf_counter()

    # raytracing, imbalance, reduction, total
    return t1 - t0, t2 - t1, t3 - t2, t3 - t0


def benchmark(args: argparse.Namespace, num_total: int) -> dict[str, Any]:
    sigma, minlog_tau, dlogtau, num_tau = setup_tables()

    # Uniform fields: the raytracing cost does not depend on their values.
    mesh_size = args.mesh_size
    size = mesh_size**3
    ndens = np.full(size, 1e-3)
    xHII = np.full(size, 1e-4)
    phi_ion = np.zeros(size)
    asora_core.density_to_device(ndens)
    asora_core.average_fraction_to_device(xHII)

    src_pos, src_flux = make_sources(num_total, mesh_size, args.seed)
    chunk = distribute_jobs(num_total, nprocs, rank)
    num_local = chunk.stop - chunk.start
    asora_core.source_data_to_device(src_pos[chunk], src_flux[chunk])

    dr = (50.0 * u.pc / mesh_size).cgs.value
    rt_args = (
        args.radius,
        sigma,
        dr,
        phi_ion,
        num_local,
        mesh_size,
        minlog_tau,
        dlogtau,
        num_tau,
        args.batch_size,
        args.block_size,
    )

    for _ in range(args.warmup):
        run_once(rt_args, phi_ion)

    times = Times()
    for _ in range(args.repeats):
        times.add(*run_once(rt_args, phi_ion))

    return {
        "num_sources": num_local,
        "timings": asdict(times),
    }


def summarize(
    args: argparse.Namespace, num_total: int, per_rank: list[dict[str, Any]]
) -> dict[str, Any]:
    # Arrays of shape (nprocs, repeats)
    timings = {
        f.name: np.array([r["timings"][f.name] for r in per_rank])
        for f in fields(Times)
    }

    # Max over ranks for each repeat, then median over repeats
    summary = {k: float(np.median(v.max(axis=0))) for k, v in timings.items()}
    rt = timings["raytracing"]
    summary["load_imbalance"] = float(np.median(rt.max(axis=0) / rt.mean(axis=0)))

    return {
        "config": vars(args) | {"output": str(args.output)},
        "nprocs": nprocs,
        "nodes": sum(r["local_rank"] == 0 for r in per_rank),
        "num_sources_total": num_total,
        "per_rank": per_rank,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    if libasora is None:
        raise RuntimeError("libasora not available")

    num_total = args.num_sources
    if args.mode == "weak":
        num_total *= nprocs
    if num_total < nprocs:
        raise ValueError(f"{num_total} sources for {nprocs} ranks")

    local_rank = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    asora_core.device_init(local_rank)
    try:
        local = benchmark(args, num_total)
        local["local_rank"] = local_rank
    finally:
        asora_core.device_close()

    per_rank = comm.gather(local, root=0)
    if rank != 0:
        return

    assert isinstance(per_rank, list)
    result = summarize(args, num_total, per_rank)
    s = result["summary"]
    print(
        f"mode={args.mode} p={nprocs} nodes={result['nodes']} N={args.mesh_size} "
        f"R={args.radius} Nsrc={num_total} batch={args.batch_size}\n"
        f"  raytracing {s['raytracing']:.4f} s | imbalance {s['imbalance']:.4f} s | "
        f"reduction {s['reduction']:.4f} s | total {s['total']:.4f} s | "
        f"max/mean {s['load_imbalance']:.3f}"
    )
    if args.output is not None:
        # JSON Lines: one self-describing record per run, appended
        with args.output.open("w" if args.overwrite else "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
