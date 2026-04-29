from __future__ import annotations

import importlib
import math
from typing import List, Optional

import numpy as np
import pyc2ray.domain.domain_decomposition_utils as dd_utils

def generate_sources(num_cluster_sources: int = 70, cluster_center: np.ndarray = None, cluster_width: float = 0.02,
                     num_sparse_sources: int = 30, source_strength: float = 1.0, r_max_lls: float = 12.0, boxsize: float = 100.0):
    """
    Generate toy sources in a clustered + sparse configuration.

    All positions and radii are expressed in comoving Mpc/h.

    Parameters
    ----------
    num_cluster_sources : int
        Number of sources in the dense cluster.
    cluster_center : Optional[np.ndarray]
        Center of the cluster (shape `(3,)`). If None, defaults to the box center.
    cluster_width : float
        Standard deviation of the cluster in each dimension, as a fraction of the box size.
    num_sparse_sources : int
        Number of sources uniformly distributed across the box outside the cluster.
    source_strength : float
        Emissivity/intensity assigned to each source.
    r_max_lls : float
        Maximum tracing/influence radius for each source, based on the Lyman-limit-system mean free path.
    boxsize : float
        Size of the cubic domain in comoving Mpc/h.

    Returns
    -------
    List[Source]   List of generated sources with positions, strengths, and radii.
    """
    rng = np.random.default_rng(23)

    sources = []

    # Dense cluster
    if cluster_center is None:
        cluster_center = np.array([0.5, 0.5, 0.5]) * boxsize
    cluster_sigma = cluster_width * boxsize

    for i in range(num_cluster_sources):
        pos = cluster_center + rng.normal(scale=cluster_sigma, size=3)
        strength = source_strength
        sources.append(dd_utils.Source(pos=pos, strength=strength, radius=r_max_lls, gid=i))

    # Sparse sources across the full box
    for i in range(num_cluster_sources, num_cluster_sources + num_sparse_sources):
        pos = rng.uniform(0, boxsize, size=3)
        strength = source_strength
        sources.append(dd_utils.Source(pos=pos, strength=strength, radius=r_max_lls, gid=i))

    return sources

def _box_faces(pmin: np.ndarray, pmax: np.ndarray):
    """
    Return the six faces of a box.
    
    Parameters
    ----------
    pmin : np.ndarray
        Minimum corner of the box (shape `(3,)`).
    pmax : np.ndarray
        Maximum corner of the box (shape `(3,)`).

    Returns
    -------
    List[List[np.ndarray]]
        List of 6 faces, each face is a list of 4 corner points (shape `(3,)`).
    """
    x0, y0, z0 = pmin
    x1, y1, z1 = pmax
    v = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    return [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],
        [v[1], v[2], v[6], v[5]],
        [v[2], v[3], v[7], v[6]],
        [v[3], v[0], v[4], v[7]],
    ]


def plot_grid_and_sources(
    grid: dd_utils.Grid,
    sources: List[dd_utils.Source],
    groups: Optional[List[dd_utils.Group]] = None,
    plot_sources: bool = True,
    plot_groups: bool = True,
    plot_bbox: bool = True,
):
    """
    Visualize grid patches, sources, and optional group envelopes in 3D.
    
    Parameters    
    ----------
    grid : Grid
        Grid model with patches to visualize.
    sources : List[Source]
        List of sources to plot as points.
    groups : Optional[List[Group]]
        Optional list of groups to plot enclosing spheres and bounding boxes.
    plot_sources : bool
        Whether to plot source positions as points.
    plot_groups : bool
        Whether to plot group enclosing spheres.
    plot_bbox : bool
        Whether to plot group bounding boxes.
        
    """
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        art3d = importlib.import_module("mpl_toolkits.mplot3d.art3d")
        Poly3DCollection = art3d.Poly3DCollection
    except ImportError:
        print("Matplotlib is not available. Skipping grid plot.")
        return

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the uniform grid as a single translucent box.
    pmin = np.asarray(grid.get_domain_min(), dtype=float)
    pmax = np.asarray(grid.get_domain_max(), dtype=float)
    if pmin.ndim == 0:
        pmin = np.repeat(pmin, 3)
    if pmax.ndim == 0:
        pmax = np.repeat(pmax, 3)
    faces = _box_faces(pmin, pmax)
    poly = Poly3DCollection(
        faces,
        facecolors=(0.0, 0.0, 0.0, 0.0),
        edgecolors=(0.15, 0.55, 0.85, 0.55),
        linewidths=0.7,
    )
    ax.add_collection3d(poly)

    # Plot sources as points, optionally colored by group membership.
    if plot_sources:
        pos = np.array([s.pos for s in sources], dtype=float)
        if groups:
            gid_to_group = {}
            for ig, g in enumerate(groups):
                for s in g.sources:
                    gid_to_group[s.gid] = ig

            group_ids = np.array([gid_to_group.get(s.gid, -1) for s in sources], dtype=int)
            n_groups = max(len(groups), 1)
            # Separate colors by group
            hues = np.mod(np.arange(n_groups) * 0.61803398875, 1.0)
            sat = np.full(n_groups, 0.9)
            val = np.full(n_groups, 0.95)
            hsv = np.stack([hues, sat, val], axis=1)
            colors_mod = importlib.import_module("matplotlib.colors")
            palette = colors_mod.hsv_to_rgb(hsv)
            default_color = np.array([0.45, 0.45, 0.45, 1.0])
            colors = np.array([
                np.append(palette[gidx], 1.0) if gidx >= 0 else default_color
                for gidx in group_ids
            ])
            ax.scatter(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                c=colors,
                s=22,
                depthshade=True,
                label="sources (by group)",
            )
        else:
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="crimson", s=18, depthshade=True, label="sources")

    # # Single-source influence spheres as faint wireframes.
    # u_src = np.linspace(0.0, 2.0 * math.pi, 14)
    # v_src = np.linspace(0.0, math.pi, 10)
    # uu_src, vv_src = np.meshgrid(u_src, v_src)
    # for i, s in enumerate(sources):
    #     x0, y0, z0 = s.pos
    #     r0 = s.radius
    #     xs = x0 + r0 * np.cos(uu_src) * np.sin(vv_src)
    #     ys = y0 + r0 * np.sin(uu_src) * np.sin(vv_src)
    #     zs = z0 + r0 * np.cos(vv_src)
    #     label = "source sphere" if i == 0 else None
    #     ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, color="tab:green", linewidth=1., alpha=0.10, label=label)

    # Plot group envelopes as larger wireframes.
    if groups and plot_groups:
        u = np.linspace(0.0, 2.0 * math.pi, 22)
        v = np.linspace(0.0, math.pi, 14)
        uu, vv = np.meshgrid(u, v)
        for i, g in enumerate(groups):
            if plot_bbox:
                # Bounding box of the group (edge-only for readability).
                bb_faces = _box_faces(g.bbox_min, g.bbox_max)
                bb = Poly3DCollection(
                    bb_faces,
                    facecolors=(0.0, 0.0, 0.0, 0.0),
                    edgecolors=(1.0, 0.55, 0.0, 0.7),
                    linewidths=0.8,
                )
                ax.add_collection3d(bb)

            # Enclosing sphere wireframe.
            cx, cy, cz = g.center
            r = g.radius
            xs = cx + r * np.cos(uu) * np.sin(vv)
            ys = cy + r * np.sin(uu) * np.sin(vv)
            zs = cz + r * np.cos(vv)
            label = "group sphere" if i == 0 else None
            ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, color="tab:blue", linewidth=0.45, alpha=0.25, label=label)

    dmin = np.asarray(grid.get_domain_min(), dtype=float)
    dmax = np.asarray(grid.get_domain_max(), dtype=float)
    if dmin.ndim == 0:
        dmin = np.repeat(dmin, 3)
    if dmax.ndim == 0:
        dmax = np.repeat(dmax, 3)
    ax.set_xlim(dmin[0], dmax[0])
    ax.set_ylim(dmin[1], dmax[1])
    ax.set_zlim(dmin[2], dmax[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Variable-resolution grid, source spheres, and group envelopes")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def main():
    """Run a complete toy example: source generation, grouping, and plotting."""
    # ========================================================
    # Physical / cosmological properties of the domain
    # ========================================================
    # Box size in cMpc/h
    boxsize = 100.0

    # Source distribution parameters
    num_cluster_sources = 10
    cluster_center = None  # default is 50% of box size
    cluster_width = 0.05   # fraction of box size
    num_sparse_sources = 0
    source_strength = 1.0

    # Grid structure
    num_cells = 256
    cell_dx = boxsize / num_cells

    # Maximum tracing radius from the Lyman-limit-system
    # mean free path. Same units as the domain: comoving Mpc/h.
    lambda_mfp_mpc_h = 20.0
    r_max_lls = lambda_mfp_mpc_h

    r_max_lls_cells = lambda_mfp_mpc_h / cell_dx

    # Create source distribution
    sources = generate_sources(
        num_cluster_sources=num_cluster_sources,
        cluster_center=cluster_center,
        cluster_width=cluster_width,
        num_sparse_sources=num_sparse_sources,
        source_strength=source_strength,
        r_max_lls=r_max_lls,
        boxsize=boxsize,
    )

    # Create variable-resolution grid
    grid = dd_utils.Grid(num_cells = num_cells, dx = cell_dx)

    # Build groups with a uniform radius constraint based on the LLS mean free path.
    groups = dd_utils.build_groups(
        sources=sources,
        grid=grid,
        r_group_max=1.5 * r_max_lls,
        nsrc_max=12,
        ncell_max=80000,
        cost_max=500000,
    )

    # Distribute groups across ranks by estimated cost.
    rank_groups, rank_costs = dd_utils.assign_groups_to_ranks(groups, nranks=4)

    # Report results and plot. 
    print(f"Toy box size                = {boxsize:.1f} cMpc/h")
    print(f"Cell size                   = {cell_dx:.3f} cMpc/h")
    print(f"Mean free path (LLS)        = {lambda_mfp_mpc_h:.1f} cMpc/h")
    print(f"R_max_LLS (cells)           = {r_max_lls_cells:.1f}\n")

    print(f"Total sources: {len(sources)}")
    print(f"Total groups:  {len(groups)}\n")
    print(f"Using a uniform source radius R_max_LLS = {r_max_lls:.2f} cMpc/h\n")

    for ig, g in enumerate(groups):
        print(
            f"Group {ig:02d}: nsrc={len(g.sources):2d}, "
            f"R={g.radius:6.2f}, ncell={g.get_num_cells():7d}, cost={g.cost:10.1f}"
        )

    print("\nAssignment to ranks:")
    for r, gs in enumerate(rank_groups):
        print(f"Rank {r}: {len(gs)} groups, total cost = {rank_costs[r]:.1f}")
        for g in gs:
            print(
                f"   nsrc={len(g.sources):2d}, R={g.radius:6.2f}, "
                f"ncell={g.get_num_cells():7d}, cost={g.cost:10.1f}"
            )

    plot_grid_and_sources(grid, sources, groups, )

if __name__ == "__main__":
    main()
