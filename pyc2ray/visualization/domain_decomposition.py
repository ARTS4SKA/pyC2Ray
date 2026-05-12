from __future__ import annotations

from pathlib import Path
from typing import Any, List, cast

import matplotlib.pyplot as plt
import numpy as np

from pyc2ray.domain.regular_grid import RegularGrid
from pyc2ray.domain.sources import SourceGroup

def _box_corners(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [box_min[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_max[0], box_max[1], box_max[2]],
            [box_min[0], box_max[1], box_max[2]],
        ],
        dtype=float,
    )


def _plot_box(ax, box_min: np.ndarray, box_max: np.ndarray, **line_kwargs) -> None:
    if np.any(box_max <= box_min):
        return

    corners = _box_corners(box_min, box_max)
    edge_pairs = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )

    for edge_index, (start_idx, end_idx) in enumerate(edge_pairs):
        edge_kwargs = dict(line_kwargs)
        if edge_index > 0:
            edge_kwargs.pop("label", None)
        start = corners[start_idx]
        end = corners[end_idx]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            **edge_kwargs,
        )


def plot_domain_decomposition(
    global_grid: RegularGrid,
    source_groups: List[SourceGroup],
    local_grids: List[RegularGrid],
    show: bool = True,
):
    fig = plt.figure(figsize=(10, 8))
    ax = cast(Any, fig.add_subplot(111, projection="3d"))

    global_min = global_grid.get_domain_min().astype(float)
    global_max = global_grid.get_domain_max().astype(float)
    domain_extent = np.maximum(global_max - global_min, 1.0)
    axis_padding = 0.05 * domain_extent
    label_offset = 0.02 * domain_extent
    _plot_box(
        ax,
        global_min,
        global_max,
        color="black",
        linewidth=2.0,
        alpha=0.9,
        label="Global grid",
    )

    color_map = plt.get_cmap("tab20")
    num_items = max(len(source_groups), len(local_grids))
    for index in range(num_items):
        color = color_map(index % color_map.N)

        if index < len(local_grids) and local_grids[index].num_cells > 0:
            local_min = local_grids[index].get_domain_min().astype(float)
            local_max = local_grids[index].get_domain_max().astype(float)
            _plot_box(
                ax,
                local_min,
                local_max,
                color=color,
                linewidth=1.5,
                alpha=0.8,
                label="Local grid" if index == 0 else None,
            )

        if index < len(source_groups):
            group = source_groups[index]
            _plot_box(
                ax,
                group.bbox_min.astype(float),
                group.bbox_max.astype(float),
                color=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.5,
                label="Group bounding box" if index == 0 else None,
            )

            if group.sources:
                positions = np.array([source.pos for source in group.sources], dtype=float)
                ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    color=[color],
                    s=30,
                    depthshade=False,
                    label="Sources" if index == 0 else None,
                )

            ax.text(
                min(group.bbox_max[0] + label_offset[0], global_max[0] + axis_padding[0]),
                min(group.bbox_max[1] + label_offset[1], global_max[1] + axis_padding[1]),
                min(group.bbox_max[2] + label_offset[2], global_max[2] + axis_padding[2]),
                str(group.id),
                color=color,
                fontsize=9,
            )

    ax.set_xlim(global_min[0] - axis_padding[0], global_max[0] + axis_padding[0])
    ax.set_ylim(global_min[1] - axis_padding[1], global_max[1] + axis_padding[1])
    ax.set_zlim(global_min[2] - axis_padding[2], global_max[2] + axis_padding[2])
    ax.set_box_aspect(tuple(domain_extent.tolist()))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"Domain decomposition: {len(source_groups)} groups, {len(local_grids)} local grids"
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig("domain_decomposition.png", dpi=300, bbox_inches="tight")
    return fig, ax

def export_domain_decomposition_npz(
    global_grid: RegularGrid,
    source_groups: List[SourceGroup],
    local_grids: List[RegularGrid],
    output_path: str | Path = "domain_decomposition_data.npz",
) -> Path:
    """Export decomposition data to a compressed NumPy archive.

    The resulting file can be copied to another machine and opened with
    ``np.load(..., allow_pickle=False)`` for interactive inspection.
    """
    out = Path(output_path)

    group_ids: list[int] = []
    group_centers: list[np.ndarray] = []
    group_radii: list[float] = []
    group_bbox_min: list[np.ndarray] = []
    group_bbox_max: list[np.ndarray] = []
    group_source_counts: list[int] = []

    source_ids: list[int] = []
    source_positions: list[np.ndarray] = []
    source_group_ids: list[int] = []

    for group in source_groups:
        group_ids.append(int(group.id))
        group_centers.append(group.center.astype(float))
        group_radii.append(float(group.radius))
        group_bbox_min.append(group.bbox_min.astype(float))
        group_bbox_max.append(group.bbox_max.astype(float))
        group_source_counts.append(len(group.sources))

        for source in group.sources:
            source_ids.append(int(source.id))
            source_positions.append(np.asarray(source.pos, dtype=float))
            source_group_ids.append(int(group.id))

    local_offsets = np.array([grid.offset for grid in local_grids], dtype=np.int64)
    local_num_cells = np.array([grid.num_cells for grid in local_grids], dtype=np.int64)

    np.savez_compressed(
        out,
        global_cell_size=float(global_grid.cell_size),
        global_num_cells=np.asarray(global_grid.num_cells, dtype=np.int64),
        global_offset=np.asarray(global_grid.offset, dtype=np.int64),
        global_domain_min=global_grid.get_domain_min().astype(float),
        global_domain_max=global_grid.get_domain_max().astype(float),
        group_ids=np.array(group_ids, dtype=np.int64),
        group_centers=np.array(group_centers, dtype=np.float64),
        group_radii=np.array(group_radii, dtype=np.float64),
        group_bbox_min=np.array(group_bbox_min, dtype=np.float64),
        group_bbox_max=np.array(group_bbox_max, dtype=np.float64),
        group_source_counts=np.array(group_source_counts, dtype=np.int64),
        source_ids=np.array(source_ids, dtype=np.int64),
        source_positions=np.array(source_positions, dtype=np.float64),
        source_group_ids=np.array(source_group_ids, dtype=np.int64),
        local_offsets=local_offsets,
        local_num_cells=local_num_cells,
    )

    return out

def plot_domain_decomposition_from_npz(
    npz_path: str | Path,
    show: bool = True,
):
    """Load a decomposition dump from NPZ and render the 3D plot.

    Parameters
    ----------
    npz_path : str | Path
        Path to an ``.npz`` file produced by ``export_domain_decomposition_npz``.
    show : bool
        If ``True``, display the figure. Otherwise, save to
        ``domain_decomposition_from_npz.png``.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        global_min = np.asarray(data["global_domain_min"], dtype=float)
        global_max = np.asarray(data["global_domain_max"], dtype=float)
        group_ids = np.asarray(data["group_ids"], dtype=np.int64)
        group_bbox_min = np.asarray(data["group_bbox_min"], dtype=float)
        group_bbox_max = np.asarray(data["group_bbox_max"], dtype=float)
        source_positions = np.asarray(data["source_positions"], dtype=float)
        source_group_ids = np.asarray(data["source_group_ids"], dtype=np.int64)
        local_offsets = np.asarray(data["local_offsets"], dtype=np.int64)
        local_num_cells = np.asarray(data["local_num_cells"], dtype=np.int64)
        cell_size = float(data["global_cell_size"])

    fig = plt.figure(figsize=(10, 8))
    ax = cast(Any, fig.add_subplot(111, projection="3d"))

    domain_extent = np.maximum(global_max - global_min, 1.0)
    axis_padding = 0.05 * domain_extent
    label_offset = 0.02 * domain_extent
    _plot_box(
        ax,
        global_min,
        global_max,
        color="black",
        linewidth=2.0,
        alpha=0.9,
        label="Global grid",
    )

    color_map = plt.get_cmap("tab20")
    num_items = max(len(group_ids), len(local_num_cells))

    for index in range(num_items):
        color = color_map(index % color_map.N)

        if index < len(local_num_cells) and local_num_cells[index] > 0:
            local_min = local_offsets[index].astype(float) * cell_size
            local_max = (local_offsets[index].astype(float) + local_num_cells[index]) * cell_size
            _plot_box(
                ax,
                local_min,
                local_max,
                color=color,
                linewidth=1.5,
                alpha=0.8,
                label="Local grid" if index == 0 else None,
            )

        if index < len(group_ids):
            gid = group_ids[index]
            _plot_box(
                ax,
                group_bbox_min[index],
                group_bbox_max[index],
                color=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.5,
                label="Group bounding box" if index == 0 else None,
            )

            group_mask = source_group_ids == gid
            if np.any(group_mask):
                positions = source_positions[group_mask]
                ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    color=[color],
                    s=30,
                    depthshade=False,
                    label="Sources" if index == 0 else None,
                )

            ax.text(
                min(group_bbox_max[index, 0] + label_offset[0], global_max[0] + axis_padding[0]),
                min(group_bbox_max[index, 1] + label_offset[1], global_max[1] + axis_padding[1]),
                min(group_bbox_max[index, 2] + label_offset[2], global_max[2] + axis_padding[2]),
                str(int(gid)),
                color=color,
                fontsize=9,
            )

    ax.set_xlim(global_min[0] - axis_padding[0], global_max[0] + axis_padding[0])
    ax.set_ylim(global_min[1] - axis_padding[1], global_max[1] + axis_padding[1])
    ax.set_zlim(global_min[2] - axis_padding[2], global_max[2] + axis_padding[2])
    ax.set_box_aspect(tuple(domain_extent.tolist()))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"Domain decomposition: {len(group_ids)} groups, {len(local_num_cells)} local grids"
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig("domain_decomposition_from_npz.png", dpi=300, bbox_inches="tight")
    return fig, ax
