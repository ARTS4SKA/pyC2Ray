from typing import List, Tuple
import numpy as np

# ==========================================================================
# This file contains the implementation of the Source and of the SourceGroup 
# classes, which represent a single source in the simulation and the groups 
# of them that belong together in the domain decomposition, respectively.
# ==========================================================================

class Source:
    """Represents a single radiation source.

    Attributes
    ----------
    id : int
        Global source identifier.
    pos : np.ndarray
        Source position in 3D Cartesian coordinates (shape `(3,)`).
    strength : float
        Source emissivity/intensity used by downstream physics models.
    radius : float
        Maximum tracing/influence radius for this source.
    """
    def __init__(
        self,
        id: int,
        pos: np.ndarray,
        strength: float,
        radius: float,
    ) -> None:
        self.id = id
        self.pos = pos
        self.strength = strength
        self.radius = radius


class SourceGroup:
    """Container for a set of sources grouped for joint processing.

    A source group represents a set of sources that are grouped together for 
    joint processing in the domain decomposition. This group is defined as a 
    list of sources and a region of influence of the group itself which, in 
    the current implementation, is represented by a sphere enclosing the 
    spheres of influence of the single sources. In the future, more complex
    shapes could be used to better fit the actual region of influence of the
    group, such as a convex hull of the spheres of influence of the single
    sources, or even the actual union of the spheres of influence of the single
    sources. 

    Attributes
    ----------
    id : int
        Global group identifier.
    sources : List[Source]
        Sources that belong to this group.
    center : np.ndarray
        Center of the enclosing sphere of grouped source spheres.
    radius : float
        Radius of the enclosing sphere.
    bbox_min : np.ndarray
        Minimum corner of the cubic axis-aligned bounding box around the group.
    bbox_max : np.ndarray
        Maximum corner of the cubic axis-aligned bounding box around the group.
    mem_cost : float
        Estimated memory cost for this group.
    comp_cost : float
        # TODO: to be correcly estimated
        Estimated computational cost for this group.
    """
    def __init__(
        self,
        id: int,
        sources: List[Source],
        center: np.ndarray,
        radius: float,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        mem_cost: float,
        comp_cost: float
    ) -> None:
        self.id = id
        self.sources = sources
        self.center = center.copy()
        self.radius = radius
        self.bbox_min = bbox_min.copy()
        self.bbox_max = bbox_max.copy()
        self.mem_cost = mem_cost
        self.comp_cost = comp_cost

    def get_source_ids(self) -> List[int]:
        """Return the list of source global IDs in this group."""
        return [s.id for s in self.sources]

    def get_num_sources(self) -> int:
        """Return the number of sources in this group."""
        return len(self.sources)