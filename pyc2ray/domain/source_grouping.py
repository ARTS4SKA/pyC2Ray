from abc import abstractmethod
from dataclasses import dataclass
# import logging
from typing import List

from pyc2ray.domain.cost_model import CostModel
from pyc2ray.domain.sources import SourceGroup

# ==============================================================================
# This file contains the class SourceGrouping, which provides a common interface
# for all grouping algorithms. 
# ==============================================================================

# logger = logging.getLogger(__name__)

@dataclass
class GroupingParams:
    """Common parameters for all source grouping algorithms.
    """
    def __init__(self, max_num_sources_per_group: int = 10) -> None:
         self.max_num_sources_per_group: int = max_num_sources_per_group

class SourceGrouping:
    """Class providing a common interface for all grouping algorithms.
    """
    @abstractmethod
    def build_groups(self, sources, grid, grouping_params: GroupingParams, cost_model: CostModel) -> List[SourceGroup]:
        raise NotImplementedError("Call to build_groups abstract method.")
