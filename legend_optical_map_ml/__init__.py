"""LEGEND LAr optical-map data for RESUM_FLEX: LH5 reading, event cache, prepared views.

RESUM_FLEX reads the prepared data through ``legend_optical_map_ml.provider:OpticalDataset``
(set as ``data_provider`` in its run configs).
"""

from legend_optical_map_ml.pipeline import ensure_prepared_optical_data, prepare_optical_data
from legend_optical_map_ml.provider import OpticalDataset

__all__ = ["OpticalDataset", "ensure_prepared_optical_data", "prepare_optical_data"]
