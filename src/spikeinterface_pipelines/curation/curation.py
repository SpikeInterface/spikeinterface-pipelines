
from pathlib import Path
import re

import numpy as np

import spikeinterface as si
import spikeinterface.curation as sc

from ..logger import logger
from .params import CurationParams


def curate(
    sorting_analyzer: si.SortingAnalyzer,
    curation_params: CurationParams = CurationParams(),
    scratch_folder: Path = Path("./scratch/"),
    results_folder: Path = Path("./results/curation/"),
) -> si.BaseSorting | None:
    """
    Apply automatic curation to spike sorting output.
    The returned Sorting object has a property 'default_qc' that can be used to filter out units.

    Parameters
    ----------
    sorting_analyzer: si.SortingAnalyzer
        The input sorting analyzer
    curation_params: CurationParams
        Curation parameters
    scratch_folder: Path
        Path to the scratch folder
    results_folder: Path
        Path to the results folder

    Returns
    -------
    si.BaseSorting | None
        Curated sorting
    """
    # get quality metrics
    if not sorting_analyzer.has_extension("quality_metrics"):
        logger.info(f"[Curation] \tQuality metrics not found in SortingAnalyzer.")
        return

    qm = sorting_analyzer.get_extension("quality_metrics").get_data()
    # check query validity against quality metrics
    quality_metrics_in_query = re.split(">|<|>=|<=|==|and", curation_params.curation_query)[::2]
    quality_metrics_in_query = [qm_name.strip() for qm_name in quality_metrics_in_query]

    if not all([qm_name in qm.columns for qm_name in quality_metrics_in_query]):
        logger.info(
            f"[Curation] \tQuality metrics in curation query ({quality_metrics_in_query}) not found in quality metrics."
        )
        return
    logger.info(f"[Curation] \tApplying curation query: {curation_params.curation_query} to quality metrics.")
    qm_curated = qm.query(curation_params.curation_query)
    curated_unit_ids = qm_curated.index.values

    # flag units as good/bad depending on QC selection
    qc_quality = [True if unit in curated_unit_ids else False for unit in sorting_analyzer.unit_ids]
    sorting_curated = sorting_analyzer.sorting
    sorting_curated.set_property("default_qc", qc_quality)
    n_units = int(len(sorting_curated.unit_ids))
    n_passing = int(np.sum(qc_quality))
    logger.info(f"[Curation] \t{n_passing}/{n_units} units passing default QC.\n")

    return sorting_curated
