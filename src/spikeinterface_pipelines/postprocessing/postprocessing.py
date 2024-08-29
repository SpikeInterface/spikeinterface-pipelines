import warnings
from pathlib import Path
import shutil

import spikeinterface as si
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc

from .params import PostprocessingParams
from ..logger import logger


warnings.filterwarnings("ignore")


def postprocess(
    recording: si.BaseRecording,
    sorting: si.BaseSorting,
    postprocessing_params: PostprocessingParams = PostprocessingParams(),
    scratch_folder: Path = Path("./scratch/"),
    results_folder: Path = Path("./results/postprocessing/"),
) -> si.SortingAnalyzer:
    """
    Postprocess preprocessed and spike sorting output

    Parameters
    ----------
    recording: si.BaseRecording
        The input recording
    sorting: si.BaseSorting
        The input sorting
    postprocessing_params: PostprocessingParams
        Postprocessing parameters
    results_folder: Path
        Path to the results folder

    Returns
    -------
    si.SortingAnalyzer
        The sorting analyzer
    """

    tmp_folder = scratch_folder / "tmp_postprocessing"
    tmp_folder.mkdir(parents=True, exist_ok=True)

    # first extract some raw waveforms in memory to deduplicate based on peak alignment
    logger.info(f"[Postprocessing] \tCreating sorting analyzer")
    sorting_analyzer_full = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        sparse=True,
        return_scaled=postprocessing_params.return_scaled,
        **postprocessing_params.sparsity.model_dump()
    )
    # compute templates for de-duplication
    # now postprocess
    analyzer_dict = postprocessing_params.extension_params.model_dump().copy()
    sorting_analyzer_full.compute("random_spikes", **analyzer_dict["random_spikes"])
    sorting_analyzer_full.compute("templates")
    # de-duplication
    sorting_deduplicated = sc.remove_redundant_units(
        sorting_analyzer_full, duplicate_threshold=postprocessing_params.duplicate_threshold
    )
    logger.info(
        f"[Postprocessing] \tNumber of original units: {len(sorting.unit_ids)} -- Number of units after de-duplication: {len(sorting_deduplicated.unit_ids)}"
    )
    n_duplicated = int(len(sorting.unit_ids) - len(sorting_deduplicated.unit_ids))
    deduplicated_unit_ids = sorting_deduplicated.unit_ids

    sorting_analyzer_dedup = sorting_analyzer_full.select_units(sorting_deduplicated.unit_ids)

    # now extract waveforms on de-duplicated units
    sorting_analyzer = si.create_sorting_analyzer(
        sorting=sorting_deduplicated,
        recording=recording,
        sparse=True,
        return_scaled=postprocessing_params.return_scaled,
        sparsity=sorting_analyzer_dedup.sparsity
    )

    # save
    logger.info("[Postprocessing] \tSaving sparse de-duplicated sorting analyzer zarr folder")
    sorting_analyzer = sorting_analyzer.save_as(
        format="zarr",
        folder=results_folder
    )

    logger.info("[Postprocessing] \tComputing all postprocessing extensions")
    sorting_analyzer.compute(analyzer_dict)
    logger.info("[Postprocessing] \tComputing quality metrics")
    quality_metrics_params = postprocessing_params.quality_metrics
    _ = sorting_analyzer.compute(
        "quality_metrics",
        metric_names=quality_metrics_params.metric_names,
        qm_params=quality_metrics_params.qm_params.model_dump(),
    )

    return sorting_analyzer

