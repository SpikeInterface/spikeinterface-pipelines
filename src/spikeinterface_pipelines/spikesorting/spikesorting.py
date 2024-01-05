from __future__ import annotations
from pathlib import Path
import shutil
import spikeinterface.full as si
import spikeinterface.curation as sc

from ..logger import logger
from .params import SpikeSortingParams


def spikesort(
    recording: si.BaseRecording,
    spikesorting_params: SpikeSortingParams = SpikeSortingParams(),
    scratch_folder: Path = Path("./scratch/"),
    results_folder: Path = Path("./results/spikesorting/"),
) -> si.BaseSorting | None:
    """
    Apply spike sorting to recording

    Parameters
    ----------
    recording: si.BaseRecording
        The input recording
    sorting_params: SpikeSortingParams
        Spike sorting parameters
    scratch_folder: Path
        Path to the scratch folder
    results_folder: Path
        Path to the results folder

    Returns
    -------
    si.BaseSorting | None
        Spike sorted sorting. If spike sorting fails, None is returned
    """
    output_folder = scratch_folder / "tmp_spikesorting"

    try:
        logger.info(f"[Spikesorting] \tStarting {spikesorting_params.sorter_name} spike sorter")

        ## TEST ONLY - REMOVE LATER ##
        # si.get_default_sorter_params('kilosort2_5')
        # params_kilosort2_5 = {'do_correction': False}
        ## --------------------------##

        sorting = si.run_sorter(
            recording=recording,
            sorter_name=spikesorting_params.sorter_name,
            output_folder=str(output_folder),
            verbose=True,
            delete_output_folder=True,
            remove_existing_folder=True,
            **spikesorting_params.sorter_kwargs.model_dump(),
            # **params_kilosort2_5
        )
        logger.info(f"[Spikesorting] \tFound {len(sorting.unit_ids)} raw units")
        # remove spikes beyond num_Samples (if any)
        sorting = sc.remove_excess_spikes(sorting=sorting, recording=recording)
        # save results
        logger.info(f"[Spikesorting]\tSaving results to {results_folder}")
        return sorting
    except Exception as e:
        # save log to results
        results_folder.mkdir(exist_ok=True, parents=True)
        if (output_folder).is_dir():
            shutil.copy(output_folder / "spikeinterface_log.json", results_folder)
            shutil.rmtree(output_folder)
        logger.info(f"Spike sorting error:\n{e}")
        return None
