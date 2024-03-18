from __future__ import annotations
import shutil
import numpy as np
from pathlib import Path

import spikeinterface.full as si
import spikeinterface.curation as sc

from ..logger import logger
from .params import SpikeSortingParams


def spikesort(
    recording: si.BaseRecording,
    spikesorting_params: SpikeSortingParams,
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

        if spikesorting_params.spikesort_by_group and len(np.unique(recording.get_channel_groups())) > 1:
            logger.info(f"[Spikesorting] \tSorting by channel groups")
            sorting = si.run_sorter_by_property(
                recording=recording,
                sorter_name=spikesorting_params.sorter_name,
                grouping_property="group",
                working_folder=str(output_folder),
                verbose=True,
                delete_output_folder=True,
                remove_existing_folder=True,
                **spikesorting_params.sorter_kwargs.model_dump(),
            )
        else:
            sorting = si.run_sorter(
                recording=recording,
                sorter_name=spikesorting_params.sorter_name,
                output_folder=str(output_folder),
                verbose=True,
                delete_output_folder=True,
                remove_existing_folder=True,
                **spikesorting_params.sorter_kwargs.model_dump(),
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
        if not spikesorting_params.spikesort_by_group:
            if (output_folder).is_dir():
                shutil.copy(output_folder / "spikeinterface_log.json", results_folder)
                shutil.rmtree(output_folder)
        else:
            for group_folder in output_folder.iterdir():
                if group_folder.is_dir():
                    shutil.copy(group_folder / "spikeinterface_log.json", results_folder / group_folder.name)
            shutil.rmtree(output_folder)
        logger.info(f"Spike sorting error:\n{e}")
        return None
