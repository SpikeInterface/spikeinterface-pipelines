from pathlib import Path
from typing import Tuple

import spikeinterface as si

from .logger import logger
from .global_params import JobKwargs
from .preprocessing import preprocess, PreprocessingParams
from .spikesorting import spikesort, SpikeSortingParams
from .postprocessing import postprocess, PostprocessingParams


def run_pipeline(
    recording: si.BaseRecording,
    scratch_folder: Path | str = Path("./scratch/"),
    results_folder: Path | str = Path("./results/"),
    job_kwargs: JobKwargs | dict = JobKwargs(),
    preprocessing_params: PreprocessingParams | dict = PreprocessingParams(),
    spikesorting_params: SpikeSortingParams | dict = SpikeSortingParams(),
    postprocessing_params: PostprocessingParams | dict = PostprocessingParams(),
    run_preprocessing: bool = True,
) -> Tuple[si.BaseRecording, si.BaseSorting, si.WaveformExtractor]:
    # Create folders
    results_folder = Path(results_folder)
    scratch_folder = Path(scratch_folder)
    scratch_folder.mkdir(exist_ok=True, parents=True)
    results_folder.mkdir(exist_ok=True, parents=True)

    # Paths
    results_folder_preprocessing = results_folder / "preprocessing"
    results_folder_spikesorting = results_folder / "spikesorting"
    results_folder_postprocessing = results_folder / "postprocessing"

    # Arguments Models validation, in case of dict
    if isinstance(job_kwargs, dict):
        job_kwargs = JobKwargs(**job_kwargs)
    if isinstance(preprocessing_params, dict):
        preprocessing_params = PreprocessingParams(**preprocessing_params)
    if isinstance(spikesorting_params, dict):
        spikesorting_params = SpikeSortingParams(**spikesorting_params)
    if isinstance(postprocessing_params, dict):
        postprocessing_params = PostprocessingParams(**postprocessing_params)

    # set global job kwargs
    si.set_global_job_kwargs(**job_kwargs.model_dump())

    # Preprocessing
    if run_preprocessing:
        logger.info("Preprocessing recording")
        recording_preprocessed = preprocess(
            recording=recording,
            preprocessing_params=preprocessing_params,
            scratch_folder=scratch_folder,
            results_folder=results_folder_preprocessing,
        )
        if recording_preprocessed is None:
            raise Exception("Preprocessing failed")
    else:
        logger.info("Skipping preprocessing")
        recording_preprocessed = recording

    # Spike Sorting
    sorting = spikesort(
        recording=recording_preprocessed,
        scratch_folder=scratch_folder,
        spikesorting_params=spikesorting_params,
        results_folder=results_folder_spikesorting,
    )
    if sorting is None:
        raise Exception("Spike sorting failed")

    # Postprocessing
    waveform_extractor = postprocess(
        recording=recording_preprocessed,
        sorting=sorting,
        postprocessing_params=postprocessing_params,
        scratch_folder=scratch_folder,
        results_folder=results_folder_postprocessing,
    )

    # TODO: Curation

    # TODO: Visualization

    return (recording_preprocessed, sorting, waveform_extractor)
