from __future__ import annotations
from pathlib import Path
import re
from typing import Tuple
import spikeinterface as si

from .logger import logger
from .global_params import JobKwargs
from .preprocessing import preprocess, PreprocessingParams
from .spikesorting import spikesort, SpikeSortingParams
from .postprocessing import postprocess, PostprocessingParams
from .curation import curate, CurationParams
from .visualization import visualize, VisualizationParams


def run_pipeline(
    recording: si.BaseRecording,
    scratch_folder: Path | str = Path("./scratch/"),
    results_folder: Path | str = Path("./results/"),
    job_kwargs: JobKwargs | dict = JobKwargs(),
    preprocessing_params: PreprocessingParams | dict = PreprocessingParams(),
    spikesorting_params: SpikeSortingParams | dict = SpikeSortingParams(),
    postprocessing_params: PostprocessingParams | dict = PostprocessingParams(),
    curation_params: CurationParams | dict = CurationParams(),
    visualization_params: VisualizationParams | dict = VisualizationParams(),
    run_preprocessing: bool = True,
    run_spikesorting: bool = True,
    run_postprocessing: bool = True,
    run_curation: bool = True,
    run_visualization: bool = True,
) -> Tuple[
    si.BaseRecording | None,
    si.BaseSorting | None,
    si.WaveformExtractor | None,
    si.BaseSorting | None,
    dict | None,
]:
    # Create folders
    results_folder = Path(results_folder)
    scratch_folder = Path(scratch_folder)
    scratch_folder.mkdir(exist_ok=True, parents=True)
    results_folder.mkdir(exist_ok=True, parents=True)

    # Paths
    results_folder_preprocessing = results_folder / "preprocessing"
    results_folder_spikesorting = results_folder / "spikesorting"
    results_folder_postprocessing = results_folder / "postprocessing"
    results_folder_curation = results_folder / "curation"
    results_folder_visualization = results_folder / "visualization"

    # Arguments Models validation, in case of dict
    if isinstance(job_kwargs, dict):
        job_kwargs = JobKwargs(**job_kwargs)
    if isinstance(preprocessing_params, dict):
        preprocessing_params = PreprocessingParams(**preprocessing_params)
    if isinstance(spikesorting_params, dict):
        spikesorting_params = SpikeSortingParams(**spikesorting_params)
    if isinstance(postprocessing_params, dict):
        postprocessing_params = PostprocessingParams(**postprocessing_params)
    if isinstance(curation_params, dict):
        curation_params = CurationParams(**curation_params)
    if isinstance(visualization_params, dict):
        visualization_params = VisualizationParams(**visualization_params)

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
    if run_spikesorting:
        sorting = spikesort(
            recording=recording_preprocessed,
            scratch_folder=scratch_folder,
            spikesorting_params=spikesorting_params,
            results_folder=results_folder_spikesorting,
        )
        if sorting is None:
            raise Exception("Spike sorting failed")

        # Postprocessing
        sorting_curated = sorting
        if run_postprocessing:
            logger.info("Postprocessing sorting")
            waveform_extractor = postprocess(
                recording=recording_preprocessed,
                sorting=sorting,
                postprocessing_params=postprocessing_params,
                scratch_folder=scratch_folder,
                results_folder=results_folder_postprocessing,
            )

            # Curation
            if run_curation:
                logger.info("Curating sorting")
                sorting_curated = curate(
                    waveform_extractor=waveform_extractor,
                    curation_params=curation_params,
                    scratch_folder=scratch_folder,
                    results_folder=results_folder_curation,
                )
            else:
                logger.info("Skipping curation")
        else:
            logger.info("Skipping postprocessing")
            waveform_extractor = None
            
    else:
        logger.info("Skipping spike sorting")
        sorting = None
        waveform_extractor = None
        sorting_curated = None
        

    # Visualization
    visualization_output = None
    if run_visualization:
        logger.info("Visualizing results")
        visualization_output = visualize(
            recording=recording_preprocessed,
            sorting_curated=sorting_curated,
            waveform_extractor=waveform_extractor,
            visualization_params=visualization_params,
            scratch_folder=scratch_folder,
            results_folder=results_folder_visualization,
        )

    return (recording_preprocessed, sorting, waveform_extractor, sorting_curated, visualization_output)
