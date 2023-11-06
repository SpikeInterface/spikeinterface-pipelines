from pathlib import Path
from typing import Tuple
import spikeinterface as si

from .logger import logger
from .preprocessing import preprocessing, PreprocessingParamsModel
from .sorting import sorting, SortingParamsModel
# from .postprocessing import postprocessing, PostprocessingParamsModel


# TODO - WIP
def pipeline(
    recording: si.BaseRecording,
    results_path: Path = Path("./results/"),
    preprocessing_params: PreprocessingParamsModel = PreprocessingParamsModel(),
    sorting_params: SortingParamsModel = SortingParamsModel(),
    # postprocessing_params: PostprocessingParamsModel = PostprocessingParamsModel(),
    run_preprocessing: bool = True,
) -> Tuple[si.BaseRecording, si.BaseSorting]:
    # Paths
    results_path_preprocessing = results_path / "preprocessing"
    results_path_sorting = results_path / "sorting"

    # Preprocessing
    if run_preprocessing:
        logger.info("Preprocessing recording")
        recording_preprocessed = preprocessing(
            recording=recording,
            preprocessing_params=preprocessing_params,
            results_path=results_path_preprocessing,
        )
        if recording_preprocessed is None:
            raise Exception("Preprocessing failed")
    else:
        logger.info("Skipping preprocessing")
        recording_preprocessed = recording

    # Spike Sorting
    sorter = sorting(
        recording=recording_preprocessed,
        sorting_params=sorting_params,
        results_path=results_path_sorting,
    )

    # #  TODO - Postprocessing
    # postprocessing(postprocessing_params=postprocessing_params)

    return (recording_preprocessed, sorter)
