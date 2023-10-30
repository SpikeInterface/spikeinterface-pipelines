from pathlib import Path
import shutil
import spikeinterface as si
import spikeinterface.sorters as ss

from .logger import logger
from .preprocessing import preprocessing, PreprocessingParamsModel
from .sorting import SortingParamsModel


# TODO - WIP
def pipeline(
    recording: si.BaseRecording,
    results_path: Path = Path("./results/"),
    preprocessing_params: PreprocessingParamsModel = PreprocessingParamsModel(),
    sorting_params: SortingParamsModel = SortingParamsModel(),
):
    # Preprocessing
    results_path_preprocessing = results_path / "preprocessing"
    recording_preprocessed = preprocessing(
        recording=recording,
        preprocessing_params=preprocessing_params,
        results_path=results_path_preprocessing,
    )
    if recording_preprocessed is None:
        return None

    # Spike Sorting
    results_path_sorting = results_path / "sorting"
    try:
        sorting = ss.run_sorter(
            recording=recording_preprocessed,
            output_folder=str(results_path_sorting),
            verbose=False,
            delete_output_folder=True,
            **sorting_params.model_dump(),
        )
        # remove empty units
        sorting = sorting.remove_empty_units()
    except Exception as e:
        # save log to results
        results_path_sorting.mkdir()
        shutil.copy(spikesorted_raw_output_folder / "spikeinterface_log.json", sorting_output_folder)
        raise e

    return recording
