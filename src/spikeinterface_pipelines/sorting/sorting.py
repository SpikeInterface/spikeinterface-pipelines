import spikeinterface.sorters as ss
import spikeinterface as si
from pathlib import Path
import shutil

from ..logger import logger
from ..models import JobKwargs
from .models import SortingParamsModel


def sorting(
    recording: si.BaseRecording,
    sorting_params: SortingParamsModel = SortingParamsModel(),
    results_path: Path = Path("./results/sorting/"),
    job_kwargs: JobKwargs = JobKwargs(),
) -> si.BaseSorting | None:
    try:
        sorter = ss.run_sorter(
            recording=recording,
            sorter_name=sorting_params.sorter_name,
            output_folder=str(results_path / "tmp"),
            verbose=False,
            delete_output_folder=True,
            **sorting_params.sorter_kwargs.model_dump(),
        )
        # remove empty units
        sorter = sorter.remove_empty_units()
        # save results
        logger.info(f"\tSaving results to {results_path}")
        sorter = sorter.save(folder=results_path)
        return sorter
    except Exception as e:
        # save log to results
        results_path.mkdir(exist_ok=True, parents=True)
        if (results_path / "tmp").exists():
            shutil.copy(results_path / "tmp/spikeinterface_log.json", results_path)
        raise e
    finally:
        pass
