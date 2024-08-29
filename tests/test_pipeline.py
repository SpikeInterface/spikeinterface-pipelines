import os
import shutil
import pytest
import numpy as np
from pathlib import Path

import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

from spikeinterface_pipelines import run_pipeline

from spikeinterface_pipelines.preprocessing import preprocess, PreprocessingParams
from spikeinterface_pipelines.spikesorting import spikesort, SpikeSortingParams
from spikeinterface_pipelines.postprocessing import postprocess, PostprocessingParams
from spikeinterface_pipelines.curation import curate, CurationParams
from spikeinterface_pipelines.visualization import visualize, VisualizationParams

from spikeinterface_pipelines.spikesorting.params import Kilosort25Model, Kilosort4Model


ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))


def _generate_gt_recording():
    recording, sorting = si.generate_ground_truth_recording(durations=[15], num_channels=128, seed=0)
    # add inter sample shift (but fake)
    inter_sample_shifts = np.zeros(recording.get_num_channels())
    recording.set_property("inter_sample_shift", inter_sample_shifts)
    analyzer = si.create_sorting_analyzer(sorting, recording)
    analyzer.compute(
        [
            "random_spikes",
            "templates",
            "noise_levels",
            "spike_amplitudes",
            "spike_locations",
            "correlograms",
            "unit_locations",
            "template_similarity",
            "quality_metrics",
        ]
    )

    return recording, sorting, analyzer


@pytest.fixture
def generate_recording():
    return _generate_gt_recording()


def test_preprocessing(tmp_path, generate_recording):
    recording, _, _ = generate_recording

    results_folder = Path(tmp_path) / "results_preprocessing"
    scratch_folder = Path(tmp_path) / "scratch_prepocessing"

    recording_processed = preprocess(
        recording=recording,
        preprocessing_params=PreprocessingParams(),
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )

    assert isinstance(recording_processed, si.BaseRecording)


@pytest.mark.skipif(not "kilosort4" in ss.installed_sorters(), reason="kilosort4 not installed")
def test_spikesorting(tmp_path, generate_recording):
    recording, _, _ = generate_recording
    if "inter_sample_shift" in recording.get_property_keys():
        recording.delete_property("inter_sample_shift")

    results_folder = Path(tmp_path) / "results_spikesorting"
    scratch_folder = Path(tmp_path) / "scratch_spikesorting"

    sorter_params = Kilosort4Model(do_correction=False)
    spikesorting_params = SpikeSortingParams(
        sorter_name="kilosort4",
        sorter_kwargs=sorter_params,
    )

    sorting = spikesort(
        recording=recording,
        spikesorting_params=spikesorting_params,
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )

    assert isinstance(sorting, si.BaseSorting)

    #  by group
    num_channels = recording.get_num_channels()
    groups = [0] * (num_channels // 2) + [1] * (num_channels // 2)
    recording.set_channel_groups(groups)

    spikesorting_params.spikesort_by_group = True
    sorting_group = spikesort(
        recording=recording,
        spikesorting_params=spikesorting_params,
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )

    assert isinstance(sorting_group, si.BaseSorting)
    assert "group" in sorting_group.get_property_keys()


def test_postprocessing(tmp_path, generate_recording):
    recording, sorting, _ = generate_recording
    if "inter_sample_shift" in recording.get_property_keys():
        recording.delete_property("inter_sample_shift")

    results_folder = Path(tmp_path) / "results_postprocessing"
    scratch_folder = Path(tmp_path) / "scratch_postprocessing"

    sorting_analyzer = postprocess(
        recording=recording,
        sorting=sorting,
        postprocessing_params=PostprocessingParams(),
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )

    assert isinstance(sorting_analyzer, si.SortingAnalyzer)


def test_curation(tmp_path, generate_recording):
    _, _, sorting_analyzer = generate_recording

    results_folder = Path(tmp_path) / "results_curation"
    scratch_folder = Path(tmp_path) / "scratch_curation"

    sorting_curated = curate(
        sorting_analyzer=sorting_analyzer,
        curation_params=CurationParams(),
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )

    assert isinstance(sorting_curated, si.BaseSorting)

    # Unavailable quality metric, returns None
    sorting_curated = curate(
        sorting_analyzer=sorting_analyzer,
        curation_params=CurationParams(curation_query="l_ratio < 0.5"),
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )
    assert sorting_curated is None


def test_visualization(tmp_path, generate_recording):
    recording, sorting, sorting_analyzer = generate_recording

    results_folder = Path(tmp_path) / "results_visualization"
    scratch_folder = Path(tmp_path) / "scratch_visualization"

    visualization_output = visualize(
        recording=recording,
        sorting_curated=sorting,
        sorting_analyzer=sorting_analyzer,
        visualization_params=VisualizationParams(),
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )

    assert isinstance(visualization_output, dict)
    if not ON_GITHUB:
        # on github traces will fail because it requires pyvips
        assert "recording" in visualization_output
    assert "sorting_summary" in visualization_output

    visualization_output = visualize(
        recording=recording,
        sorting_curated=None,
        sorting_analyzer=None,
        visualization_params=VisualizationParams(),
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )
    assert isinstance(visualization_output, dict)
    if not ON_GITHUB:
        # on github traces will fail because it requires pyvips
        assert "recording" in visualization_output
    assert "sorting_summary" not in visualization_output


@pytest.mark.skipif(not "kilosort2_5" in ss.installed_sorters(), reason="kilosort2_5 not installed")
def test_pipeline(tmp_path, generate_recording):
    recording, _, _ = generate_recording
    if "inter_sample_shift" in recording.get_property_keys():
        recording.delete_property("inter_sample_shift")

    results_folder = Path(tmp_path) / "results"
    scratch_folder = Path(tmp_path) / "scratch"

    sorter_params = Kilosort4Model(do_correction=False)
    spikesorting_params = SpikeSortingParams(
        sorter_name="kilosort4",
        sorter_kwargs=sorter_params,
    )

    recording_processed, sorting, analyzer, sorting_curated, vis_output = run_pipeline(
        recording=recording,
        results_folder=results_folder,
        scratch_folder=scratch_folder,
        spikesorting_params=spikesorting_params,
    )

    assert isinstance(recording_processed, si.BaseRecording)
    assert isinstance(sorting, si.BaseSorting)
    assert isinstance(sorting_analyzer, si.SortingAnalyzer)
    assert isinstance(sorting_curated, si.BaseSorting)
    assert isinstance(vis_output, dict)


if __name__ == "__main__":
    tmp_folder = Path("./tmp_pipeline_output")
    if tmp_folder.is_dir():
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir()

    recording, sorting, sorting_analyzer = _generate_gt_recording()

    # print("TEST PREPROCESSING")
    # test_preprocessing(tmp_folder, (recording, sorting, sorting_analyzer))
    # print("TEST SPIKESORTING")
    # test_spikesorting(tmp_folder, (recording, sorting, sorting_analyzer))
    # print("TEST POSTPROCESSING")
    # test_postprocessing(tmp_folder, (recording, sorting, sorting_analyzer))
    # print("TEST CURATION")
    # test_curation(tmp_folder, (recording, sorting, sorting_analyzer))
    print("TEST VISUALIZATION")
    test_visualization(tmp_folder, (recording, sorting, sorting_analyzer))

    print("TEST PIPELINE")
    test_pipeline(tmp_folder, (recording, sorting, sorting_analyzer))
