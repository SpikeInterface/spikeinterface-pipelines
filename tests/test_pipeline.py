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

from spikeinterface_pipelines.spikesorting.params import Kilosort25Model


ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))


def _generate_gt_recording():
    recording, sorting = si.generate_ground_truth_recording(durations=[30], num_channels=64, seed=0)
    # add inter sample shift (but fake)
    inter_sample_shifts = np.zeros(recording.get_num_channels())
    recording.set_property("inter_sample_shift", inter_sample_shifts)
    waveform_extractor = si.extract_waveforms(recording, sorting, mode="memory")
    _ = spost.compute_spike_amplitudes(waveform_extractor)
    _ = spost.compute_spike_locations(waveform_extractor, method="center_of_mass")
    _ = spost.compute_correlograms(waveform_extractor)
    _ = spost.compute_unit_locations(waveform_extractor)
    _ = spost.compute_template_similarity(waveform_extractor)
    _ = sqm.compute_quality_metrics(waveform_extractor)

    return recording, sorting, waveform_extractor


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


@pytest.mark.skipif(not "kilosort2_5" in ss.installed_sorters(), reason="kilosort2_5 not installed")
def test_spikesorting(tmp_path, generate_recording):
    recording, _, _ = generate_recording
    if "inter_sample_shift" in recording.get_property_keys():
        recording.delete_property("inter_sample_shift")

    results_folder = Path(tmp_path) / "results_spikesorting"
    scratch_folder = Path(tmp_path) / "scratch_spikesorting"

    sorting = spikesort(
        recording=recording,
        spikesorting_params=SpikeSortingParams(),
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )

    assert isinstance(sorting, si.BaseSorting)


def test_postprocessing(tmp_path, generate_recording):
    recording, sorting, _ = generate_recording
    if "inter_sample_shift" in recording.get_property_keys():
        recording.delete_property("inter_sample_shift")

    results_folder = Path(tmp_path) / "results_postprocessing"
    scratch_folder = Path(tmp_path) / "scratch_postprocessing"

    waveform_extractor = postprocess(
        recording=recording,
        sorting=sorting,
        postprocessing_params=PostprocessingParams(),
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )

    assert isinstance(waveform_extractor, si.WaveformExtractor)


def test_curation(tmp_path, generate_recording):
    _, _, waveform_extractor = generate_recording

    results_folder = Path(tmp_path) / "results_curation"
    scratch_folder = Path(tmp_path) / "scratch_curation"

    sorting_curated = curate(
        waveform_extractor=waveform_extractor,
        curation_params=CurationParams(),
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )

    assert isinstance(sorting_curated, si.BaseSorting)

    # Unavailable quality metric, returns None
    sorting_curated = curate(
        waveform_extractor=waveform_extractor,
        curation_params=CurationParams(curation_query="l_ratio < 0.5"),
        results_folder=results_folder,
        scratch_folder=scratch_folder,
    )
    assert sorting_curated is None


def test_visualization(tmp_path, generate_recording):
    recording, sorting, waveform_extractor = generate_recording

    results_folder = Path(tmp_path) / "results_visualization"
    scratch_folder = Path(tmp_path) / "scratch_visualization"

    visualization_output = visualize(
        recording=recording,
        sorting_curated=sorting,
        waveform_extractor=waveform_extractor,
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
        waveform_extractor=None,
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

    ks25_params = Kilosort25Model(do_correction=False)
    spikesorting_params = SpikeSortingParams(
        sorter_name="kilosort2_5",
        sorter_kwargs=ks25_params,
    )

    recording_processed, sorting, waveform_extractor, sorting_curated, vis_output = run_pipeline(
        recording=recording,
        results_folder=results_folder,
        scratch_folder=scratch_folder,
        spikesorting_params=spikesorting_params,
    )

    assert isinstance(recording_processed, si.BaseRecording)
    assert isinstance(sorting, si.BaseSorting)
    assert isinstance(waveform_extractor, si.WaveformExtractor)
    assert isinstance(sorting_curated, si.BaseSorting)
    assert isinstance(vis_output, dict)


if __name__ == "__main__":
    tmp_folder = Path("./tmp_pipeline_output")
    if tmp_folder.is_dir():
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir()

    recording, sorting, waveform_extractor = _generate_gt_recording()

    print("TEST PREPROCESSING")
    test_preprocessing(tmp_folder, (recording, sorting))
    print("TEST SPIKESORTING")
    test_spikesorting(tmp_folder, (recording, sorting))
    print("TEST POSTPROCESSING")
    test_postprocessing(tmp_folder, (recording, sorting))
    print("TEST CURATION")
    test_curation(tmp_folder, (recording, sorting, waveform_extractor))
    print("TEST VISUALIZATION")
    test_visualization(tmp_folder, (recording, sorting, waveform_extractor))

    print("TEST PIPELINE")
    test_pipeline(tmp_folder, (recording, sorting))
