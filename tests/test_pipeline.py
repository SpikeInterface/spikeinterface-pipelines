import shutil
import pytest
import numpy as np
from pathlib import Path

import spikeinterface as si
import spikeinterface.sorters as ss

from spikeinterface_pipelines import pipeline

from spikeinterface_pipelines.preprocessing import preprocess
from spikeinterface_pipelines.spikesorting import spikesort
from spikeinterface_pipelines.postprocessing import postprocess

from spikeinterface_pipelines.preprocessing.params import PreprocessingParams
from spikeinterface_pipelines.spikesorting.params import Kilosort25Model, SpikeSortingParams
from spikeinterface_pipelines.postprocessing.params import PostprocessingParams


def _generate_gt_recording():
    recording, sorting = si.generate_ground_truth_recording(durations=[30], num_channels=64, seed=0)
    # add inter sample shift (but fake)
    inter_sample_shifts = np.zeros(recording.get_num_channels())
    recording.set_property("inter_sample_shift", inter_sample_shifts)

    return recording, sorting


@pytest.fixture
def generate_recording():
    return _generate_gt_recording()


def test_preprocessing(tmp_path, generate_recording):
    recording, _ = generate_recording

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
    recording, _ = generate_recording
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
    recording, sorting = generate_recording
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


@pytest.mark.skipif(not "kilosort2_5" in ss.installed_sorters(), reason="kilosort2_5 not installed")
def test_pipeline(tmp_path, generate_recording):
    recording, _ = generate_recording
    if "inter_sample_shift" in recording.get_property_keys():
        recording.delete_property("inter_sample_shift")

    results_folder = Path(tmp_path) / "results"
    scratch_folder = Path(tmp_path) / "scratch"

    ks25_params = Kilosort25Model(do_correction=False)
    spikesorting_params = SpikeSortingParams(
        sorter_name="kilosort2_5",
        sorter_kwargs=ks25_params,
    )

    recording_processed, sorting, waveform_extractor = pipeline.run_pipeline(
        recording=recording,
        results_folder=results_folder,
        scratch_folder=scratch_folder,
        spikesorting_params=spikesorting_params,
    )

    assert isinstance(recording_processed, si.BaseRecording)
    assert isinstance(sorting, si.BaseSorting)
    assert isinstance(waveform_extractor, si.WaveformExtractor)


if __name__ == "__main__":
    tmp_folder = Path("./tmp_pipeline_output")
    if tmp_folder.is_dir():
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir()

    recording, sorting = _generate_gt_recording()

    print("TEST PREPROCESSING")
    test_preprocessing(tmp_folder, (recording, sorting))
    print("TEST SPIKESORTING")
    test_spikesorting(tmp_folder, (recording, sorting))
    print("TEST POSTPROCESSING")
    test_postprocessing(tmp_folder, (recording, sorting))

    print("TEST PIPELINE")
    test_pipeline(tmp_folder, (recording, sorting))
