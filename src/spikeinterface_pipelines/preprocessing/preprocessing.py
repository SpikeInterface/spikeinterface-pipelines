import warnings
import numpy as np
from pathlib import Path
import spikeinterface as si
import spikeinterface.preprocessing as spre

from ..models import JobKwargs
from .models import PreprocessingParamsModel


warnings.filterwarnings("ignore")


def preprocessing(
    job_kwargs: JobKwargs,
    recording: si.BaseRecording,
    preprocessing_params: PreprocessingParamsModel,
    results_path: Path = Path("./results/"),
    debug: bool = False,
    duration_s: float = 1.
) -> None | si.BaseRecording:
    """
    Preprocessing pipeline for ephys data.

    Parameters
    ----------
    recording: si.BaseRecording
        Recording extractor.
    preprocessing_params: PreprocessingParamsModel
        Preprocessing parameters.
    results_path: Path
        Path to the results folder.
    debug: bool
        Flag to run in debug mode.
    duration_s: float
        Duration in seconds to use in the debug mode.
    """
    si.set_global_job_kwargs(**job_kwargs.model_dump())

    if debug:
        print(f"DEBUG ENABLED - Only running with {duration_s} seconds")

    recording_name = recording.name
    preprocessing_notes = ""
    preprocessing_output_process_json = results_path / f"{data_process_prefix}_{recording_name}.json"
    preprocessing_output_folder = results_path / f"preprocessed_{recording_name}"
    preprocessing_output_json = results_path / f"preprocessed_{recording_name}.json"

    print(f"Preprocessing recording: {recording_name}")
    print(f"\tDuration: {np.round(recording.get_total_duration(), 2)} s")

    recording_ps_full = spre.phase_shift(
        recording,
        **preprocessing_params.phase_shift.model_dump()
    )
    recording_hp_full = spre.highpass_filter(
        recording_ps_full,
        **preprocessing_params.highpass_filter.model_dump()
    )

    # Detect bad channels
    _, channel_labels = spre.detect_bad_channels(
        recording_hp_full,
        **preprocessing_params.detect_bad_channels.model_dump()
    )
    dead_channel_mask = channel_labels == "dead"
    noise_channel_mask = channel_labels == "noise"
    out_channel_mask = channel_labels == "out"
    print("\tBad channel detection:")
    print(f"\t\t- dead channels - {np.sum(dead_channel_mask)}\n\t\t- noise channels - {np.sum(noise_channel_mask)}\n\t\t- out channels - {np.sum(out_channel_mask)}")
    dead_channel_ids = recording_hp_full.channel_ids[dead_channel_mask]
    noise_channel_ids = recording_hp_full.channel_ids[noise_channel_mask]
    out_channel_ids = recording_hp_full.channel_ids[out_channel_mask]
    all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

    max_bad_channel_fraction_to_remove = preprocessing_params.max_bad_channel_fraction_to_remove
    if len(all_bad_channel_ids) >= int(max_bad_channel_fraction_to_remove * recording.get_num_channels()):
        print(f"\tMore than {max_bad_channel_fraction_to_remove * 100}% bad channels ({len(all_bad_channel_ids)}). ")
        print("Skipping further processing for this recording.")
        preprocessing_notes += f"\n- Found {len(all_bad_channel_ids)} bad channels. Skipping further processing\n"
        return None

    if preprocessing_params.remove_out_channels:
        print(f"\tRemoving {len(out_channel_ids)} out channels")
        recording_rm_out = recording_hp_full.remove_channels(out_channel_ids)
        preprocessing_notes += f"\n- Removed {len(out_channel_ids)} outside of the brain."
    else:
        recording_rm_out = recording_hp_full

    bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids))

    if preprocessing_params.preprocessing_strategy == "cmr":
        recording_processed = spre.common_reference(
            recording_rm_out,
            **preprocessing_params.common_reference.model_dump()
        )
    else:
        recording_interp = spre.interpolate_bad_channels(recording_rm_out, bad_channel_ids)
        recording_processed = spre.highpass_spatial_filter(
            recording_interp,
            **preprocessing_params.highpass_spatial_filter.model_dump()
        )

    if preprocessing_params.remove_bad_channels:
        print(f"\tRemoving {len(bad_channel_ids)} channels after {preprocessing_params.preprocessing_strategy} preprocessing")
        recording_processed = recording_processed.remove_channels(bad_channel_ids)
        preprocessing_notes += f"\n- Removed {len(bad_channel_ids)} bad channels after preprocessing.\n"

    # motion correction
    if preprocessing_params.motion_correction.compute:
        preset = preprocessing_params.motion_correction.preset
        print(f"\tComputing motion correction with preset: {preset}")
        motion_folder = output_path / f"motion_{recording_name}"
        recording_corrected = spre.correct_motion(
            recording_processed, preset=preset,
            folder=motion_folder,
            **job_kwargs.model_dump()
        )
        if preprocessing_params.motion_correction.apply:
            print("\tApplying motion correction")
            recording_processed = recording_corrected

    # recording_saved = recording_processed.save(folder=preprocessing_output_folder)
    # recording_processed.dump_to_json(preprocessing_output_json, relative_to=data_folder)

    return recording_processed
