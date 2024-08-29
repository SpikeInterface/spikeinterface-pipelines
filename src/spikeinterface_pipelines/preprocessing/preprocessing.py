import warnings
import numpy as np
from pathlib import Path

import spikeinterface as si
import spikeinterface.preprocessing as spre

from ..logger import logger
from .params import (
    PreprocessingParams,
    MCDredge,
    MCDredgeFast,
    MCNonrigidAccurate,
    MCNonrigidFastAndAccurate,
    MCRigidFast,
    MCKilosortLike
)


warnings.filterwarnings("ignore")

_motion_correction_presets_to_params = dict(
    dredge=MCDredge,
    dredge_fast=MCDredgeFast,
    nonrigid_accurate=MCNonrigidAccurate,
    nonrigid_fast_and_accurate=MCNonrigidFastAndAccurate,
    rigid_fast=MCKilosortLike,
    kilosort_like=MCKilosortLike,
)


def preprocess(
    recording: si.BaseRecording,
    preprocessing_params: PreprocessingParams = PreprocessingParams(),
    scratch_folder: Path = Path("./scratch/"),
    results_folder: Path = Path("./results/preprocessing/"),
) -> si.BaseRecording:
    """
    Apply preprocessing to recording.

    Parameters
    ----------
    recording: si.BaseRecording
        The input recording
    preprocessing_params: PreprocessingParams
        Preprocessing parameters
    scratch_folder: Path
        Path to the scratch folder
    results_folder: Path
        Path to the results folder

    Returns
    -------
    si.BaseRecording | None
        Preprocessed recording. If more than `max_bad_channel_fraction_to_remove` channels are detected as bad,
        returns None.
    """
    logger.info("[Preprocessing] \tRunning Preprocessing stage")
    logger.info(f"[Preprocessing] \tDuration: {np.round(recording.get_total_duration(), 2)} s")

    # Phase shift correction
    if "inter_sample_shift" in recording.get_property_keys():
        logger.info("[Preprocessing] \tPhase shift")
        recording = spre.phase_shift(recording, **preprocessing_params.phase_shift.model_dump())
    else:
        logger.info("[Preprocessing] \tSkipping phase shift: 'inter_sample_shift' property not found")

    # Highpass filter
    if preprocessing_params.filter_type == "highpass":
        logger.info("[Preprocessing] \tHighpass filter")
        recording_filt_full = spre.highpass_filter(recording, **preprocessing_params.highpass_filter.model_dump())
    else:
        logger.info("[Preprocessing] \tBandpass filter")
        recording_filt_full = spre.bandpass_filter(recording, **preprocessing_params.bandpass_filter.model_dump())

    # Detect and remove bad channels
    _, channel_labels = spre.detect_bad_channels(
        recording_filt_full, **preprocessing_params.detect_bad_channels.model_dump()
    )
    dead_channel_mask = channel_labels == "dead"
    noise_channel_mask = channel_labels == "noise"
    out_channel_mask = channel_labels == "out"
    logger.info(
        f"[Preprocessing] \tBad channel detection found: {np.sum(dead_channel_mask)} dead, {np.sum(noise_channel_mask)} noise, {np.sum(out_channel_mask)} out channels"
    )
    dead_channel_ids = recording_filt_full.channel_ids[dead_channel_mask]
    noise_channel_ids = recording_filt_full.channel_ids[noise_channel_mask]
    out_channel_ids = recording_filt_full.channel_ids[out_channel_mask]
    all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

    max_bad_channel_fraction_to_remove = preprocessing_params.max_bad_channel_fraction_to_remove
    if len(all_bad_channel_ids) >= int(max_bad_channel_fraction_to_remove * recording.get_num_channels()):
        logger.info(
            f"[Preprocessing] \tMore than {max_bad_channel_fraction_to_remove * 100}% bad channels ({len(all_bad_channel_ids)}). "
        )
        if preprocessing_params.remove_bad_channels:
            logger.info("[Preprocessing] \tSkipping further processing for this recording.")
            return recording_filt_full

    if preprocessing_params.remove_out_channels:
        logger.info(f"[Preprocessing] \tRemoving {len(out_channel_ids)} out channels")
        recording_rm_out = recording_filt_full.remove_channels(out_channel_ids)
    else:
        recording_rm_out = recording_filt_full

    bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids))

    # Denoise: CMR or destripe
    if preprocessing_params.preprocessing_strategy == "cmr":
        recording_processed = spre.common_reference(
            recording_rm_out, **preprocessing_params.common_reference.model_dump()
        )
    else:
        # protection against short probes
        try:
            recording_interp = spre.interpolate_bad_channels(recording_rm_out, bad_channel_ids)
            recording_processed = spre.highpass_spatial_filter(
                recording_interp, **preprocessing_params.highpass_spatial_filter.model_dump()
            )
        except:
            recording_processed = recording_rm_out
            logger.warning(
                "[Preprocessing] \tInterpolation failed. Skipping interpolation and highpass spatial filter."
            )

    if preprocessing_params.remove_bad_channels:
        logger.info(
            f"[Preprocessing] \tRemoving {len(bad_channel_ids)} channels after {preprocessing_params.preprocessing_strategy} preprocessing"
        )
        recording_processed = recording_processed.remove_channels(bad_channel_ids)

    # Motion correction
    if preprocessing_params.motion_correction.strategy != "skip":
        preset = preprocessing_params.motion_correction.preset
        motion_correction_kwargs = _motion_correction_presets_to_params[preset](
            **preprocessing_params.motion_correction.motion_kwargs.model_dump()
        )
        logger.info(f"[Preprocessing] \tComputing motion correction with preset: {preset}")
        motion_folder = results_folder / "motion_correction"
        # interpolation requires float
        recording_processed_f = spre.astype(recording_processed, "float32")

        # fix for multi-segment
        concat_motion = False
        if recording_processed.get_num_segments() > 1:
            recording_processed_c = si.concatenate_recordings([recording_processed_f])
            concat_motion = True
        else:
            recording_processed_c = recording_processed_f

        recording_corrected = spre.correct_motion(
            recording_processed_c,
            preset=preset,
            folder=motion_folder,
            detect_kwargs=motion_correction_kwargs.detect_kwargs.model_dump(),
            localize_peaks_kwargs=motion_correction_kwargs.localize_peaks_kwargs.model_dump(),
            estimate_motion_kwargs=motion_correction_kwargs.estimate_motion_kwargs.model_dump(),
            interpolate_motion_kwargs=motion_correction_kwargs.interpolate_motion_kwargs.model_dump(),
        )

        # split segments back
        if concat_motion:
            rec_corrected_list = []
            for segment_index in range(recording_processed.get_num_segments()):
                num_samples = recording_processed.get_num_samples(segment_index)
                if segment_index == 0:
                    start_frame = 0
                else:
                    start_frame = recording_processed.get_num_samples(segment_index - 1)
                end_frame = start_frame + num_samples
                rec_split_corrected = recording_corrected.frame_slice(
                    start_frame=start_frame,
                    end_frame=end_frame
                )
                rec_corrected_list.append(rec_split_corrected)
            # append all segments
            recording_corrected = si.append_recordings(rec_corrected_list)

        if preprocessing_params.motion_correction.strategy == "apply":
            logger.info("[Preprocessing] \tApplying motion correction")
            recording_processed = recording_corrected

    return recording_processed
