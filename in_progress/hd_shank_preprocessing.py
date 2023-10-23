from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
from pyrsistent import v
import spikeinterface as si
import spikeinterface.preprocessing as spre


########################################################################################################################
# Preprocessing parameters

class PhaseShiftParameters(BaseModel):
    margin_ms: float = Field(default=100.0, description="Margin in ms to use for phase shift")

class HighpassFilterParameters(BaseModel):
    freq_min: float = Field(default=300.0, description="Minimum frequency in Hz")
    margin_ms: float = Field(default=5.0, description="Margin in ms to use for highpass filter")

class DetectBadChannelsParameters(BaseModel):
    method: str = Field(default="coherence+psd", description="Method to use for detecting bad channels: 'coherence+psd' or ...")
    dead_channel_threshold: float = Field(default=-0.5, description="Threshold for dead channels")
    noisy_channel_threshold: float = Field(default=1.0, description="Threshold for noisy channels")
    outside_channel_threshold: float = Field(default=-0.3, description="Threshold for outside channels")
    n_neighbors: int = Field(default=11, description="Number of neighbors to use for bad channel detection")
    seed: int = Field(default=0, description="Seed for random number generator")

class CommonReferenceParameters(BaseModel):
    reference: str = Field(default="global", description="Reference to use for common reference: 'global' or ...")
    operator: str = Field(default="median", description="Operator to use for common reference: 'median' or ...")

class HighpassSpatialFilterParameters(BaseModel):
    n_channel_pad: int = Field(default=60, description="Number of channels to pad")
    n_channel_taper: int = Field(default=None, description="Number of channels to taper")
    direction: str = Field(default="y", description="Direction to use for highpass spatial filter: 'y' or ...")
    apply_agc: bool = Field(default=True, description="Whether to apply automatic gain control")
    agc_window_length_s: float = Field(default=0.01, description="Window length in seconds for automatic gain control")
    highpass_butter_order: int = Field(default=3, description="Butterworth order for highpass filter")
    highpass_butter_wn: float = Field(default=0.01, description="Butterworth wn for highpass filter")

class HDShankPreprocessingParameters(BaseModel):
    preprocessing_strategy: str = Field(default="cmr", description="Preprocessing strategy to use: destripe or cmr")
    highpass_filter: HighpassFilterParameters = Field(default_factory=HighpassFilterParameters, description="Highpass filter parameters")
    phase_shift: PhaseShiftParameters = Field(default_factory=PhaseShiftParameters, description="Phase shift parameters")
    detect_bad_channels: DetectBadChannelsParameters = Field(default_factory=DetectBadChannelsParameters, description="Detect bad channels parameters")
    remove_out_channels: bool = Field(default=True, description="Whether to remove out channels")
    remove_bad_channels: bool = Field(default=True, description="Whether to remove bad channels")
    max_bad_channel_fraction_to_remove: float = Field(default=0.5, description="Maximum fraction of bad channels to remove")
    common_reference: CommonReferenceParameters = Field(default_factory=CommonReferenceParameters, description="Common reference parameters")
    highpass_spatial_filter: HighpassSpatialFilterParameters = Field(default_factory=HighpassSpatialFilterParameters, description="Highpass spatial filter parameters")

########################################################################################################################

def hd_shank_preprocessing(
    recording: si.BaseRecording,
    params: HDShankPreprocessingParameters,
    preprocessed_output_folder: Path,
    verbose: bool = False
):
    recording_ps_full = spre.phase_shift(
        recording,
        margin_ms=params.phase_shift.margin_ms
    )

    recording_hp_full = spre.highpass_filter(
        recording_ps_full,
        freq_min=params.highpass_filter.freq_min,
        margin_ms=params.highpass_filter.margin_ms
    )

    # IBL bad channel detection
    _, channel_labels = spre.detect_bad_channels(
        recording_hp_full,
        method=params.detect_bad_channels.method,
        dead_channel_threshold=params.detect_bad_channels.dead_channel_threshold,
        noisy_channel_threshold=params.detect_bad_channels.noisy_channel_threshold,
        outside_channel_threshold=params.detect_bad_channels.outside_channel_threshold,
        n_neighbors=params.detect_bad_channels.n_neighbors,
        seed=params.detect_bad_channels.seed
    )

    dead_channel_mask = channel_labels == "dead"
    noise_channel_mask = channel_labels == "noise"
    out_channel_mask = channel_labels == "out"

    if verbose:
        print("\tBad channel detection:")
        print(
            f"\t\t- dead channels - {np.sum(dead_channel_mask)}\n\t\t- noise channels - {np.sum(noise_channel_mask)}\n\t\t- out channels - {np.sum(out_channel_mask)}"
        )
    dead_channel_ids = recording_hp_full.channel_ids[dead_channel_mask]
    noise_channel_ids = recording_hp_full.channel_ids[noise_channel_mask]
    out_channel_ids = recording_hp_full.channel_ids[out_channel_mask]

    all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

    max_bad_channel_fraction_to_remove = params.max_bad_channel_fraction_to_remove

    # skip_processing = False
    if len(all_bad_channel_ids) >= int(
        max_bad_channel_fraction_to_remove * recording.get_num_channels()
    ):
        # always print this message even if verbose is False?
        print(
            f"\tMore than {max_bad_channel_fraction_to_remove * 100}% bad channels ({len(all_bad_channel_ids)}). "
            f"Skipping further processing for this recording."
        )
        # skip_processing = True
        recording_ret = recording_hp_full
    else:
        if params.remove_out_channels:
            if verbose:
                print(f"\tRemoving {len(out_channel_ids)} out channels")
            recording_rm_out = recording_hp_full.remove_channels(out_channel_ids)
        else:
            recording_rm_out = recording_hp_full

        recording_processed_cmr = spre.common_reference(
            recording_rm_out,
            reference=params.common_reference.reference,
            operator=params.common_reference.operator
        )

        bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids))
        recording_interp = spre.interpolate_bad_channels(recording_rm_out, bad_channel_ids)
        recording_hp_spatial = spre.highpass_spatial_filter(
            recording_interp,
            n_channel_pad=params.highpass_spatial_filter.n_channel_pad,
            n_channel_taper=params.highpass_spatial_filter.n_channel_taper,
            direction=params.highpass_spatial_filter.direction,
            apply_agc=params.highpass_spatial_filter.apply_agc,
            agc_window_length_s=params.highpass_spatial_filter.agc_window_length_s,
            highpass_butter_order=params.highpass_spatial_filter.highpass_butter_order,
            highpass_butter_wn=params.highpass_spatial_filter.highpass_butter_wn,
        )

        preproc_strategy = params.preprocessing_strategy
        if preproc_strategy == "cmr":
            recording_processed = recording_processed_cmr
        else:
            recording_processed = recording_hp_spatial

        if params.remove_bad_channels:
            if verbose:
                print(f"\tRemoving {len(bad_channel_ids)} channels after {preproc_strategy} preprocessing")
            recording_processed = recording_processed.remove_channels(bad_channel_ids)
        recording_saved = recording_processed.save(folder=preprocessed_output_folder / recording_name)
        recording_ret = recording_saved
    return recording_ret
