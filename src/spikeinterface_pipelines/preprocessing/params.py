from pydantic import BaseModel, Field
from typing import Optional, Union, List, Literal
from enum import Enum


class PreprocessingStrategy(str, Enum):
    cmr = "cmr"
    destripe = "destripe"


class HighpassFilter(BaseModel):
    freq_min: float = Field(default=300.0, description="Minimum frequency for the highpass filter")
    margin_ms: float = Field(default=5.0, description="Margin in milliseconds")


class PhaseShift(BaseModel):
    margin_ms: float = Field(default=100.0, description="Margin in milliseconds for phase shift")


class DetectBadChannels(BaseModel):
    method: str = Field(default="coherence+psd", description="Method to detect bad channels")
    dead_channel_threshold: float = Field(default=-0.5, description="Threshold for dead channel")
    noisy_channel_threshold: float = Field(default=1.0, description="Threshold for noisy channel")
    outside_channel_threshold: float = Field(default=-0.3, description="Threshold for outside channel")
    n_neighbors: int = Field(default=11, description="Number of neighbors")
    seed: int = Field(default=0, description="Seed value")


class CommonReference(BaseModel):
    reference: str = Field(default="global", description="Type of reference")
    operator: str = Field(default="median", description="Operator used for common reference")


class HighpassSpatialFilter(BaseModel):
    n_channel_pad: int = Field(default=60, description="Number of channels to pad")
    n_channel_taper: Optional[int] = Field(default=None, description="Number of channels to taper")
    direction: str = Field(default="y", description="Direction for the spatial filter")
    apply_agc: bool = Field(default=True, description="Whether to apply automatic gain control")
    agc_window_length_s: float = Field(default=0.01, description="Window length in seconds for AGC")
    highpass_butter_order: int = Field(default=3, description="Order for the Butterworth filter")
    highpass_butter_wn: float = Field(default=0.01, description="Natural frequency for the Butterworth filter")


# Motion correction ---------------------------------------------------------------
class MCDetectKwargs(BaseModel):
    method: str = Field(default="locally_exclusive", description="The method for peak detection.")
    peak_sign: Literal["pos", "neg", "both"] = Field(default="neg", description="The peak sign to detect peaks.")
    detect_threshold: float = Field(default=8.0, description="The detection threshold in MAD units.")
    exclude_sweep_ms: float = Field(default=0.1, description="The time sweep to exclude for time de-duplication.")
    radius_um: float = Field(default=50.0, description="The radius in um for channel de-duplication.")


class MCLocalizeCenterOfMass(BaseModel):
    radius_um: float = Field(default=75.0, description="Radius in um for channel sparsity.")
    feature: str = Field(default="ptp", description="'ptp', 'mean', 'energy' or 'peak_voltage'. Feature to consider for computation")


class MCLocalizeMonopolarTriangulation(BaseModel):
    radius_um: float = Field(default=75.0, description="Radius in um for channel sparsity.")
    max_distance_um: float = Field(default=150.0, description="Boundary for distance estimation.")
    optimizer: str = Field(default="minimize_with_log_penality", description="")
    enforce_decrease: bool = Field(default=True, description="Enforce spatial decreasingness for PTP vectors")
    feature: str = Field(default="ptp", description="'ptp', 'energy' or 'peak_voltage'. The available features to consider for estimating the position via monopolar triangulation are peak-to-peak amplitudes (ptp, default), energy ('energy', as L2 norm) or voltages at the center of the waveform (peak_voltage)")


class MCLocalizeGridConvolution(BaseModel):
    radius_um: float = Field(default=40.0, description="Radius in um for channel sparsity.")
    upsampling_um: float = Field(default=5.0, description="Upsampling resolution for the grid of templates.")
    sigma_um: List[float] = Field(default=[5.0, 25.0, 5], description="Spatial decays of the fake templates.")
    sigma_ms: float = Field(default=0.25, description="The temporal decay of the fake templates.")
    margin_um: float = Field(default=30.0, description="The margin for the grid of fake templates.")
    percentile: float = Field(default=10.0, description="The percentage in [0, 100] of the best scalar products kept to estimate the position.")
    sparsity_threshold: float = Field(default=0.01, description="The sparsity threshold (in [0, 1]) below which weights should be considered as 0.")


class MCEstimateMotionDecentralized(BaseModel):
    method: str = Field(default="decentralized", description="")
    direction: str = Field(default="y", description="")
    bin_duration_s: float = Field(default=2.0, description="")
    rigid: bool = Field(default=False, description="")
    bin_um: float = Field(default=5.0, description="")
    margin_um: float = Field(default=0.0, description="")
    win_shape: str = Field(default="gaussian", description="")
    win_step_um: float = Field(default=100.0, description="")
    win_sigma_um: float = Field(default=200.0, description="")
    histogram_depth_smooth_um: float = Field(default=5.0, description="")
    histogram_time_smooth_s: Optional[float] = Field(default=None, description="")
    pairwise_displacement_method: str = Field(default="conv", description="")
    max_displacement_um: float = Field(default=100.0, description="")
    weight_scale: str = Field(default="linear", description="")
    error_sigma: float = Field(default=0.2, description="")
    conv_engine: Optional[str] = Field(default=None, description="")
    torch_device: Optional[str] = Field(default=None, description="")
    batch_size: int = Field(default=1, description="")
    corr_threshold: float = Field(default=0.0, description="")
    time_horizon_s: Optional[float] = Field(default=None, description="")
    convergence_method: str = Field(default="lsmr", description="")
    soft_weights: bool = Field(default=False, description="")
    normalized_xcorr: bool = Field(default=True, description="")
    centered_xcorr: bool = Field(default=True, description="")
    temporal_prior: bool = Field(default=True, description="")
    spatial_prior: bool = Field(default=False, description="")
    force_spatial_median_continuity: bool = Field(default=False, description="")
    reference_displacement: str = Field(default="median", description="")
    reference_displacement_time_s: float = Field(default=0, description="")
    robust_regression_sigma: int = Field(default=2, description="")
    weight_with_amplitude: bool = Field(default=False, description="")


class MCEstimateMotionIterativeTemplate(BaseModel):
    bin_duration_s: float = Field(default=2.0, description="")
    rigid: bool = Field(default=False, description="")
    win_step_um: float = Field(default=50.0, description="")
    win_sigma_um: float = Field(default=150.0, description="")
    margin_um: float = Field(default=0.0, description="")
    win_shape: str = Field(default="rect", description="")


class MCInterpolateMotionKwargs(BaseModel):
    direction: int = Field(default=1, description="0 | 1 | 2. Dimension along which channel_locations are shifted (0 - x, 1 - y, 2 - z).")
    border_mode: str = Field(default="remove_channels", description="'remove_channels' | 'force_extrapolate' | 'force_zeros'. Control how channels are handled on border.")
    spatial_interpolation_method: str = Field(default="idw", description="The spatial interpolation method used to interpolate the channel locations.")
    sigma_um: float = Field(default=20.0, description="Used in the 'kriging' formula")
    p: int = Field(default=1, description="Used in the 'kriging' formula")
    num_closest: int = Field(default=3, description="Number of closest channels used by 'idw' method for interpolation.")


class MCNonrigidAccurate(BaseModel):
    detect_kwargs: MCDetectKwargs = Field(default=MCDetectKwargs(), description="")
    localize_peaks_kwargs: MCLocalizeMonopolarTriangulation = Field(default=MCLocalizeMonopolarTriangulation(), description="")
    estimate_motion_kwargs: MCEstimateMotionDecentralized = Field(default=MCEstimateMotionDecentralized(), description="")
    interpolate_motion_kwargs: MCInterpolateMotionKwargs = Field(default=MCInterpolateMotionKwargs(), description="")


class MCRigidFast(BaseModel):
    detect_kwargs: MCDetectKwargs = Field(default=MCDetectKwargs(), description="")
    localize_peaks_kwargs: MCLocalizeCenterOfMass = Field(default=MCLocalizeCenterOfMass(), description="")
    estimate_motion_kwargs: MCEstimateMotionDecentralized = Field(default=MCEstimateMotionDecentralized(bin_duration_s=10.0, rigid=True), description="")
    interpolate_motion_kwargs: MCInterpolateMotionKwargs = Field(default=MCInterpolateMotionKwargs(), description="")


class MCKilosortLike(BaseModel):
    detect_kwargs: MCDetectKwargs = Field(default=MCDetectKwargs(), description="")
    localize_peaks_kwargs: MCLocalizeGridConvolution = Field(default=MCLocalizeGridConvolution(), description="")
    estimate_motion_kwargs: MCEstimateMotionIterativeTemplate = Field(default=MCEstimateMotionIterativeTemplate(), description="")
    interpolate_motion_kwargs: MCInterpolateMotionKwargs = Field(default=MCInterpolateMotionKwargs(border_mode="force_extrapolate", spatial_interpolation_method="kriging"), description="")


class MCPreset(str, Enum):
    nonrigid_accurate = "nonrigid_accurate"
    rigid_fast = "rigid_fast"
    kilosort_like = "kilosort_like"


class MotionCorrection(BaseModel):
    strategy: Literal["skip", "compute", "apply"] = Field(default="compute", description="What strategy to use for motion correction")
    preset: MCPreset = Field(default=MCPreset.nonrigid_accurate.value, description="Preset for motion correction")
    motion_kwargs: Union[MCNonrigidAccurate, MCRigidFast, MCKilosortLike] = Field(default=MCNonrigidAccurate(), description="Motion correction parameters")


# Preprocessing params ---------------------------------------------------------------
class PreprocessingParams(BaseModel):
    preprocessing_strategy: PreprocessingStrategy = Field(default="cmr", description="Strategy for preprocessing")
    highpass_filter: HighpassFilter = Field(default=HighpassFilter(), description="Highpass filter")
    phase_shift: PhaseShift = Field(default=PhaseShift(), description="Phase shift")
    common_reference: CommonReference = Field(default=CommonReference(), description="Common reference")
    highpass_spatial_filter: HighpassSpatialFilter = Field(
        default=HighpassSpatialFilter(), description="Highpass spatial filter"
    )
    motion_correction: MotionCorrection = Field(default=MotionCorrection(), description="Motion correction")
    detect_bad_channels: DetectBadChannels = Field(default=DetectBadChannels(), description="Detect bad channels")
    remove_out_channels: bool = Field(default=True, description="Flag to remove out channels")
    remove_bad_channels: bool = Field(default=True, description="Flag to remove bad channels")
    max_bad_channel_fraction_to_remove: float = Field(
        default=0.5, description="Maximum fraction of bad channels to remove"
    )
