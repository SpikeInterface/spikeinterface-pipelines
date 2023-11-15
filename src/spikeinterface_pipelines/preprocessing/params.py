from pydantic import BaseModel, Field
from typing import Optional
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


class MotionCorrection(BaseModel):
    compute: bool = Field(default=True, description="Whether to compute motion correction")
    apply: bool = Field(default=False, description="Whether to apply motion correction")
    preset: str = Field(default="nonrigid_accurate", description="Preset for motion correction")


class PreprocessingParams(BaseModel):
    preprocessing_strategy: PreprocessingStrategy = Field(default="cmr", description="Strategy for preprocessing")
    highpass_filter: HighpassFilter = Field(default=HighpassFilter(), description="Highpass filter")
    phase_shift: PhaseShift = Field(default=PhaseShift(), description="Phase shift")
    common_reference: CommonReference = Field(default=CommonReference(), description="Common reference")
    highpass_spatial_filter: HighpassSpatialFilter = Field(default=HighpassSpatialFilter(), description="Highpass spatial filter")
    motion_correction: MotionCorrection = Field(default=MotionCorrection(), description="Motion correction")
    detect_bad_channels: DetectBadChannels = Field(default=DetectBadChannels(), description="Detect bad channels")
    remove_out_channels: bool = Field(default=True, description="Flag to remove out channels")
    remove_bad_channels: bool = Field(default=True, description="Flag to remove bad channels")
    max_bad_channel_fraction_to_remove: float = Field(default=0.5, description="Maximum fraction of bad channels to remove")
