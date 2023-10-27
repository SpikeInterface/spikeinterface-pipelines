from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
from enum import Enum


class PresenceRatio(BaseModel):
    bin_duration_s: float = Field(60, description="Duration of the bin in seconds.")


class SNR(BaseModel):
    peak_sign: str = Field("neg", description="Sign of the peak.")
    peak_mode: str = Field("extremum", description="Mode of the peak.")
    random_chunk_kwargs_dict: Optional[dict] = Field(None, description="Random chunk arguments.")


class ISIViolation(BaseModel):
    isi_threshold_ms: float = Field(1.5, description="ISI threshold in milliseconds.")
    min_isi_ms: float = Field(0., description="Minimum ISI in milliseconds.")


class RPViolation(BaseModel):
    refractory_period_ms: float = Field(1., description="Refractory period in milliseconds.")
    censored_period_ms: float = Field(0.0, description="Censored period in milliseconds.")


class SlidingRPViolation(BaseModel):
    bin_size_ms: float = Field(0.25, description="The size of binning for the autocorrelogram in ms, by default 0.25.")
    window_size_s: float = Field(1, description="Window in seconds to compute correlogram, by default 1.")
    exclude_ref_period_below_ms: float = Field(0.5, description="Refractory periods below this value are excluded, by default 0.5")
    max_ref_period_ms: float = Field(10, description="Maximum refractory period to test in ms, by default 10 ms.")
    contamination_values: Optional[list] = Field(None, description="The contamination values to test, by default np.arange(0.5, 35, 0.5) %")


class PeakSign(str, Enum):
    neg = "neg"
    pos = "pos"
    both = "both"


class AmplitudeCutoff(BaseModel):
    peak_sign: PeakSign = Field("neg", description="The sign of the peaks.")
    num_histogram_bins: int = Field(100, description="The number of bins to use to compute the amplitude histogram.")
    histogram_smoothing_value: int = Field(3, description="Controls the smoothing applied to the amplitude histogram.")
    amplitudes_bins_min_ratio: int = Field(5, description="The minimum ratio between number of amplitudes for a unit and the number of bins. If the ratio is less than this threshold, the amplitude_cutoff for the unit is set to NaN.")


class AmplitudeMedian(BaseModel):
    peak_sign: PeakSign = Field("neg", description="The sign of the peaks.")


class NearestNeighbor(BaseModel):
    max_spikes: int = Field(10000, description="The number of spikes to use, per cluster. Note that the calculation can be very slow when this number is >20000.")
    min_spikes: int = Field(10, description="Minimum number of spikes.")
    n_neighbors: int = Field(4, description="The number of neighbors to use.")


class NNIsolation(NearestNeighbor):
    n_components: int = Field(10, description="The number of PC components to use to project the snippets to.")
    radius_um: int = Field(100, description="The radius, in um, that channels need to be within the peak channel to be included.")


class QMParams(BaseModel):
    presence_ratio: PresenceRatio
    snr: SNR
    isi_violation: ISIViolation
    rp_violation: RPViolation
    sliding_rp_violation: SlidingRPViolation
    amplitude_cutoff: AmplitudeCutoff
    amplitude_median: AmplitudeMedian
    nearest_neighbor: NearestNeighbor
    nn_isolation: NNIsolation
    nn_noise_overlap: NNIsolation


class QualityMetrics(BaseModel):
    qm_params: QMParams = Field(..., description="Quality metric parameters.")
    metric_names: List[str] = Field(..., description="List of metric names to compute.")
    n_jobs: int = Field(1, description="Number of jobs.")


class Sparsity(BaseModel):
    method: str = Field("radius", description="Method for determining sparsity.")
    radius_um: int = Field(100, description="Radius in micrometers for sparsity.")


class Waveforms(BaseModel):
    ms_before: float = Field(3.0, description="Milliseconds before")
    ms_after: float = Field(4.0, description="Milliseconds after")
    max_spikes_per_unit: int = Field(500, description="Maximum spikes per unit")
    return_scaled: bool = Field(True, description="Flag to determine if results should be scaled")
    dtype: Optional[str] = Field(None, description="Data type for the waveforms")
    precompute_template: Tuple[str, str] = Field(("average", "std"), description="Precomputation template method")
    use_relative_path: bool = Field(True, description="Use relative paths")


class SpikeAmplitudes(BaseModel):
    peak_sign: str = Field("neg", description="Sign of the peak")
    return_scaled: bool = Field(True, description="Flag to determine if amplitudes should be scaled")
    outputs: str = Field("concatenated", description="Output format for the spike amplitudes")


class Similarity(BaseModel):
    method: str = Field("cosine_similarity", description="Method to compute similarity")


class Correlograms(BaseModel):
    window_ms: float = Field(100.0, description="Size of the window in milliseconds")
    bin_ms: float = Field(2.0, description="Size of the bin in milliseconds")


class ISIS(BaseModel):
    window_ms: float = Field(100.0, description="Size of the window in milliseconds")
    bin_ms: float = Field(5.0, description="Size of the bin in milliseconds")


class Locations(BaseModel):
    method: str = Field("monopolar_triangulation", description="Method to determine locations")


class TemplateMetrics(BaseModel):
    upsampling_factor: int = Field(10, description="Upsampling factor")
    sparsity: Optional[str] = Field(None, description="Sparsity method")


class PrincipalComponents(BaseModel):
    n_components: int = Field(5, description="Number of principal components")
    mode: str = Field("by_channel_local", description="Mode of principal component analysis")
    whiten: bool = Field(True, description="Whiten the components")


class PostprocessingParamsModel(BaseModel):
    sparsity: Sparsity
    waveforms_deduplicate: Waveforms
    waveforms: Waveforms
    spike_amplitudes: SpikeAmplitudes
    similarity: Similarity
    correlograms: Correlograms
    isis: ISIS
    locations: Locations
    template_metrics: TemplateMetrics
    principal_components: PrincipalComponents
    quality_metrics: QualityMetrics
    duplicate_threshold: float = Field(0.9, description="Duplicate threshold")
