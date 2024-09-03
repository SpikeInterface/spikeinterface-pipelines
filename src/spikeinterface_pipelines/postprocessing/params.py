from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
from enum import Enum


class PeakSign(str, Enum):
    neg = "neg"
    pos = "pos"
    both = "both"

class Sparsity(BaseModel):
    method: str = Field(default="radius", description="Method for determining sparsity.")
    radius_um: int = Field(default=100, description="Radius in micrometers for sparsity.")

class RandomSpikesParams(BaseModel):
    max_spikes_per_unit: int = Field(default=500, description="Maximum spikes per unit.")
    method: str = Field(default="uniform", description="Method for generating random spikes.")
    margin_size: Optional[int] = Field(default=None, description="Margin size.")
    seed: Optional[int] = Field(default=None, description="Seed for random number generator.")

class NoiseLevelsParams(BaseModel):
    num_chunks_per_segment: int = Field(default=20, description="Number of chunks per segment.")
    chunk_size: int = Field(default=10000, description="Size of the chunk.")
    seed: Optional[int] = Field(default=None, description="Seed for random number generator.")    

class Waveforms(BaseModel):
    ms_before: float = Field(default=1.0, description="Milliseconds before")
    ms_after: float = Field(default=2.0, description="Milliseconds after")
    dtype: Optional[str] = Field(default=None, description="Data type for the waveforms")

class Templates(BaseModel):
    ms_before: float = Field(default=1.0, description="Milliseconds before")
    ms_after: float = Field(default=2.0, description="Milliseconds after")
    operators : List[str] = Field(default=["average", "std"], description="Operators for the templates")

class SpikeAmplitudes(BaseModel):
    peak_sign: str = Field(default="neg", description="Sign of the peak")


class TemplateSimilarity(BaseModel):
    method: str = Field(default="cosine_similarity", description="Method to compute similarity")


class Correlograms(BaseModel):
    window_ms: float = Field(default=100.0, description="Size of the window in milliseconds")
    bin_ms: float = Field(default=2.0, description="Size of the bin in milliseconds")


class ISIHistograms(BaseModel):
    window_ms: float = Field(default=100.0, description="Size of the window in milliseconds")
    bin_ms: float = Field(default=5.0, description="Size of the bin in milliseconds")


class UnitLocations(BaseModel):
    method: str = Field(default="monopolar_triangulation", description="Method to determine locations")


class SpikeLocations(BaseModel):
    method: str = Field(default="grid_convolution", description="Method to determine locations")


class TemplateMetrics(BaseModel):
    upsampling_factor: int = Field(default=10, description="Upsampling factor")
    sparsity: Optional[str] = Field(default=None, description="Sparsity method")
    include_muti_channel_metrics: bool = Field(default=True, description="Flag to include multi-channel metrics")


class PrincipalComponents(BaseModel):
    n_components: int = Field(default=5, description="Number of principal components")
    mode: str = Field(default="by_channel_local", description="Mode of principal component analysis")
    whiten: bool = Field(default=True, description="Whiten the components")

### QUALITY METRICS ###
class AmplitudeCutoff(BaseModel):
    peak_sign: PeakSign = Field(default="neg", description="The sign of the peaks.")
    num_histogram_bins: int = Field(
        default=100, description="The number of bins to use to compute the amplitude histogram."
    )
    histogram_smoothing_value: int = Field(
        default=3, description="Controls the smoothing applied to the amplitude histogram."
    )
    amplitudes_bins_min_ratio: int = Field(
        default=5,
        description="The minimum ratio between number of amplitudes for a unit and the number of bins. If the ratio is less than this threshold, the amplitude_cutoff for the unit is set to NaN.",
    )

class AmplitudeCV(BaseModel):
    average_num_spikes_per_bin: int = Field(
        default=50, description="The average number of spikes per bin to compute the amplitude CV."
    )
    percentiles: List[int] = Field(
        default=[5, 95], description="The percentiles to use to compute the amplitude CV."
    )
    min_num_bins: int = Field(
        default=10, description="The minimum number of bins to use to compute the amplitude CV."
    )
    amplitude_extension: str = Field(
        default="spike_amplitudes", description="The extension to use to compute the amplitude CV."
    )


class AmplitudeMedian(BaseModel):
    peak_sign: PeakSign = Field(default="neg", description="The sign of the peaks.")


class Drift(BaseModel):
    interval_s: float = Field(default=60, description="The interval in seconds.")
    min_spikes_per_interval: int = Field(
        default=100, description="The minimum number of spikes per interval."
    )
    direction: str = Field(default="y", description="The direction to compute the drift.")
    min_num_bins: int = Field(default=2, description="The minimum number of bins to use to compute the drift.")


class FiringRange(BaseModel):
    bin_size_s: float = Field(default=5, description="The size of the bin in seconds.")
    percentiles: List[int] = Field(
        default=[5, 95], description="The percentiles to use to compute the firing range."
    )


class ISIViolation(BaseModel):
    isi_threshold_ms: float = Field(default=1.5, description="ISI threshold in milliseconds.")
    min_isi_ms: float = Field(default=0.0, description="Minimum ISI in milliseconds.")


class NearestNeighbor(BaseModel):
    max_spikes: int = Field(
        default=10000,
        description="The number of spikes to use, per cluster. Note that the calculation can be very slow when this number is >20000.",
    )
    min_spikes: int = Field(default=10, description="Minimum number of spikes.")
    n_neighbors: int = Field(default=4, description="The number of neighbors to use.")


class NNIsolation(NearestNeighbor):
    n_components: int = Field(default=10, description="The number of PC components to use to project the snippets to.")
    radius_um: int = Field(
        default=100, description="The radius, in um, that channels need to be within the peak channel to be included."
    )


class PresenceRatio(BaseModel):
    bin_duration_s: float = Field(default=60, description="Duration of the bin in seconds.")


class RPViolation(BaseModel):
    refractory_period_ms: float = Field(default=1.0, description="Refractory period in milliseconds.")
    censored_period_ms: float = Field(default=0.0, description="Censored period in milliseconds.")


class Silhouette(BaseModel):
    method: List[str] = Field(default=["simplified"], description="The method to use to compute the silhouette.")


class SlidingRPViolation(BaseModel):
    bin_size_ms: float = Field(
        default=0.25, description="The size of binning for the autocorrelogram in ms, by default 0.25."
    )
    window_size_s: float = Field(default=1, description="Window in seconds to compute correlogram, by default 1.")
    exclude_ref_period_below_ms: float = Field(
        default=0.5, description="Refractory periods below this value are excluded, by default 0.5"
    )
    max_ref_period_ms: float = Field(
        default=10, description="Maximum refractory period to test in ms, by default 10 ms."
    )
    contamination_values: Optional[list] = Field(
        default=None, description="The contamination values to test, by default np.arange(0.5, 35, 0.5) %"
    )


class SNR(BaseModel):
    peak_sign: str = Field(default="neg", description="Sign of the peak.")
    peak_mode: str = Field(default="extremum", description="Mode of the peak.")


class Synchrony(BaseModel):
    synchrony_sizes: List[int] = Field(
        default=[2, 4, 8], description="The synchrony sizes to compute the synchrony metric."
    )



class QMParams(BaseModel):
    amplitude_cv: AmplitudeCV = Field(default=AmplitudeCV(), description="Amplitude CV.")
    firing_range: FiringRange = Field(default=FiringRange(), description="Firing range.")
    synchorny: Synchrony = Field(default=Synchrony(), description="Synchrony.")
    silhouette: Silhouette = Field(default=Silhouette(), description="Silhouette.")
    presence_ratio: PresenceRatio = Field(default=PresenceRatio(), description="Presence ratio.")
    snr: SNR = Field(default=SNR(), description="Signal to noise ratio.")
    isi_violation: ISIViolation = Field(default=ISIViolation(), description="ISI violation.")
    rp_violation: RPViolation = Field(default=RPViolation(), description="Refractory period violation.")
    sliding_rp_violation: SlidingRPViolation = Field(
        default=SlidingRPViolation(), description="Sliding refractory period violation."
    )
    amplitude_cutoff: AmplitudeCutoff = Field(default=AmplitudeCutoff(), description="Amplitude cutoff.")
    amplitude_median: AmplitudeMedian = Field(default=AmplitudeMedian(), description="Amplitude median.")
    nearest_neighbor: NearestNeighbor = Field(default=NearestNeighbor(), description="Nearest neighbor.")
    nn_isolation: NNIsolation = Field(default=NNIsolation(), description="Nearest neighbor isolation.")
    nn_noise_overlap: NNIsolation = Field(default=NNIsolation(), description="Nearest neighbor noise overlap.")


class QualityMetrics(BaseModel):
    qm_params: QMParams = Field(default=QMParams(), description="Quality metric parameters.")
    metric_names: List[str] = Field(
        default=[
            "amplitude_cv",
            "amplitude_cutoff",
            "amplitude_median",
            "d_prime",
            "drift",
            "firing_range",
            "isi_violation",
            "l_ratio",
            "nearest_neighbor",
            "num_spikes",
            "presence_ratio",
            "rp_violation",
            "silhouette",
            "sliding_rp_violation",
            "snr",
            "synchrony"
        ],
        description="List of metric names to compute. If None, all available metrics are computed.",
    )


class QMParams(BaseModel):
    presence_ratio: PresenceRatio = Field(default=PresenceRatio(), description="Presence ratio.")
    snr: SNR = Field(default=SNR(), description="Signal to noise ratio.")
    isi_violation: ISIViolation = Field(default=ISIViolation(), description="ISI violation.")
    rp_violation: RPViolation = Field(default=RPViolation(), description="Refractory period violation.")
    sliding_rp_violation: SlidingRPViolation = Field(
        default=SlidingRPViolation(), description="Sliding refractory period violation."
    )
    amplitude_cutoff: AmplitudeCutoff = Field(default=AmplitudeCutoff(), description="Amplitude cutoff.")
    amplitude_median: AmplitudeMedian = Field(default=AmplitudeMedian(), description="Amplitude median.")
    nearest_neighbor: NearestNeighbor = Field(default=NearestNeighbor(), description="Nearest neighbor.")
    nn_isolation: NNIsolation = Field(default=NNIsolation(), description="Nearest neighbor isolation.")
    nn_noise_overlap: NNIsolation = Field(default=NNIsolation(), description="Nearest neighbor noise overlap.")

class ExtensionParams(BaseModel):
    random_spikes: RandomSpikesParams = Field(default=RandomSpikesParams(), description="Random spikes")
    noise_levels: NoiseLevelsParams = Field(default=NoiseLevelsParams(), description="Noise levels")
    waveforms: Waveforms = Field(default=Waveforms(), description="Waveforms")
    templates: Templates = Field(default=Templates(), description="Templates")
    spike_amplitudes: SpikeAmplitudes = Field(default=SpikeAmplitudes(), description="Spike amplitudes")
    template_similarity: TemplateSimilarity = Field(default=TemplateSimilarity(), description="Template similarity")
    correlograms: Correlograms = Field(default=Correlograms(), description="Correlograms")
    isi_histograms: ISIHistograms = Field(default=ISIHistograms(), description="ISIHistograms")
    unit_locations: UnitLocations = Field(default=UnitLocations(), description="Unit locations")
    principal_components: PrincipalComponents = Field(default=PrincipalComponents(), description="Principal components")
    spike_locations: SpikeLocations = Field(default=SpikeLocations(), description="Spike locations")
    template_metrics: TemplateMetrics = Field(default=TemplateMetrics(), description="Template metrics")
    

class QualityMetrics(BaseModel):
    qm_params: QMParams = Field(default=QMParams(), description="Quality metric parameters.")
    metric_names: List[str] = Field(
        default=[
            "num_spikes",
            "firing_rate",
            "presence_ratio",
            "snr",
            "isi_violation",
            "rp_violation",
            "sliding_rp_violation",
            "amplitude_cutoff",
            "amplitude_median",
            "amplitude_cv",
            "synchrony",
            "firing_range",
            "drift",
            "isolation_distance",
            "l_ratio",
            "d_prime",
            "nearest_neighbor",
            "silhouette"
        ],
        description="List of metric names to compute. If None, all available metrics are computed.",
    )
    n_jobs: int = Field(default=-1, description="Number of jobs.")



class PostprocessingParams(BaseModel):
    sparsity: Sparsity = Field(default=Sparsity(), description="Sparsity")
    duplicate_threshold: float = Field(default=0.9, description="Duplicate threshold")
    return_scaled: bool = Field(default=True, description="Return scaled")
    extension_params: ExtensionParams = Field(default=ExtensionParams(), description="Extension parameters")
    quality_metrics: QualityMetrics = Field(default=QualityMetrics(), description="Quality metrics")
