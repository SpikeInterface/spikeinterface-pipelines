from pydantic import BaseModel, Field, ConfigDict
from typing import Union, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from typing import Union, List


class SorterName(str, Enum):
    kilosort25 = "kilosort2_5"
    kilosort3 = "kilosort3"
    kilosort4 = "kilosort4"
    mountainsort5 = "mountainsort5"
    # spykingcircus2 = "spykingcircus2"


class Kilosort25Model(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detect_threshold: float = Field(default=6, description="Threshold for spike detection")
    projection_threshold: List[float] = Field(default=[10, 4], description="Threshold on projections")
    preclust_threshold: float = Field(
        default=8, description="Threshold crossings for pre-clustering (in PCA projection space)"
    )
    car: bool = Field(default=True, description="Enable or disable common reference")
    minFR: float = Field(
        default=0.1, description="Minimum spike rate (Hz), if a cluster falls below this for too long it gets removed"
    )
    minfr_goodchannels: float = Field(default=0.1, description="Minimum firing rate on a 'good' channel")
    nblocks: int = Field(
        default=5,
        description="blocks for registration. 0 turns it off, 1 does rigid registration. Replaces 'datashift' option.",
    )
    sig: float = Field(default=20, description="spatial smoothness constant for registration")
    freq_min: float = Field(default=150, description="High-pass filter cutoff frequency")
    sigmaMask: float = Field(default=30, description="Spatial constant in um for computing residual variance of spike")
    lam: float = Field(
        default=10.0,
        description="The importance of the amplitude penalty (like in Kilosort1: 0 means not used, 10 is average, 50 is a lot)",
    )
    nPCs: int = Field(default=3, description="Number of PCA dimensions")
    ntbuff: int = Field(default=64, description="Samples of symmetrical buffer for whitening and spike detection")
    nfilt_factor: int = Field(default=4, description="Max number of clusters per good channel (even temporary ones) 4")
    NT: Optional[int] = Field(default=None, description="Batch size (if None it is automatically computed)")
    AUCsplit: float = Field(
        default=0.9,
        description="Threshold on the area under the curve (AUC) criterion for performing a split in the final step",
    )
    do_correction: bool = Field(default=True, description="If True drift registration is applied")
    wave_length: float = Field(
        default=61, description="size of the waveform extracted around each detected peak, (Default 61, maximum 81)"
    )
    keep_good_only: bool = Field(default=False, description="If True only 'good' units are returned")
    skip_kilosort_preprocessing: bool = Field(
        default=False, description="Can optionaly skip the internal kilosort preprocessing"
    )
    scaleproc: int = Field(default=-1, description="int16 scaling of whitened data, if -1 set to 200.")


class Kilosort3Model(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pass

class Kilosort4Model(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pass

class MountainSort5Model(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scheme: str = Field(default="2", description="Sorting scheme", json_schema_extra={"options": ["1", "2", "3"]})
    detect_threshold: float = Field(default=5.5, description="Threshold for spike detection")
    detect_sign: int = Field(default=-1, description="Sign of the peak")
    detect_time_radius_msec: float = Field(default=0.5, description="Time radius in milliseconds")
    snippet_T1: int = Field(default=20, description="Snippet T1")
    snippet_T2: int = Field(default=20, description="Snippet T2")
    npca_per_channel: int = Field(default=3, description="Number of PCA per channel")
    npca_per_subdivision: int = Field(default=10, description="Number of PCA per subdivision")
    snippet_mask_radius: int = Field(default=250, description="Snippet mask radius")
    scheme1_detect_channel_radius: int = Field(default=150, description="Scheme 1 detect channel radius")
    scheme2_phase1_detect_channel_radius: int = Field(default=200, description="Scheme 2 phase 1 detect channel radius")
    scheme2_detect_channel_radius: int = Field(default=50, description="Scheme 2 detect channel radius")
    scheme2_max_num_snippets_per_training_batch: int = Field(
        default=200, description="Scheme 2 max number of snippets per training batch"
    )
    scheme2_training_duration_sec: int = Field(default=300, description="Scheme 2 training duration in seconds")
    scheme2_training_recording_sampling_mode: str = Field(
        default="uniform", description="Scheme 2 training recording sampling mode"
    )
    scheme3_block_duration_sec: int = Field(default=1800, description="Scheme 3 block duration in seconds")
    freq_min: int = Field(default=300, description="High-pass filter cutoff frequency")
    freq_max: int = Field(default=6000, description="Low-pass filter cutoff frequency")
    filter: bool = Field(default=True, description="Enable or disable filter")
    whiten: bool = Field(default=True, description="Enable or disable whiten")


class Kilosort4Model(BaseModel):
    batch_size: int = Field(default=60000, description="Number of samples included in each batch of data.")
    nblocks: int = Field(default=1, description="Number of non-overlapping blocks for drift correction (additional nblocks-1 blocks are created in the overlaps). Default value: 1.")
    Th_universal: int = Field(default=9, description="Spike detection threshold for universal templates. Th(1) in previous versions of Kilosort. Default value: 9.")
    Th_learned: int = Field(default=8, description="Spike detection threshold for learned templates. Th(2) in previous versions of Kilosort. Default value: 8.")
    do_CAR: bool = Field(default=True, description="Whether to perform common average reference. Default value: True.")
    invert_sign: bool = Field(default=False, description="Invert the sign of the data. Default value: False.")
    nt: int = Field(default=61, description="Number of samples per waveform. Also size of symmetric padding for filtering. Default value: 61.")
    shift: Union[None, int] = Field(default=None, description="Scalar shift to apply to data before all other operations. Default None.")
    scale: Union[None, int] = Field(default=None, description="Scaling factor to apply to data before all other operations. Default None.")
    artifact_threshold: Union[None, int] = Field(default=None, description="If a batch contains absolute values above this number, it will be zeroed out under the assumption that a recording artifact is present. By default, the threshold is infinite (so that no zeroing occurs). Default value: None.")
    nskip: int = Field(default=25, description="Batch stride for computing whitening matrix. Default value: 25.")
    whitening_range: int = Field(default=32, description="Number of nearby channels used to estimate the whitening matrix. Default value: 32.")
    binning_depth: int = Field(default=5, description="For drift correction, vertical bin size in microns used for 2D histogram. Default value: 5.")
    sig_interp: int = Field(default=20, description="For drift correction, sigma for interpolation (spatial standard deviation). Approximate smoothness scale in units of microns. Default value: 20.")
    drift_smoothing: List[float] = Field(default=[0.5, 0.5, 0.5], description="Amount of gaussian smoothing to apply to the spatiotemporal drift estimation, for x,y,time axes in units of registration blocks (for x,y axes) and batch size (for time axis). The x,y smoothing has no effect for `nblocks = 1`.")
    nt0min: Union[None, int] = Field(default=None, description="Sample index for aligning waveforms, so that their minimum or maximum value happens here. Default of 20. Default value: None.")
    dmin: Union[None, int] = Field(default=None, description="Vertical spacing of template centers used for spike detection, in microns. Determined automatically by default. Default value: None.")
    dminx: int = Field(default=32, description="Horizontal spacing of template centers used for spike detection, in microns. Default value: 32.")
    min_template_size: int = Field(default=10, description="Standard deviation of the smallest, spatial envelope Gaussian used for universal templates. Default value: 10.")
    template_sizes: int = Field(default=5, description="Number of sizes for universal spike templates (multiples of the min_template_size). Default value: 5.")
    nearest_chans: int = Field(default=10, description="Number of nearest channels to consider when finding local maxima during spike detection. Default value: 10.")
    nearest_templates: int = Field(default=100, description="Number of nearest spike template locations to consider when finding local maxima during spike detection. Default value: 100.")
    max_channel_distance: Union[None, int] = Field(default=None, description="Templates farther away than this from their nearest channel will not be used. Also limits distance between compared channels during clustering. Default value: None.")
    templates_from_data: bool = Field(default=True, description="Indicates whether spike shapes used in universal templates should be estimated from the data or loaded from the predefined templates. Default value: True.")
    n_templates: int = Field(default=6, description="Number of single-channel templates to use for the universal templates (only used if templates_from_data is True). Default value: 6.")
    n_pcs: int = Field(default=6, description="Number of single-channel PCs to use for extracting spike features (only used if templates_from_data is True). Default value: 6.")
    Th_single_ch: int = Field(default=6, description="For single channel threshold crossings to compute universal- templates. In units of whitened data standard deviations. Default value: 6.")
    acg_threshold: float = Field(default=0.2, description='Fraction of refractory period violations that are allowed in the ACG compared to baseline; used to assign "good" units. Default value: 0.2.')
    ccg_threshold: float = Field(default=0.25, description="Fraction of refractory period violations that are allowed in the CCG compared to baseline; used to perform splits and merges. Default value: 0.25.")
    cluster_downsampling: int = Field(default=20, description="Inverse fraction of nodes used as landmarks during clustering (can be 1, but that slows down the optimization). Default value: 20.")
    cluster_pcs: int = Field(default=64, description="Maximum number of spatiotemporal PC features used for clustering. Default value: 64.")
    x_centers: Union[None, int] = Field(default=None, description="Number of x-positions to use when determining center points for template groupings. If None, this will be determined automatically by finding peaks in channel density. For 2D array type probes, we recommend specifying this so that centers are placed every few hundred microns.")
    duplicate_spike_bins: int = Field(default=7, description="Number of bins for which subsequent spikes from the same cluster are assumed to be artifacts. A value of 0 disables this step. Default value: 7.")
    do_correction: bool = Field(default=True, description="If True, drift correction is performed")
    save_extra_kwargs: bool = Field(default=False, description="If True, additional kwargs are saved to the output")
    skip_kilosort_preprocessing: bool = Field(default=False, description="Can optionally skip the internal kilosort preprocessing")
    scaleproc: Union[None, int] = Field(default=None, description="int16 scaling of whitened data, if None set to 200.")
    save_preprocessed_copy: bool = Field(default=False, description="save a pre-processed copy of the data (including drift correction) to temp_wh.dat in the results directory and format Phy output to use that copy of the data")
    torch_device: str = Field(default="auto", description="Select the torch device auto/cuda/cpu")
    bad_channels: Optional[List[int]] = Field(default=None, description="List of bad channels to exclude from spike detection and clustering.")



## SpykingCircus2 - WIP
# class SpykingCircus2GeneralModel(BaseModel):
#     ms_before: int = Field(default=2, description="ms before")
#     ms_after: int = Field(default=2, description="ms after")
#     radius_um: int = Field(default=100, description="radius um")


# class SpykingCircus2WaveformsModel(BaseModel):
#     max_spikes_per_unit: int = Field(default=200, description="Max spikes per unit")
#     overwrite: bool = Field(default=True, description="Overwrite")
#     sparse: bool = Field(default=True, description="Sparse")
#     method: str = Field(default="energy", description="Method")
#     threshold: float = Field(default=0.25, description="Threshold")


# class SpykingCircus2FilteringModel(BaseModel):
#     freq_min: int = Field(default=150, description="High-pass filter cutoff frequency")
#     dtype: str = Field(default="float32", description="Data type")


# class SpykingCircus2DetectionModel(BaseModel):
#     peak_sign: str = Field(default="neg", description="Peak sign")
#     detect_threshold: int = Field(default=4, description="Detect threshold")


# class SpykingCircus2SelectionModel(BaseModel):
#     method: str = Field(default="smart_sampling_amplitudes", description="Method")
#     n_peaks_per_channel: int = Field(default=5000, description="Number of peaks per channel")
#     min_n_peaks: int = Field(default=20000, description="Minimum number of peaks")
#     select_per_channel: bool = Field(default=False, description="Select per channel")


# class SpykingCircus2ClusteringModel(BaseModel):
#     legacy: bool = Field(default=False, description="Legacy")


# class SpykingCircus2CacheModel(BaseModel):
#     mode: str = Field(default="memory", description="Mode")
#     memory_limit: float = Field(default=0.5, description="Memory limit")
#     delete_cache: bool = Field(default=True, description="Delete cache")


# class SpykingCircus2Model(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     general: SpykingCircus2GeneralModel = Field(default=SpykingCircus2GeneralModel(), description="General parameters")
#     waveforms: SpykingCircus2WaveformsModel = Field(default=SpykingCircus2WaveformsModel(), description="Waveforms parameters")
#     filtering: SpykingCircus2FilteringModel = Field(default=SpykingCircus2FilteringModel(), description="Filtering parameters")
#     detection: SpykingCircus2DetectionModel = Field(default=SpykingCircus2DetectionModel(), description="Detection parameters")
#     selection: SpykingCircus2SelectionModel = Field(default=SpykingCircus2SelectionModel(), description="Selection parameters")
#     clustering: SpykingCircus2ClusteringModel = Field(default=SpykingCircus2ClusteringModel(), description="Clustering parameters")
#     apply_preprocessing: bool = Field(default=True, description="Apply preprocessing")
#     shared_memory: bool = Field(default=True, description="Shared memory")
#     cache_preprocessing: SpykingCircus2CacheModel = Field(default=SpykingCircus2CacheModel(), description="Cache preprocessing")


class SpikeSortingParams(BaseModel):
    sorter_name: SorterName = Field(description="Name of the sorter to use.")
    sorter_kwargs: Union[Kilosort25Model, Kilosort3Model, Kilosort4Model, MountainSort5Model] = Field(
        description="Sorter specific kwargs.", union_mode="left_to_right"
    )
    spikesort_by_group: bool = Field(
        default=False, description="If True, spike sorting is run for each group separately."
    )



