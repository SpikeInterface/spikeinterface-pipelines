import warnings
from pathlib import Path
import shutil

import spikeinterface as si
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc

from .params import PostprocessingParams
from ..logger import logger


warnings.filterwarnings("ignore")


def postprocess(
    recording: si.BaseRecording,
    sorting: si.BaseSorting,
    postprocessing_params: PostprocessingParams = PostprocessingParams(),
    scratch_folder: Path = Path("./scratch/"),
    results_folder: Path = Path("./results/postprocessing/"),
) -> si.WaveformExtractor:
    """
    Postprocess preprocessed and spike sorting output

    Parameters
    ----------
    recording: si.BaseRecording
        The input recording
    sorting: si.BaseSorting
        The input sorting
    postprocessing_params: PostprocessingParams
        Postprocessing parameters
    results_folder: Path
        Path to the results folder

    Returns
    -------
    si.WaveformExtractor
        The waveform extractor
    """

    tmp_folder = scratch_folder / "tmp_postprocessing"
    tmp_folder.mkdir(parents=True, exist_ok=True)

    # first extract some raw waveforms in memory to deduplicate based on peak alignment
    wf_dedup_folder = tmp_folder / "waveforms_dense"
    waveform_extractor_raw = si.extract_waveforms(
        recording, sorting, folder=wf_dedup_folder, sparse=False, **postprocessing_params.waveforms_raw.model_dump()
    )

    # de-duplication
    sorting_deduplicated = sc.remove_redundant_units(
        waveform_extractor_raw, duplicate_threshold=postprocessing_params.duplicate_threshold
    )
    logger.info(
        f"[Postprocessing] \tNumber of original units: {len(waveform_extractor_raw.sorting.unit_ids)} -- Number of units after de-duplication: {len(sorting_deduplicated.unit_ids)}"
    )
    deduplicated_unit_ids = sorting_deduplicated.unit_ids

    # use existing deduplicated waveforms to compute sparsity
    sparsity_raw = si.compute_sparsity(waveform_extractor_raw, **postprocessing_params.sparsity.model_dump())
    sparsity_mask = sparsity_raw.mask[sorting.ids_to_indices(deduplicated_unit_ids), :]
    sparsity = si.ChannelSparsity(mask=sparsity_mask, unit_ids=deduplicated_unit_ids, channel_ids=recording.channel_ids)

    # this is a trick to make the postprocessed folder "self-contained
    sorting_folder = results_folder / "sorting"
    sorting_deduplicated = sorting_deduplicated.save(folder=sorting_folder)

    # now extract waveforms on de-duplicated units
    logger.info("[Postprocessing] \tSaving sparse de-duplicated waveform extractor folder")
    waveform_extractor = si.extract_waveforms(
        recording,
        sorting_deduplicated,
        folder=results_folder / "waveforms",
        sparsity=sparsity,
        sparse=True,
        overwrite=True,
        **postprocessing_params.waveforms.model_dump(),
    )

    logger.info("[Postprocessing] \tComputing spike amplitides")
    _ = spost.compute_spike_amplitudes(waveform_extractor, **postprocessing_params.spike_amplitudes.model_dump())
    logger.info("[Postprocessing] \tComputing unit locations")
    _ = spost.compute_unit_locations(waveform_extractor, **postprocessing_params.locations.model_dump())
    logger.info("[Postprocessing] \tComputing spike locations")
    _ = spost.compute_spike_locations(waveform_extractor, **postprocessing_params.locations.model_dump())
    logger.info("[Postprocessing] \tComputing correlograms")
    _ = spost.compute_correlograms(waveform_extractor, **postprocessing_params.correlograms.model_dump())
    logger.info("[Postprocessing] \tComputing ISI histograms")
    _ = spost.compute_isi_histograms(waveform_extractor, **postprocessing_params.isis.model_dump())
    logger.info("[Postprocessing] \tComputing template similarity")
    _ = spost.compute_template_similarity(waveform_extractor, **postprocessing_params.similarity.model_dump())
    logger.info("[Postprocessing] \tComputing template metrics")
    _ = spost.compute_template_metrics(waveform_extractor, **postprocessing_params.template_metrics.model_dump())
    logger.info("[Postprocessing] \tComputing PCA")
    _ = spost.compute_principal_components(
        waveform_extractor, **postprocessing_params.principal_components.model_dump()
    )
    logger.info("[Postprocessing] \tComputing quality metrics")
    _ = sqm.compute_quality_metrics(waveform_extractor, **postprocessing_params.quality_metrics.model_dump())

    # Cleanup
    logger.info("[Postprocessing] \tCleaning up")
    shutil.rmtree(tmp_folder)

    return waveform_extractor
