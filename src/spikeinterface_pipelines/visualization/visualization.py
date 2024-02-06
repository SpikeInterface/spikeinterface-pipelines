
from __future__ import annotations

from pathlib import Path
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import spikeinterface.full as si
import spikeinterface.widgets as sw

import sortingview.views as svv

import kachery_cloud as kcl
from spikeinterface.widgets import sorting_summary

from ..logger import logger

from .params import VisualizationParams

matplotlib.use("agg")


def visualize(
    recording: si.BaseRecording,
    sorting_curated: si.BaseSorting | None = None,
    waveform_extractor: si.WaveformExtractor | None = None,
    visualization_params: VisualizationParams = VisualizationParams(),
    scratch_folder: Path = Path("./scratch/"),
    results_folder: Path = Path("./results/visualization/"),
) -> dict | None:
    """ 
    Generate visualization of preprocessing, spikesorting and curation results.

    Parameters
    ----------
    recording_processed: si.BaseRecording
        The input processed recording
    sorting_curated: si.BaseSorting | None
        The input curated sorting. If None, only the recording visualization will be generated.
    waveform_extractor: si.WaveformExtractor | None
        The input waveform extractor from postprocessing. If None, only the recording visualization will be generated.
    visualization_params: VisualizationParams
        The visualization parameters
    scratch_folder: Path
        The scratch folder
    results_folder: Path
        The results folder

    Returns
    -------
    visualization_output: dict
        The visualization output dictionary    
    """
    logger.info("[Visualization] \tRunning Visualization stage")
    visualization_output = {}
    results_folder.mkdir(exist_ok=True, parents=True)

    if kcl.get_client_info() is None:
        logger.info(
            "[Visualization] \tKachery client not found. Use `kachery-cloud-init` to initialize kachery client."
        )
        return
    visualization_params_dict = visualization_params.model_dump()
    recording_params = visualization_params_dict["recording"]

    # Recording visualization
    cmap = plt.get_cmap(recording_params["drift"]["cmap"])
    norm = Normalize(
        vmin=recording_params["drift"]["vmin"], vmax=recording_params["drift"]["vmax"], clip=True
    )
    decimation_factor = recording_params["drift"]["decimation_factor"]
    alpha = recording_params["drift"]["alpha"]

    # use spike locations
    if not waveform_extractor.has_extension("quality_metrics"):
        logger.info("[Visualization] \tVisualizing drift maps using pre-computed spike locations")
        peaks = waveform_extractor.sorting.to_spike_vector()
        peak_locations = waveform_extractor.load_extension("spike_locations").get_data()
        peak_amps = np.concatenate(waveform_extractor.load_extension("spike_amplitudes").get_data())
    # otherwise detect peaks
    else:
        from spikeinterface.core.node_pipeline import ExtractDenseWaveforms, run_node_pipeline
        from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive
        from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass

        logger.info("[Visualization] \tVisualizing drift maps using detected peaks (no spike locations available)")

        # Here we use the node pipeline implementation
        peak_detector_node = DetectPeakLocallyExclusive(recording, **recording_params["drift"]["detection"])
        extract_dense_waveforms_node = ExtractDenseWaveforms(
            recording,
            ms_before=recording_params["drift"]["localization"]["ms_before"],
            ms_after=recording_params["drift"]["localization"]["ms_after"],
            parents=[peak_detector_node],
            return_output=False,
        )
        localize_peaks_node = LocalizeCenterOfMass(
            recording,
            radius_um=recording_params["drift"]["localization"]["radius_um"],
            parents=[peak_detector_node, extract_dense_waveforms_node],
        )
        job_kwargs = si.get_global_job_kwargs()
        pipeline_nodes = [peak_detector_node, extract_dense_waveforms_node, localize_peaks_node]
        peaks, peak_locations = run_node_pipeline(recording, nodes=pipeline_nodes, job_kwargs=job_kwargs)
        logger.info(f"[Visualization] \tDetected {len(peaks)} peaks")
        peak_amps = peaks["amplitude"]

    y_locs = recording.get_channel_locations()[:, 1]
    ylim = [np.min(y_locs), np.max(y_locs)]

    fig_drift, axs_drift = plt.subplots(
        ncols=recording.get_num_segments(), figsize=recording_params["drift"]["figsize"]
    )
    for segment_index in range(recording.get_num_segments()):
        segment_mask = peaks["segment_index"] == segment_index
        x = peaks[segment_mask]["sample_index"] / recording.sampling_frequency
        y = peak_locations[segment_mask]["y"]
        # subsample
        x_sub = x[::decimation_factor]
        y_sub = y[::decimation_factor]
        a_sub = peak_amps[::decimation_factor]
        colors = cmap(norm(a_sub))

        if recording.get_num_segments() == 1:
            ax_drift = axs_drift
        else:
            ax_drift = axs_drift[segment_index]
        ax_drift.scatter(x_sub, y_sub, s=1, c=colors, alpha=alpha)
        ax_drift.set_xlabel("time (s)", fontsize=12)
        ax_drift.set_ylabel("depth ($\\mu$m)", fontsize=12)
        ax_drift.set_xlim(0, recording.get_num_samples(segment_index=segment_index) / recording.sampling_frequency)
        ax_drift.set_ylim(ylim)
        ax_drift.spines["top"].set_visible(False)
        ax_drift.spines["right"].set_visible(False)
    fig_drift_folder = results_folder / "drift_maps"
    fig_drift_folder.mkdir(exist_ok=True)
    fig_drift.savefig(fig_drift_folder / f"drift.png", dpi=300)

    # make a sorting view View
    v_drift = svv.TabLayoutItem(
        label=f"Drift map", view=svv.Image(image_path=str(fig_drift_folder / f"drift.png"))
    )

    # timeseries
    if not recording_params["timeseries"]["skip"]:
        logger.info("[Visualization] \tVisualizing recording traces")
        timeseries_tab_items = []
        # get random chunks to estimate clims
        random_data_chunk = si.get_random_data_chunks(recording)
        max_value = np.quantile(random_data_chunk, 0.99) * 1.2
        clims = (-max_value, max_value)

        fs = recording.get_sampling_frequency()
        n_snippets_per_seg = recording_params["timeseries"]["n_snippets_per_segment"]
        try:
            for segment_index in range(recording.get_num_segments()):
                segment_duration = recording.get_num_samples(segment_index) / fs
                # evenly distribute t_starts across segments
                t_starts = np.linspace(0, segment_duration, n_snippets_per_seg + 2)[1:-1]
                for t_start in t_starts:
                    time_range = np.round(
                        np.array([t_start, t_start + recording_params["timeseries"]["snippet_duration_s"]]), 1
                    )
                    w_traces = sw.plot_traces(
                        recording,
                        order_channel_by_depth=True,
                        time_range=time_range,
                        segment_index=segment_index,
                        clim=clims,
                        mode="map",
                        backend="sortingview",
                        generate_url=False,
                    )
                    v_item = svv.TabLayoutItem(
                        label=f"Timeseries - Segment {segment_index} - Time: {time_range}", view=w_traces.view
                    )
                    timeseries_tab_items.append(v_item)
            # add drift map
            timeseries_tab_items.append(v_drift)

            v_timeseries = svv.TabLayout(items=timeseries_tab_items)
            try:
                url = v_timeseries.url(label=recording_params["label"])
                print(f"\n{url}\n")
                visualization_output["recording"] = url
            except Exception as e:
                print("KCL error", e)
        except Exception as e:
            print(f"Something wrong when visualizing timeseries: {e}")

    # Sorting summary
    if waveform_extractor is None:
        logger.info("[Visualization] \tNo waveform extractor found. Skipping sorting summary visualization")
        return visualization_output

    logger.info("[Visualization] \tVisualizing sorting summary")
    # set waveform_extractor sorting object to have pass_qc property
    if sorting_curated is not None:
        waveform_extractor.sorting = sorting_curated

    sorting_summary_params = visualization_params_dict["sorting_summary"]

    if len(waveform_extractor.unit_ids) > 0:
        unit_table_properties = sorting_summary_params["unit_table_properties"]
        # skip missing properties
        for prop in unit_table_properties:
            if prop not in waveform_extractor.sorting.get_property_keys():
                logger.info(
                    f"[Visualization] \tProperty {prop} not found in sorting object. "
                    "Not adding to unit table"
                )
                unit_table_properties.remove(prop)
        v_sorting = sw.plot_sorting_summary(
            waveform_extractor,
            unit_table_properties=sorting_summary_params["unit_table_properties"],
            curation=sorting_summary_params["unit_table_properties"],
            label_choices=sorting_summary_params["label_choices"],
            backend="sortingview",
            generate_url=False
        ).view

        try:
            # pre-generate gh for curation
            url = v_sorting.url(
                label=sorting_summary_params["label"]
            )
            print(f"\n{url}\n")
            visualization_output["sorting_summary"] = url
        except Exception as e:
            logger.info("[Visualization] \tSortingview visualization failed with error:\n{e}")
    else:
        logger.info("[Visualization] \tNo units to visualize")

    return visualization_output
