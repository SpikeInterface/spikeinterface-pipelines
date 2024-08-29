

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
    sorting_analyzer: si.SortingAnalyzer | None = None,
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
    sorting_analyzer: si.SortingAnalyzer | None
        The input sorting analyzer from postprocessing. If None, only the recording visualization will be generated.
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

    kcl_client = None
    try:
        kcl_client = kcl.get_client_info()
    except:
        pass
    if kcl_client is None:
        logger.info(
            "[Visualization] \tKachery client not found. Use `kachery-cloud-init` to initialize kachery client."
        )
    visualization_params_dict = visualization_params.model_dump()
    recording_params = visualization_params_dict["recording"]

    # Recording visualization
    drift_params = recording_params["drift"]

    # use spike locations
    skip_drift = False
    spike_locations_available = False
    # use spike locations
    if sorting_analyzer is not None:
        if sorting_analyzer.has_extension("spike_locations"):
            logger.info("[Visualization] \tVisualizing drift maps using pre-computed spike locations")
            spike_locations_available = True

    # if spike locations are not available, detect and localize peaks
    if not spike_locations_available:
        from spikeinterface.core.node_pipeline import ExtractDenseWaveforms, run_node_pipeline
        from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive
        from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass

        logger.info("[Visualization] \tVisualizing drift maps using detected peaks (no spike sorting available)")
        # locally_exclusive + pipeline steps LocalizeCenterOfMass + PeakToPeakFeature
        peak_detector_node = DetectPeakLocallyExclusive(recording, **drift_params["detection"])
        extract_dense_waveforms_node = ExtractDenseWaveforms(
            recording,
            ms_before=drift_params["localization"]["ms_before"],
            ms_after=drift_params["localization"]["ms_after"],
            parents=[peak_detector_node],
            return_output=False,
        )
        localize_peaks_node = LocalizeCenterOfMass(
            recording,
            radius_um=drift_params["localization"]["radius_um"],
            parents=[peak_detector_node, extract_dense_waveforms_node],
        )
        pipeline_nodes = [peak_detector_node, extract_dense_waveforms_node, localize_peaks_node]
        peaks, peak_locations = run_node_pipeline(
            recording, nodes=pipeline_nodes, job_kwargs=si.get_global_job_kwargs()
        )
        logger.info("[Visualization] \t\tDetected {len(peaks)} peaks")
        peak_amps = peaks["amplitude"]
        if len(peaks) == 0:
            logger.info("[Visualization] \t\tNo peaks detected. Skipping drift map")
            skip_drift = True

    if not skip_drift:
        fig_drift, axs_drift = plt.subplots(
            ncols=recording.get_num_segments(), figsize=drift_params["figsize"]
        )
        y_locs = recording.get_channel_locations()[:, 1]
        depth_lim = [np.min(y_locs), np.max(y_locs)]

        for segment_index in range(recording.get_num_segments()):
            if recording.get_num_segments() == 1:
                ax_drift = axs_drift
            else:
                ax_drift = axs_drift[segment_index]
            if spike_locations_available:
                sorting_analyzer_to_plot = sorting_analyzer
                peaks_to_plot = None
                peak_locations_to_plot = None
                sampling_frequency = None
            else:
                sorting_analyzer_to_plot = None
                peaks_to_plot = peaks
                peak_locations_to_plot = peak_locations
                sampling_frequency = recording.sampling_frequency

            _ = sw.plot_drift_raster_map(
                sorting_analyzer=sorting_analyzer_to_plot,
                peaks=peaks_to_plot,
                peak_locations=peak_locations_to_plot,
                sampling_frequency=sampling_frequency,
                segment_index=segment_index,
                depth_lim=depth_lim,
                clim=(drift_params["vmin"], drift_params["vmax"]),
                cmap=drift_params["cmap"],
                scatter_decimate=drift_params["scatter_decimate"],
                alpha=drift_params["alpha"],
                ax=ax_drift
            )
            ax_drift.spines["top"].set_visible(False)
            ax_drift.spines["right"].set_visible(False)

    fig_drift_folder = results_folder / "drift_maps"
    fig_drift_folder.mkdir(exist_ok=True)
    fig_drift.savefig(fig_drift_folder / f"drift.png", dpi=300)

    # make a sorting view View
    v_drift = svv.TabLayoutItem(label=f"Drift map", view=svv.Image(image_path=str(fig_drift_folder / f"drift.png")))

    # TODO add motion

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
    if sorting_analyzer is None:
        logger.info("[Visualization] \tNo sorting analyzer found. Skipping sorting summary visualization")
        return visualization_output

    logger.info("[Visualization] \tVisualizing sorting summary")
    # set sorting_analyzer sorting object to have pass_qc property
    if sorting_curated is not None:
        sorting_analyzer.sorting = sorting_curated

    sorting_summary_params = visualization_params_dict["sorting_summary"]

    if len(sorting_analyzer.unit_ids) > 0:
        unit_table_properties = sorting_summary_params["unit_table_properties"]
        v_sorting = sw.plot_sorting_summary(
            sorting_analyzer,
            unit_table_properties=sorting_summary_params["unit_table_properties"],
            curation=sorting_summary_params["unit_table_properties"],
            label_choices=sorting_summary_params["label_choices"],
            backend="sortingview",
            generate_url=False,
        ).view

        try:
            # pre-generate gh for curation
            url = v_sorting.url(label=sorting_summary_params["label"])
            print(f"\n{url}\n")
            visualization_output["sorting_summary"] = url
        except Exception as e:
            logger.info(f"[Visualization] \tSortingview visualization failed with error:\n{e}")
    else:
        logger.info("[Visualization] \tNo units to visualize")

    return visualization_output
