import warnings
import os
import numpy as np
from pathlib import Path
import shutil
import json
import time
from datetime import datetime
import spikeinterface as si
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.curation as sc

from .models import PostprocessingParamsModel


warnings.filterwarnings("ignore")

n_jobs_co = os.getenv('CO_CPUS')
n_jobs = int(n_jobs_co) if n_jobs_co is not None else -1

job_kwargs = {
    'n_jobs': n_jobs,
    'chunk_duration': '1s',
    'progress_bar': True
}

data_folder = Path("../data/")
results_folder = Path("../results/")
tmp_folder = results_folder / "tmp"
tmp_folder.mkdir()


def postprocessing(
    data_folder: Path,
    results_folder: Path,
    job_kwargs: dict,
    postprocessing_params: PostprocessingParamsModel,
) -> None:
    data_process_prefix = "data_process_postprocessing"
    si.set_global_job_kwargs(**job_kwargs)
    print("\nPOSTPROCESSING")
    t_postprocessing_start_all = time.perf_counter()

    # check if test
    if (data_folder / "preprocessing_pipeline_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        preprocessed_folder = data_folder / "preprocessing_pipeline_output_test"
        spikesorted_folder = data_folder / "spikesorting_pipeline_output_test"
    else:
        preprocessed_folder = data_folder
        spikesorted_folder = data_folder

    preprocessed_folders = [p for p in preprocessed_folder.iterdir() if p.is_dir() and "preprocessed_" in p.name]

    # load job json files
    job_config_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    print(f"Found {len(job_config_json_files)} json configurations")

    if len(job_config_json_files) > 0:
        recording_names = []
        for json_file in job_config_json_files:
            with open(json_file, "r") as f:
                config = json.load(f)
            recording_name = config["recording_name"]
            assert (preprocessed_folder / f"preprocessed_{recording_name}").is_dir(), f"Preprocessed folder for {recording_name} not found!"
            recording_names.append(recording_name)
    else:
        recording_names = [("_").join(p.name.split("_")[1:]) for p in preprocessed_folders]

    for recording_name in recording_names:
        datetime_start_postprocessing = datetime.now()
        t_postprocessing_start = time.perf_counter()
        postprocessing_notes = ""

        print(f"\tProcessing {recording_name}")
        postprocessing_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
        postprocessing_output_folder = results_folder / f"postprocessed_{recording_name}"
        postprocessing_sorting_output_folder = results_folder / f"postprocessed-sorting_{recording_name}"

        recording = si.load_extractor(preprocessed_folder / f"preprocessed_{recording_name}")
        # make sure we have spikesorted output for the block-stream
        sorted_folder = spikesorted_folder / f"spikesorted_{recording_name}"
        if not sorted_folder.is_dir():
            raise FileNotFoundError(f"Spike sorted data for {recording_name} not found!")

        sorting = si.load_extractor(sorted_folder)

        # first extract some raw waveforms in memory to deduplicate based on peak alignment
        wf_dedup_folder = tmp_folder / "postprocessed" / recording_name
        we_raw = si.extract_waveforms(
            recording,
            sorting,
            folder=wf_dedup_folder,
            **postprocessing_params.waveforms_deduplicate.model_dump()
        )

        # de-duplication
        sorting_deduplicated = sc.remove_redundant_units(
            we_raw,
            duplicate_threshold=postprocessing_params.duplicate_threshold
        )
        print(f"\tNumber of original units: {len(we_raw.sorting.unit_ids)} -- Number of units after de-duplication: {len(sorting_deduplicated.unit_ids)}")
        n_duplicated = int(len(sorting.unit_ids) - len(sorting_deduplicated.unit_ids))
        postprocessing_notes += f"\n- Removed {n_duplicated} duplicated units.\n"
        deduplicated_unit_ids = sorting_deduplicated.unit_ids
        
        # use existing deduplicated waveforms to compute sparsity
        sparsity_raw = si.compute_sparsity(we_raw, **sparsity_params)
        sparsity_mask = sparsity_raw.mask[sorting.ids_to_indices(deduplicated_unit_ids), :]
        sparsity = si.ChannelSparsity(mask=sparsity_mask, unit_ids=deduplicated_unit_ids, channel_ids=recording.channel_ids)
        shutil.rmtree(wf_dedup_folder)
        del we_raw

        # this is a trick to make the postprocessed folder "self-contained
        sorting_deduplicated = sorting_deduplicated.save(folder=postprocessing_sorting_output_folder)

        # now extract waveforms on de-duplicated units
        print("\tSaving sparse de-duplicated waveform extractor folder")
        we = si.extract_waveforms(
            recording,
            sorting_deduplicated,
            folder=postprocessing_output_folder,
            sparsity=sparsity,
            sparse=True,
            overwrite=True,
            **postprocessing_params.waveforms.model_dump()
        )

        print("\tComputing spike amplitides")
        amps = spost.compute_spike_amplitudes(we, **postprocessing_params.spike_amplitudes.model_dump())

        print("\tComputing unit locations")
        unit_locs = spost.compute_unit_locations(we, **postprocessing_params.locations.model_dump())

        print("\tComputing spike locations")
        spike_locs = spost.compute_spike_locations(we, **postprocessing_params.locations.model_dump())

        print("\tComputing correlograms")
        corr = spost.compute_correlograms(we, **postprocessing_params.correlograms.model_dump())

        print("\tComputing ISI histograms")
        tm = spost.compute_isi_histograms(we, **postprocessing_params.isis.model_dump())

        print("\tComputing template similarity")
        sim = spost.compute_template_similarity(we, **postprocessing_params.similarity.model_dump())

        print("\tComputing template metrics")
        tm = spost.compute_template_metrics(we, **postprocessing_params.template_metrics.model_dump())

        print("\tComputing PCA")
        pc = spost.compute_principal_components(we, **postprocessing_params.principal_components.model_dump())

        print("\tComputing quality metrics")
        qm = sqm.compute_quality_metrics(we, **postprocessing_params.quality_metrics.model_dump())

        t_postprocessing_end = time.perf_counter()
        elapsed_time_postprocessing = np.round(t_postprocessing_end - t_postprocessing_start, 2)

    t_postprocessing_end_all = time.perf_counter()
    elapsed_time_postprocessing_all = np.round(t_postprocessing_end_all - t_postprocessing_start_all, 2)
    print(f"POSTPROCESSING time: {elapsed_time_postprocessing_all}s")