import warnings
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from spikeinterface.core.core_tools import check_json

from .models import PreprocessingParamsModel


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


def preprocessing(
    data_folder: Path,
    results_folder: Path,
    job_kwargs: dict,
    preprocessing_params: PreprocessingParamsModel,
    debug: bool = False,
    duration_s: float = 1.
) -> None:
    """
    Preprocessing pipeline for ephys data.
    """

    data_process_prefix = "data_process_preprocessing"

    if debug:
        print(f"DEBUG ENABLED - Only running with {duration_s} seconds")

    si.set_global_job_kwargs(**job_kwargs)

    # load job json files
    job_config_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    print(f"Found {len(job_config_json_files)} json configurations")

    if len(job_config_json_files) > 0:
        print("\n\nPREPROCESSING")
        t_preprocessing_start_all = time.perf_counter()
        preprocessing_vizualization_data = {}
        print(f"Preprocessing strategy: {preprocessing_params.preprocessing_strategy}")

        for job_config_file in job_config_json_files:
            datetime_start_preproc = datetime.now()
            t_preprocessing_start = time.perf_counter()
            preprocessing_notes = ""

            with open(job_config_file, "r") as f:
                job_config = json.load(f)
            session_name = job_config["session_name"]
            session_folder_path = job_config["session_folder_path"]

            session = data_folder / session_folder_path
            assert session.is_dir(), (
                f"Could not find {session_name} in {str((data_folder / session_folder_path).resolve())}."
                f"Make sure mapping is correct!"
            )

            ecephys_full_folder = session / "ecephys"
            ecephys_compressed_folder = session / "ecephys_compressed"
            compressed = False
            if ecephys_compressed_folder.is_dir():
                compressed = True
                ecephys_folder = session / "ecephys_clipped"
            else:
                ecephys_folder = ecephys_full_folder

            experiment_name = job_config["experiment_name"]
            stream_name = job_config["stream_name"]
            block_index = job_config["block_index"]
            segment_index = job_config["segment_index"]
            recording_name = job_config["recording_name"]

            preprocessing_vizualization_data[recording_name] = {}
            preprocessing_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
            preprocessing_output_folder = results_folder / f"preprocessed_{recording_name}"
            preprocessingviz_output_file = results_folder / f"preprocessedviz_{recording_name}.json"
            preprocessing_output_json = results_folder / f"preprocessed_{recording_name}.json"

            exp_stream_name = f"{experiment_name}_{stream_name}"
            if not compressed:
                recording = se.read_openephys(
                    ecephys_folder,
                    stream_name=stream_name,
                    block_index=block_index
                )
            else:
                recording = si.read_zarr(ecephys_compressed_folder / f"{exp_stream_name}.zarr")

            if debug:
                recording_list = []
                for segment_index in range(recording.get_num_segments()):
                    recording_one = si.split_recording(recording)[segment_index]
                    recording_one = recording_one.frame_slice(
                        start_frame=0,
                        end_frame=int(duration_s*recording.sampling_frequency)
                    )
                    recording_list.append(recording_one)
                recording = si.append_recordings(recording_list)

            if segment_index is not None:
                recording = si.split_recording(recording)[segment_index]

            print(f"Preprocessing recording: {recording_name}")
            print(f"\tDuration: {np.round(recording.get_total_duration(), 2)} s")

            recording_ps_full = spre.phase_shift(
                recording,
                **preprocessing_params.phase_shift.model_dump()
            )
            recording_hp_full = spre.highpass_filter(
                recording_ps_full,
                **preprocessing_params.highpass_filter.model_dump()
            )
            preprocessing_vizualization_data[recording_name]["timeseries"] = {}
            preprocessing_vizualization_data[recording_name]["timeseries"]["full"] = dict(
                raw=recording.to_dict(relative_to=data_folder, recursive=True),
                phase_shift=recording_ps_full.to_dict(relative_to=data_folder, recursive=True),
                highpass=recording_hp_full.to_dict(relative_to=data_folder, recursive=True)
            )

            # IBL bad channel detection
            _, channel_labels = spre.detect_bad_channels(
                recording_hp_full,
                **preprocessing_params.detect_bad_channels.model_dump()
            )
            dead_channel_mask = channel_labels == "dead"
            noise_channel_mask = channel_labels == "noise"
            out_channel_mask = channel_labels == "out"
            print(f"\tBad channel detection:")
            print(f"\t\t- dead channels - {np.sum(dead_channel_mask)}\n\t\t- noise channels - {np.sum(noise_channel_mask)}\n\t\t- out channels - {np.sum(out_channel_mask)}")
            dead_channel_ids = recording_hp_full.channel_ids[dead_channel_mask]
            noise_channel_ids = recording_hp_full.channel_ids[noise_channel_mask]
            out_channel_ids = recording_hp_full.channel_ids[out_channel_mask]

            all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

            skip_processing = False
            max_bad_channel_fraction_to_remove = preprocessing_params.max_bad_channel_fraction_to_remove
            if len(all_bad_channel_ids) >= int(max_bad_channel_fraction_to_remove * recording.get_num_channels()):
                print(f"\tMore than {max_bad_channel_fraction_to_remove * 100}% bad channels ({len(all_bad_channel_ids)}). "
                      f"Skipping further processing for this recording.")            
                preprocessing_notes += f"\n- Found {len(all_bad_channel_ids)} bad channels. Skipping further processing\n"
                skip_processing = True
                # in this case, processed timeseries will not be visualized
                preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = None
                recording_drift = recording_hp_full
            else:
                if preprocessing_params.remove_out_channels:
                    print(f"\tRemoving {len(out_channel_ids)} out channels")
                    recording_rm_out = recording_hp_full.remove_channels(out_channel_ids)
                    preprocessing_notes += f"\n- Removed {len(out_channel_ids)} outside of the brain."
                else:
                    recording_rm_out = recording_hp_full

                recording_processed_cmr = spre.common_reference(
                    recording_rm_out,
                    **preprocessing_params.common_reference.model_dump()
                )

                bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids))
                recording_interp = spre.interpolate_bad_channels(recording_rm_out, bad_channel_ids)
                recording_hp_spatial = spre.highpass_spatial_filter(
                    recording_interp,
                    **preprocessing_params.highpass_spatial_filter.model_dump()
                )
                preprocessing_vizualization_data[recording_name]["timeseries"]["proc"] = dict(
                    highpass=recording_rm_out.to_dict(relative_to=data_folder, recursive=True),
                    cmr=recording_processed_cmr.to_dict(relative_to=data_folder, recursive=True),
                    highpass_spatial=recording_hp_spatial.to_dict(relative_to=data_folder, recursive=True)
                )

                if preprocessing_params.preprocessing_strategy == "cmr":
                    recording_processed = recording_processed_cmr
                else:
                    recording_processed = recording_hp_spatial

                if preprocessing_params.remove_bad_channels:
                    print(f"\tRemoving {len(bad_channel_ids)} channels after {preprocessing_params.preprocessing_strategy} preprocessing")
                    recording_processed = recording_processed.remove_channels(bad_channel_ids)
                    preprocessing_notes += f"\n- Removed {len(bad_channel_ids)} bad channels after preprocessing.\n"

                # motion correction
                if preprocessing_params.motion_correction.compute:
                    preset = preprocessing_params.motion_correction.preset
                    print(f"\tComputing motion correction with preset: {preset}")
                    motion_folder = results_folder / f"motion_{recording_name}"
                    recording_corrected = spre.correct_motion(
                        recording_processed, preset=preset,
                        folder=motion_folder,
                        **job_kwargs
                    )
                    if preprocessing_params.motion_correction.apply:
                        print("\tApplying motion correction")
                        recording_processed = recording_corrected

                recording_saved = recording_processed.save(folder=preprocessing_output_folder)
                recording_processed.dump_to_json(preprocessing_output_json, relative_to=data_folder)
                recording_drift = recording_saved

                # store recording for drift visualization
                preprocessing_vizualization_data[recording_name]["drift"] = dict(
                                                        recording=recording_drift.to_dict(relative_to=data_folder, recursive=True)
                                                    )
                with open(preprocessingviz_output_file, "w") as f:
                    json.dump(check_json(preprocessing_vizualization_data), f, indent=4)

            t_preprocessing_end = time.perf_counter()
            elapsed_time_preprocessing = np.round(t_preprocessing_end - t_preprocessing_start, 2)

            # save params in output
            preprocessing_params["recording_name"] = recording_name
            preprocessing_outputs = dict(channel_labels=channel_labels.tolist())
            # preprocessing_process = DataProcess(
            #         name="Ephys preprocessing",
            #         version=VERSION, # either release or git commit
            #         start_date_time=datetime_start_preproc,
            #         end_date_time=datetime_start_preproc + timedelta(seconds=np.floor(elapsed_time_preprocessing)),
            #         input_location=str(data_folder),
            #         output_location=str(results_folder),
            #         code_url=URL,
            #         parameters=preprocessing_params,
            #         outputs=preprocessing_outputs,
            #         notes=preprocessing_notes
            #     )
            # with open(preprocessing_output_process_json, "w") as f:
            #     f.write(preprocessing_process.json(indent=3))

        t_preprocessing_end_all = time.perf_counter()
        elapsed_time_preprocessing_all = np.round(t_preprocessing_end_all - t_preprocessing_start_all, 2)

        print(f"PREPROCESSING time: {elapsed_time_preprocessing_all}s")