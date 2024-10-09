# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import datetime
import logging
import os
import textwrap
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tqdm
from jet.logs.queries import Field, JETLogsQuery
from jet.utils.instance import JETInstance
from tabulate import tabulate


# using logging not from NeMo to run this script outside a container
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# ANSI escape sequences for colors and formatting
class AnsiCodes:
    RED = "\033[31m"
    GREEN = "\033[32m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _apply_color_str(text: str, color: str) -> str:
    """
    Incorporates colouring to string and resets it at the end of string.
    Args:
        text: some string
        color: intended color of the string in ANSI colour format
    Returns: a coloured string with the reset colour appended

    """
    return color + text + AnsiCodes.RESET


def _wrap_text(text: str, width: int = 90) -> str:
    """
    Adds text wrapper around string to ensure it will break across many lines
    Args:
        text: a string
        width: max number of elements in string in a single line
    Returns: a string with additional "\n" indicating line breaking

    """
    wrapped_text = textwrap.fill(text, width=width)
    return wrapped_text


def filter_out_failed_and_rerun_jobs(df: pd.DataFrame, n_all_results: int) -> pd.DataFrame:
    """
    Filters out records of jobs which have been rerun, ie due to infrastructure failures.
    """
    if df["timestamp"].isna().any():
        raise ValueError(
            "The 'timestamp' column contains None values. Cannot specify time order of JET jobs"
            " to determine the retried ones."
        )
    df["tmp_label"] = df["jet_ci_pipeline_id"].astype(str) + df["job_key"]
    df.drop_duplicates("tmp_label", keep="last", inplace=True)
    df.drop(labels="tmp_label", axis=1, inplace=True)
    print("\n")
    logging.info(
        f"Keeping only the most recent jobs per pipeline id and job key: " f"{df.shape[0]}/{n_all_results} jobs \n"
    )
    return df


def print_table_with_results(df: pd.DataFrame) -> None:
    """
    Print table with essential information about the jobs to the console table
    """
    table_list = []
    for _, row in df.iterrows():
        table_dict = OrderedDict()
        table_dict["JET job ID"] = row["jet_ci_job_id"]
        table_dict["Job Type"] = row["job_type"]
        table_dict["Job Key"] = _wrap_text(text=row["job_key"].split("/")[1])
        table_dict["Job Status"] = row["jet_ci_job_status"].lower()

        # Determine condition for failure
        if "jet_test_status" in row:
            table_dict["Test Status"] = (
                row["jet_test_status"].lower() if isinstance(row["jet_test_status"], str) else ""
            )
            condition = table_dict["Test Status"] == "failed"
        else:
            condition = table_dict["Job Status"] == "failed"

        # Apply red color if the status is 'failed'
        if condition:
            for key in table_dict:
                table_dict[key] = _apply_color_str(str(table_dict[key]), color=AnsiCodes.RED)

        table_list.append(table_dict)

    # Creating table with tabulate
    table = tabulate(table_list, headers="keys", tablefmt="grid")
    print(table)

    if any(df.jet_ci_job_status.str.lower() != "success"):
        logging.error(
            "Some jobs failed to complete successfully. Detailed information about jobs is available by appending -vv\n"
        )
    else:
        logging.info("All jobs completed successfully!")

    if "jet_test_status" in df and any(df.dropna(subset=["jet_test_status"]).jet_test_status.str.lower() != "success"):
        logging.error("Some tests failed. Detailed information about tests is available by appending -vv\n")
    else:
        logging.info("All tests completed successfully!")


def get_duration_stats(
    pipelines_info: dict, job_durations: pd.Series, script_durations: pd.Series, digits: int = 1
) -> None:
    """
    Calculates duration statistics for pipelines(s) and included jobs. Differentiates between
    job duration (including queuing time) and script duration (excluding queuing time)
    """
    print("\n")
    N = len(job_durations)
    pipelines_duration = pd.Series([v["duration"] for _, v in pipelines_info.items()])
    logging.info(f"Duration information for {N} jobs in {len(pipelines_info)} pipeline(s)\n")
    duration_info = OrderedDict({"pipeline": pipelines_duration, "job": job_durations, "scripts": script_durations})

    duration_stats = OrderedDict()
    duration_stats["Duration (s)"] = list(duration_info.keys())
    duration_stats["total"] = [round(v.sum(), digits) for k, v in duration_info.items()]
    duration_stats["mean"] = [round(v.mean(), digits) for k, v in duration_info.items()]
    duration_stats["median"] = [round(v.median(), digits) for k, v in duration_info.items()]
    duration_stats["min"] = [round(v.min(), digits) for k, v in duration_info.items()]
    duration_stats["max"] = [round(v.max(), digits) for k, v in duration_info.items()]
    table = tabulate(duration_stats, headers=list(duration_stats.keys()), tablefmt="psql")
    print(table)


def query_jet_jobs(
    jet_instance: JETInstance,
    jet_workloads_ref: Optional[str] = None,
    job_type: Optional[str] = None,
    pipeline_id: Optional[int] = None,
    job_id: Optional[int] = None,
    pipeline_type: Optional[str] = None,
    duration: Optional[str] = None,
    limit: Optional[int] = 1000,
    only_completed: bool = False,
    fields_select: Optional[List[str]] = None,
    label: Optional[str] = "bionemo",
) -> List[dict]:
    """
    Queries Kibana to get results for JET jobs according to the specifications.
    """

    log_service = jet_instance.log_service()
    query = JETLogsQuery()
    if label is not None:
        query = query.filter(Field("obj_workload.obj_labels.origin") == label)

    if pipeline_type is not None:
        logging.info(f"Query results for pipeline_type: {pipeline_type}")
        query = query.filter(Field("obj_workload.obj_labels.workload_ref") == f"bionemo/{pipeline_type}")

    if jet_workloads_ref is not None:
        logging.info(f"Query results for jet_workloads_ref: {jet_workloads_ref}")
        query = query.filter(Field("obj_workloads_registry.s_commit_ref") == jet_workloads_ref)

    if pipeline_id is not None:
        logging.info(f"Query results for Jet CI pipeline id: {pipeline_id}")
        query = query.filter(Field("obj_ci.l_pipeline_id") == pipeline_id)

    if job_id is not None:
        logging.info(f"Query results for Jet CI job id: {job_id}")
        query = query.filter(Field("obj_ci.l_job_id") == job_id)

    if job_type is not None:
        logging.info(f"Query results for Jet CI job type: {job_type}")
        query = query.filter(Field("s_type") == job_type)

    if duration is not None:
        if any(duration.endswith(s) for s in ["d", "w", "M", "y"]):
            query = query.filter(Field("ts_created") >= f"now-{duration}")
        else:
            try:
                datetime.date.fromisoformat(duration)
                query = query.filter(Field("ts_created") >= duration)

            except ValueError:
                logging.error(f"Invalid duration string: {str}. Proceeding without filtering results by date")

    if only_completed:
        query = query.filter(Field("l_exit_code") == 0.0)

    # TODO(dorotat): if filters are used, the query outputs only 10 top results so set high limit to get all results
    if limit is not None:
        query = query.limit(limit)

    if fields_select is not None:
        query = query.select(*fields_select)

    # Getting results for all jobs in the query
    jobs_results = log_service.query(query)
    return jobs_results


def log_detailed_job_info(df: pd.DataFrame) -> None:
    """
    Logs to the console detailed summary of JET test execution such as job id and the status of the job,
    its duration as well as script to execute if the job failed
    """
    logging.info("Additional information about jobs in JET listed below")

    for _, job_info in df.iterrows():
        error_line = (
            f"\nJET job error code: {job_info['jet_ci_job_status_code']}, JET job error msg: "
            f"{job_info['jet_ci_job_status']}"
            if job_info["jet_ci_job_status"].lower() != "success"
            else ""
        )
        jet_job_status = "success" if job_info["jet_ci_job_status"].lower() == "success" else "failed"
        if job_info["job_type"] == "build":
            header_str = f'DOCKER BUILD {job_info["job_key"]}\nJOB status: {jet_job_status}'
            color = AnsiCodes.RED if job_info["jet_ci_job_status"].lower() == "failed" else AnsiCodes.GREEN
            msg = f"{_apply_color_str(text=header_str, color=color)}"
            msg += f"{error_line}"
            for k, v in job_info[job_info.index.str.startswith("docker_")].items():
                msg += f"\nDocker info: \n{k}: {v}\n"

        elif job_info["job_type"] == "recipe":
            if "conv" in job_info["job_key"]:
                prefix = "CONVERGENCE TEST: "
            elif "perf" in job_info["job_key"]:
                prefix = "PERFORMANCE TEST: "
            else:
                prefix = "SMOKE TEST: "
            color = AnsiCodes.RED if job_info["jet_test_status"].lower() == "failed" else AnsiCodes.GREEN
            header_str = (
                f'{prefix}{job_info["job_key"]}\n{AnsiCodes.BOLD}'
                f'TEST status: {job_info["jet_test_status"].lower()}{AnsiCodes.BOLD}, JOB status: {jet_job_status}'
            )
            test_job_log = (
                f'\nJET test job id log: https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/{job_info["jet_ci_job_id_test"]}'
                if job_info["jet_ci_job_id_test"] != "nan"
                else ""
            )
            msg = (
                f'{_apply_color_str(text=header_str, color=color)}'
                f'\nTEST: {job_info["jest_test_check"]}'
                f'{error_line}\n'
                f'\nJET pipeline id: {job_info["jet_ci_pipeline_id"]}, BioNeMo pipeline id: {job_info["bionemo_ci_pipeline_id"]}'
                f'JET job id: {job_info["jet_ci_job_id"]}, JET test job id: {job_info["jet_ci_job_id_test"] if job_info["jet_ci_job_id_test"] else "N/A"}'
                f'\nJET job id log: https://gitlab-master.nvidia.com/dl/jet/ci/-/jobs/{job_info["jet_ci_job_id"]}'
                f'{test_job_log}'
                f'\nscript duration in sec: {round(job_info["script_duration"], 2)}, job duration in sec (SLURM queue + JET setup + script duration): {round(job_info["jet_ci_job_duration"], 2)}'
            )

            for k, v in job_info[job_info.index.str.startswith("log_")].items():
                msg += f'\n{k.replace("log_", "")}: {v}'

            if job_info["jet_ci_job_status"].lower() != "success":
                msg += f"\n\nScript:\n {job_info['script']}\n\n"
        logging.info(msg + "\n")


def get_docker_info(docker_info: dict) -> dict:
    """
    Parse information about the docker container from JET jobs results
    """
    new_docker_info = {}
    for k, v in docker_info.items():
        name = f"docker_{k}"
        new_docker_info[name] = v
    return new_docker_info


def get_job_logs_info(job_logs: dict) -> dict:
    """
    Parse information about available logs per JET job
    """
    new_job_info = {}
    for asset in job_logs:
        if asset["s_name"] == "dllogger.json":
            name = "log_dllogger"
        elif asset["s_name"] == "output_script-0.log":
            name = "log_output_script"
        elif asset["s_name"] == "trainer_logs.json":
            name = "log_trainer_logs"
        elif asset["s_name"] == "error_msg.txt":
            name = "log_error_msg"
        else:
            continue
        new_job_info[name] = asset["s_url"]
    return new_job_info


def get_job_info(job_raw_result: dict, save_dir: Optional[str]) -> dict:
    """
    Parse information about JET job from results obtained from JET
    """
    job_info = {
        "script_duration": job_raw_result.get("d_duration", None),
        "timestamp": job_raw_result.get("ts_created", None),
    }
    job_info["date"] = (
        datetime.datetime.fromtimestamp(job_info["timestamp"] / 1000).strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        if job_info["timestamp"]
        else None
    )

    ci_info = job_raw_result.get("obj_ci", {})
    workloads_info = job_raw_result.get("obj_workloads_registry", {})
    obj_workload = job_raw_result.get("obj_workload", {})
    obj_status = job_raw_result["obj_status"]

    pipeline_id = str(ci_info.get("l_pipeline_id"))

    job_info["jet_workloads_ref"] = workloads_info.get("s_commit_ref", None)
    job_info["jet_ci_pipeline_id"] = pipeline_id
    job_info["jet_ci_job_id"] = ci_info.get("l_job_id")
    job_info["jet_ci_job_duration"] = ci_info.get("d_job_duration", None)
    job_info["jet_ci_job_name"] = ci_info.get("s_job_name")
    job_info["jet_ci_job_status"] = obj_status.get("s_message", None)
    job_info["jet_ci_job_status_code"] = obj_status.get("s_code", None)
    # Workload id is needed for JET test
    job_info["jet_ci_workload_id"] = job_raw_result.get("s_id", None)

    job_info["user"] = job_raw_result["s_user"]
    job_info["job_key"] = obj_workload.get("s_key", None)
    job_info["job_type"] = obj_workload.get("s_type", None)
    job_info["exit_code"] = (
        job_raw_result.get("l_exit_code", None)
        if job_info["job_type"] == "recipe"
        else int(job_raw_result.get("b_invalid", 1))
    )
    if "recipe" in job_info["job_key"]:
        job_info["script"] = obj_workload["obj_spec"].get("s_script", None)
        if "conv" in job_info["job_key"]:
            job_info["wandb_project_name"] = obj_workload["obj_spec"].get("s_wandb_project_name", None)

    if job_info["job_type"] == "build":
        docker_info = get_docker_info(docker_info=obj_workload["obj_spec"]["obj_source"])
        job_info.update(docker_info)

    elif job_info["job_type"] == "recipe":
        jobs_logs_info = get_job_logs_info(job_logs=job_raw_result.get("nested_assets", []))
        job_info.update(jobs_logs_info)

        if save_dir is not None:
            job_info["env_info_gpu"] = job_raw_result.get("nested_gpu", None)
            job_info["env_info_cpu"] = job_raw_result.get("obj_cpu", None)
    else:
        raise ValueError("Only jobs of type 'recipe' or 'build' are supported.")

    return job_info


def get_job_results(
    jet_instance: JETInstance,
    jet_workloads_ref: Optional[str] = None,
    pipeline_id: Optional[int] = None,
    job_id: Optional[int] = None,
    pipeline_type: Optional[str] = None,
    duration: Optional[str] = None,
    limit: Optional[int] = 1000,
    only_completed: bool = False,
    save_dir: Optional[str] = None,
    all_jobs: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """
    Queries and outputs results of jet jobs run in JET given related pipeline id in JET CI, job id or pipeline type.
    """
    results_jobs = query_jet_jobs(
        jet_instance=jet_instance,
        jet_workloads_ref=jet_workloads_ref,
        pipeline_id=pipeline_id,
        job_id=job_id,
        pipeline_type=pipeline_type,
        duration=duration,
        limit=limit,
        only_completed=only_completed,
    )
    if len(results_jobs) == 0:
        raise ValueError("No entries matched the requested query in the JET logs.")

    logging.info(f"Getting {len(results_jobs)} jobs from Kibana... \n")
    output = []

    jet_ci = jet_instance.gitlab_ci()
    pipelines_info = {}
    for result in tqdm.tqdm(results_jobs, desc="Getting results for JET tests "):
        job_info = get_job_info(job_raw_result=result, save_dir=save_dir)

        pipeline_id = job_info["jet_ci_pipeline_id"]
        if job_info["jet_ci_pipeline_id"] not in pipelines_info:
            obj_workload = result.get("obj_workload", {})
            pipelines_info[pipeline_id] = obj_workload["obj_labels"]
            pipelines_info[pipeline_id]["duration"] = jet_ci.project.pipelines.get(pipeline_id).duration
        job_info["jet_ci_pipeline_duration"] = pipelines_info[pipeline_id]["duration"]
        job_info["bionemo_ci_pipeline_id"] = pipelines_info[pipeline_id].get("bionemo_ci_pipeline_id", None)
        job_info["pipeline_type"] = pipelines_info[pipeline_id].get("workload_ref", None)

        output.append(job_info)

    df = pd.DataFrame(output)

    df.sort_values(by=["jet_ci_pipeline_id", "job_key", "timestamp"], inplace=True)

    # Keeping only the most recent jobs per pipeline and job_key (to filter out jobs that were rerun)
    if not all_jobs:
        df = filter_out_failed_and_rerun_jobs(df=df, n_all_results=df.shape[0])
    return df, pipelines_info


def _get_tests_check_str(check_dict: Dict[str, str]) -> str:
    """
    Converts standard dict with JET test checks information to string
    Args:
        check_dict: dict with keys s_name, s_status and, in case a check failure, s_status_message
    Returns: str

    """
    check_str = f"{check_dict['s_name']}: {check_dict['s_status']}"

    # Append the status message if it is present in the dictionary
    if "s_status_message" in check_dict:
        check_str += f" ({check_dict['s_status_message']})"
    return f"[{check_str}]"


def get_test_results(jet_instance: JETInstance, pipeline_ids: List[str]) -> Optional[pd.DataFrame]:
    """
    Queries and outputs results of jet test jobs run in JET given related pipeline id.
    """
    results_tests = []
    for pipeline_id in pipeline_ids:
        tests_raw = query_jet_jobs(
            jet_instance=jet_instance,
            pipeline_id=int(pipeline_id),
            fields_select=["s_status", "obj_ci.l_pipeline_id", "obj_ci.l_job_id", "s_workload_logs", "nested_checks"],
            job_type="test",
            label=None,
        )
        results_tests.extend(tests_raw)

    if len(results_tests) == 0:
        logging.warning("No tests results found.")
        return None

    output = []
    for test_raw in results_tests:
        test_dict = {}
        if "s_workload_logs" not in test_raw:
            continue
        test_dict["jet_ci_workload_id"] = test_raw["s_workload_logs"][0]
        test_dict["jet_test_status"] = "success" if test_raw["s_status"] == "pass" else "failed"
        test_dict["jest_test_check"] = " AND ".join([_get_tests_check_str(d) for d in test_raw["nested_checks"]])
        test_dict["jet_ci_job_id_test"] = (
            str(test_raw["obj_ci"]["l_job_id"]) if test_raw["obj_ci"].get("l_job_id", None) else None
        )
        output.append(test_dict)
    return pd.DataFrame(output)


def log_wandb_project_links(project_names: List[str]):
    print(project_names)
    logging.info(
        f'Training curves available at {", ".join(["https://wandb.ai/clara-discovery/"+name for name in project_names if isinstance(name, str)])}\n'
    )


def get_results_from_jet(
    jet_workloads_ref: Optional[str] = None,
    pipeline_id: Optional[int] = None,
    job_id: Optional[int] = None,
    pipeline_type: Optional[str] = None,
    duration: Optional[str] = None,
    limit: Optional[int] = 1000,
    only_completed: bool = False,
    save_dir: Optional[str] = None,
    all_jobs: bool = False,
    verbosity_level: int = 0,
    return_df: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Queries and outputs results (print or saved as csv) of tests run in JET given reference (branch)
    from JET Workloads Registry, its pattern, or related pipeline id in JET CI.
    If no query arguments are provided, all results run on bionemo account are returned.

    By default, the basic info about the tests, their logs and related jobs in JET is printed to the console
    (detailed information about exit codes categories, see https://jet.nvidia.com/docs/logs/status/).
    If save_dir is provided, more detailed information is saved as csv

    Requires installed JET API https://gitlab-master.nvidia.com/dl/jet/api. Can be executed outside docker container.

    Usage:

    # Get results for all tests run for BioNeMo account (main tests and the ones run on ephemeral/bionemo)
    python internal/jet/get_results_from_jet.py

    # Get reference-related results
    python internal/jet/get_results_from_jet.py --jet_workloads_ref bionemo/training-unit-tests

    # Get pipeline id-related results
    python internal/jet/get_results_from_jet.py --pipeline_id 9469458

    # Attach to your command --pipeline_type dev to get logs for all JET test pipelines which were execute for branch "dev"
    python internal/jet/get_results_from_jet.py ..... --pipeline_type dev

    # Or attach to your command --save_dir <FOLDER_PATH> to save details of the run tests to a csv with predefined path
    python internal/jet/get_results_from_jet.py ..... --save_dir <FOLDER_PATH>

    # Attach to your command -v or -vv to have more detailed levels of verbosity of logs
    python internal/jet/get_results_from_jet.py ..... -vv

    Args:
        jet_workloads_ref: a reference (branch) in JET Workloads Registry, optional
        pipeline_id: a pipeline id in JET CI, optional
        job_id: a job id in JET CI, optional,
        pipeline_type: specifies type of BioNeMo's CI pipeline(s) to get jet logs for, optional.
                       Can be either "merge_request_event" or name of the git branch in BioNeMo repo, ie "dev"
        duration: specifies period in the past to include jobs from up to now, optional. The accepted formats are
                  either dates from which to include as ISO-8601-formatted datetime string (ie '2023-01-01T15:00','2023-01-01')
                  or durations, ie 1d, 5d, 1w, 2w, 1M, 2M, 1y, 2y
        limit: a limit of results to extract, default set to 1000,
        only_completed: a flag that determines whether to keep only successfully completed jobs (exit status != 0)
        save_dir: directory to save csv with results, optional
        all_jobs: a flag that determines whether to take all jobs in a pipeline or keep only the most recent jobs for each job key.
                     This is useful when individual jobs fail due to infrastructure issues and are rerun within
                     the same JET pipeline.
        verbosity_level: a verbosity level of logs
        return_df: a flag that determines whether to return pd.DataFrame with job details
    """
    jet_instance = JETInstance(env="prod")

    df, pipelines_info = get_job_results(
        jet_instance=jet_instance,
        jet_workloads_ref=jet_workloads_ref,
        pipeline_id=pipeline_id,
        job_id=job_id,
        pipeline_type=pipeline_type,
        duration=duration,
        limit=limit,
        only_completed=only_completed,
        save_dir=save_dir,
        all_jobs=all_jobs,
    )

    df_test = get_test_results(jet_instance=jet_instance, pipeline_ids=list(pipelines_info.keys()))
    if df_test.shape[0] > 0:
        df = pd.merge(df, df_test, on="jet_ci_workload_id", how="left")
    df.loc[(df["job_key"].str.contains("recipe")) & (df["jet_test_status"].isna()), "jet_test_status"] = "failed"
    if verbosity_level >= 2:
        print_table_with_results(df=df)

    if verbosity_level >= 2 and "wandb_project_name" in df:
        log_wandb_project_links(project_names=df["wandb_project_name"].unique())

    # Calculating duration-related analytics (overall, pipeline and job specific)
    if verbosity_level > 0 and df.shape[0] > 2:
        get_duration_stats(
            pipelines_info=pipelines_info, job_durations=df.jet_ci_job_duration, script_durations=df.script_duration
        )

    if verbosity_level > 2:
        log_detailed_job_info(df=df)

    if save_dir is not None:
        filename = f'jet_query_{"_".join(pipelines_info.keys())}.json'
        filepath = os.path.join(save_dir, filename)
        logging.info(f"Saving query results to: {filepath}")
        df.to_json(filepath, orient="records")
    if return_df:
        return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--jet_workloads_ref", type=str, default=None, help="Reference (branch) in JET Workloads Registry, optional"
    )
    parser.add_argument("--pipeline_id", type=int, default=None, help="Pipeline ID in JET CI, optional")
    parser.add_argument("--job_id", type=int, default=None, help="Job ID in JET CI, optional")
    parser.add_argument("--d", type=str, default=None, help="Specifies period in the past to include jobs from")
    parser.add_argument(
        "--pipeline_type",
        type=str,
        default=None,
        help="Specifies type of BioNeMo's CI pipeline(s) to get jet logs for, optional. Can be either 'merge_request' "
        "or name of the git branch in BioNeMo repo, ie 'dev'",
    )
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save csv with results, optional")
    parser.add_argument(
        "--return_df",
        action="store_true",
        help="A flag that determines whether to return pd.DataFrame with job results",
    )
    parser.add_argument(
        "--only_completed",
        action="store_true",
        help="A flag that determines whether to keep only " "successfully completed jobs (exit status != 0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="A flag that determines whether to take all jobs in the pipeline or keep only the most recent jobs for each job key. "
        "Useful when some jobs have been rerun due to infrastructure issues.",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of printed results")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="a flag that determines verbosity level")

    args = parser.parse_args()
    get_results_from_jet(
        jet_workloads_ref=args.jet_workloads_ref,
        pipeline_id=args.pipeline_id,
        job_id=args.job_id,
        pipeline_type=args.pipeline_type,
        duration=args.d,
        save_dir=args.save_dir,
        only_completed=args.only_completed,
        limit=args.limit,
        all_jobs=args.all,
        verbosity_level=args.verbose,
        return_df=args.return_df,
    )
