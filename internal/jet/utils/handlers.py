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
import subprocess
import time
from typing import Optional


__all__ = ["JetGitRepoHandler", "JetPipelineHandler"]


def wait_before_execute_cmd(cmd: str, nsec: int = 5):
    """
    Forces a waiting period before executing a command
    Args:
        cmd: the string of a command
        nsec: waiting period in number of sec

    """
    print(f"The script will execute the following command within {nsec}s. Please, hit CTRL + C to interrupt...")
    print(f"\033[1;32m{cmd}\033[0m")
    time.sleep(nsec)


class JetGitRepoHandler:
    """
    The helper object to handle creating and pushing ephemeral branches to JET Workloads Registry.
    The handler
    1. clones the repository to a given location (user specified, or the parent dir of the cwd),
    2. switches to the default branch in the repository with the configuration of the test that are run in JET
    3. creates ephemeral branch of the test to be run in JET
    The ephemeral branches in the repository are delete periodically.
    """

    def __init__(
        self,
        jet_workloads_ref_default: str,
        jet_workloads_repo_path: Optional[str] = None,
        jet_workloads_ref_ephemeral: Optional[str] = None,
    ):
        """
        Args:
            jet_workloads_ref_default: the default reference to access basic configuration of tests in
                                        JET Workloads Registry
            jet_workloads_repo_path: optional, the path to the workloads-registry or its parent directory
            jet_workloads_ref_ephemeral: optional, the name of the temporary branch in JET Workloads Registry to run
                                        the modified tests in JET. The agreed prefix of the branch's name is "ephemeral/bionemo"
        """
        self.jet_workloads_ref_default = jet_workloads_ref_default
        self.jet_workloads_ref_ephemeral = jet_workloads_ref_ephemeral
        self.prefix_ref_ephemeral = "ephemeral/bionemo"
        self.jet_workloads_repo_dir = "workloads-registry"
        self.jet_workloads_repo_url = "https://gitlab-master.nvidia.com/dl/jet/workloads-registry"
        self._set_jet_workloads_repo_path(jet_workloads_repo_path)

    def _set_jet_workloads_repo_path(self, jet_workloads_repo_path: Optional[str] = None):
        """
        Sets the path to the local repository of JET Workloads Registry
        Args:
            jet_workloads_repo_path: optional, the path to the workloads-registry or its parent directory
        """
        if jet_workloads_repo_path is None:
            path = os.getcwd()
            parent_dir = os.path.dirname(path)
            self.jet_workloads_repo_path = os.path.join(parent_dir, self.jet_workloads_repo_dir)
        else:
            if jet_workloads_repo_path.endswith(self.jet_workloads_repo_dir):
                self.jet_workloads_repo_path = jet_workloads_repo_path
            else:
                self.jet_workloads_repo_path = os.path.join(jet_workloads_repo_path, self.jet_workloads_repo_dir)

    def setup_local_jet_workloads_repo(self):
        """
        Checks if the local repository of JET Workloads Registry is present under specified location and
        clones it to the predefined location.
        """
        if not os.path.exists(self.jet_workloads_repo_path):
            os.makedirs(self.jet_workloads_repo_path, exist_ok=True)
            self._clone_jet_workloads_repo()

    def _clone_jet_workloads_repo(self):
        print(f"Cloning git JET Workloads Registry to {self.jet_workloads_repo_path}")
        cmd = f"git clone {self.jet_workloads_repo_url} {self.jet_workloads_repo_path}"
        wait_before_execute_cmd(cmd=cmd)
        subprocess.check_call(cmd, shell=True)
        assert os.path.exists(self.jet_workloads_repo_path), (
            f"JET Workloads Registry path "
            f"{self.jet_workloads_repo_path} does not exists after "
            f"cloning the repository: {self.jet_workloads_repo_url} "
        )

    def setup_local_ephemeral_branch(self):
        """
        Helper method that sets the ephemeral branch in the local JET Workloads Registry with configuration of
        tests to be run in JET.
        """
        if self.jet_workloads_ref_ephemeral is None:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_T%H_%M_%S_%f")
            self.jet_workloads_ref_ephemeral = f"{self.prefix_ref_ephemeral}/{timestamp}"

        if not self.jet_workloads_ref_ephemeral.startswith(self.prefix_ref_ephemeral):
            self.jet_workloads_ref_ephemeral = f"{self.prefix_ref_ephemeral}/{self.jet_workloads_ref_ephemeral}"
        cmd = (
            f"git checkout -f {self.jet_workloads_ref_default} \\\n"
            f"\t&& git checkout -b {self.jet_workloads_ref_ephemeral}"
        )
        wait_before_execute_cmd(cmd=cmd)
        subprocess.check_call(cmd, shell=True, cwd=self.jet_workloads_repo_path)

    def push_changes_to_remote_ephemeral_branch(self, dry_run: bool = False):
        """
        Optionally sets and pushes the local ephemeral branch to the remote JET Workloads Registry
        with configuration of tests to be run in JET.
        Args:
            dry_run: if false, the ephemeral branch is not pushed to the remote repository

        """
        if self.jet_workloads_ref_ephemeral is None or not self.jet_workloads_ref_ephemeral.startswith(
            self.prefix_ref_ephemeral
        ):
            self.setup_local_ephemeral_branch()
        cmd = (
            f"git add . && git commit -m'Add temporary bionemo workloads' && "
            f"git push --set-upstream origin {self.jet_workloads_ref_ephemeral}"
        )
        wait_before_execute_cmd(cmd=cmd)
        if not dry_run:
            subprocess.check_call(cmd, shell=True, cwd=self.jet_workloads_repo_path)


class JetPipelineHandler:
    """
    Runs a pipeline in JET CI for tests specified in JET Workloads Registry as well as gets info about the workloads

    JET Workloads Registry: https://gitlab-master.nvidia.com/dl/jet/workloads-registry
    List of pipelines run JET CI: https://gitlab-master.nvidia.com/dl/jet/ci/-/pipelines
    """

    def __init__(self, jet_workloads_ref: str):
        """
        jet_workloads_ref: reference to branch in JET Workloads Registry o run pipeline for
        """
        self.jet_workloads_ref = jet_workloads_ref

    def run_jet_pipeline(self, jet_filter: Optional[str] = "\"type == 'recipe'\"", dry_run: bool = False):
        """
        Runs jet pipeline for reference
        Args:
            jet_filter: the pandas format of the query to run only a subset of jobs in the workload
            dry_run: if true, the dry-run mode of JET API is enabled and the pipeline is not sent to the JET CI

        """
        cmd = f"jet ci run -w {self.jet_workloads_ref} -f {jet_filter}"
        if dry_run:
            cmd += " --dry-run"
        wait_before_execute_cmd(cmd=cmd)
        subprocess.check_call(cmd, shell=True)

    @staticmethod
    def setup_jet_api():
        """
        Sets up JET APi for the first time users

        """
        cmd = "jet secrets login;jet secrets pull"

        wait_before_execute_cmd(cmd=cmd)
        subprocess.check_call(cmd, shell=True)

    def get_workload_info(self, jet_filter: Optional[str] = "\"type == 'recipe'\"", dry_run: bool = False):
        """
        Gets information about a workloads given its reference in JET Workloads Registry
        Args:
            jet_filter: the pandas format of the query to filter by JET API the info about the workload.
                    More info about the syntax in https://jet.nvidia.com/docs/workloads/filtering/
            dry_run: if true, the try-catch prints the error msg but the method is executed

        Returns:

        """
        cmd = f"jet workloads --registry-ref {self.jet_workloads_ref} list"
        if jet_filter is not None:
            cmd += f" -f {jet_filter}"

        wait_before_execute_cmd(cmd=cmd)
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            if dry_run:
                logging.warning("Expected error msg: gitlab.exceptions.GitlabGetError: 404: 404 Commit Not Found")
            else:
                raise e
