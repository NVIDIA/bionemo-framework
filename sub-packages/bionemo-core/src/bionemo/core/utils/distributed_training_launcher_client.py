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


"""Client for Training Launcher Server - Use this in Jupyter notebooks."""

import json
import socket
import time
from typing import Any, Dict, List


class DistributedTrainingLauncherClient:
    """Distributed Training Launcher Client."""

    def __init__(self, host="localhost", port=6789):
        """Initialize the Distributed Training Launcher Client."""
        self.host = host
        self.port = port

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the launcher server."""
        try:
            # Create socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)  # 30 second timeout
            sock.connect((self.host, self.port))

            # Send request
            message = json.dumps(request)
            sock.send(message.encode("utf-8"))

            # Receive response
            response = sock.recv(4096).decode("utf-8")
            sock.close()

            return json.loads(response)

        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}

    def launch_training(
        self,
        command: List[str],
        job_name: str | None = None,
        working_dir: str | None = None,
        env_vars: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """Launch a training process.

        Args:
            command: Command and arguments to execute (e.g., ['train_evo2', '--config', 'config.yaml'])
            job_name: Optional name for the job
            working_dir: Working directory for the process
            env_vars: Environment variables to set

        Returns:
            Response dict with process_id, job_name, log_file, etc.
        """
        request = {
            "action": "launch",
            "command": command,
            "job_name": job_name,
            "working_dir": working_dir,
            "env": env_vars or {},
        }

        return self._send_request(request)

    def get_status(self, process_id: int | None = None) -> Dict[str, Any]:
        """Get status of a process or all processes.

        Args:
            process_id: Specific process ID, or None for all processes
        """
        request = {"action": "status"}
        if process_id is not None:
            request["process_id"] = process_id

        return self._send_request(request)

    def kill_process(self, process_id: int) -> Dict[str, Any]:
        """Kill a specific process."""
        request = {"action": "kill", "process_id": process_id}
        return self._send_request(request)

    def list_processes(self) -> Dict[str, Any]:
        """List all processes."""
        request = {"action": "list"}
        return self._send_request(request)

    def get_logs(self, process_id: int, lines: int = 50) -> Dict[str, Any]:
        """Get logs for a process.

        Args:
            process_id: Process ID
            lines: Number of lines to retrieve from end of log
        """
        request = {"action": "logs", "process_id": process_id, "lines": lines}
        return self._send_request(request)

    def wait_for_completion(self, process_id: int, timeout: int = 1500, check_interval: int = 30) -> Dict[str, Any]:
        """Wait for a process to complete, with periodic status updates.

        Args:
            process_id: Process ID to wait for
            timeout: Maximum time to wait for process to complete (seconds)
            check_interval: How often to check status (seconds)
        """
        print(f"Waiting for process {process_id} to complete...")

        start_time = time.time()
        while True:
            status = self.get_status(process_id)

            if status["status"] != "success":
                return status

            process_info = status["process_info"]

            if not process_info["is_running"]:
                print(f"Process {process_id} completed!")
                return status

            print(f"Process {process_id} still running... (checked at {time.strftime('%H:%M:%S')})")
            time.sleep(check_interval)

            if time.time() - start_time > timeout:
                print(f"Process {process_id} timed out after {timeout} seconds")
                return {"status": "timeout", "message": f"Process {process_id} timed out after {timeout} seconds"}


def show_all_jobs():
    """Show all running/completed jobs."""
    client = DistributedTrainingLauncherClient()
    response = client.list_processes()

    if response["status"] == "success":
        processes = response["processes"]
        if not processes:
            print("No processes found")
            return

        print(f"{'ID':<4} {'Job Name':<20} {'Status':<10} {'Start Time':<20} {'Command'}")
        print("-" * 80)

        for proc in processes:
            status = "Running" if proc["is_running"] else "Completed"
            cmd_str = " ".join(proc["cmd"][:3]) + ("..." if len(proc["cmd"]) > 3 else "")
            print(
                f"{proc['process_id']:<4} {proc['job_name']:<20} {status:<10} {proc['start_time'][:19]:<20} {cmd_str}"
            )
    else:
        print(f"Error: {response['message']}")


def kill_job(process_id: int):
    """Kill a specific job."""
    client = DistributedTrainingLauncherClient()
    response = client.kill_process(process_id)
    print(response["message"])
