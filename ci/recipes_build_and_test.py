#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CI Build and Test Logic for BioNeMo Recipes.

This script implements automated CI logic for building, testing, and pushing Docker containers
for BioNeMo recipe and model directories. It follows the logic described in ci_logic.md.

## What it does:

1. **Discovery**: Finds all subdirectories in `models/` and `recipes/` that contain Dockerfiles
2. **Analysis**: For each directory, checks git status and docker registry to determine if build is needed
3. **Build Report**: Shows a structured table of what will be built and why
4. **Parallel Building**: Builds required Docker images in parallel with progress tracking
5. **Serial Testing**: Runs pytest tests in each newly built container (with streaming output)
6. **Parallel Push**: Pushes successful, non-dirty containers to remote registry in parallel

## Prerequisites:

**GitLab Authentication**: You must have a GitLab user access token in your `~/.netrc` file:

```
machine gitlab-master.nvidia.com
  login your_username
  password your_gitlab_token
```

This is required for Docker builds to authenticate with GitLab until nvFSDP becomes publicly available.

## Usage:

```bash
# Full CI pipeline (build, test, push)
python ci/build_and_test.py

# Build and push only (skip tests)
python ci/build_and_test.py --skip-tests

# Build and test only (skip push)
python ci/build_and_test.py --skip-push

# Build only (skip both tests and push)
python ci/build_and_test.py --skip-tests --skip-push
```

## Container Naming:

Containers are tagged as: `{REGISTRY}/{parent_dir}/{dir_name}:{commit_sha}[-dirty]`

Example: `gitlab-master.nvidia.com:5005/clara-discovery/bionemo-recipes/models/esm2:abc12345`

## Exit Codes:

- 0: All operations succeeded
- 1: One or more builds, tests, or pushes failed

## Dependencies:

- Docker (with buildkit support for secrets)
- Git
- Python packages: rich
"""

import argparse
import asyncio
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


CONTAINER_REGISTRY = "gitlab-master.nvidia.com:5005/clara-discovery/bionemo-recipes"
DOCKER_BUILD_ARGS = []
DOCKER_RUN_ARGS = [
    "--rm",
    "-it",
    "--gpus",
    "all",
    "--ipc=host",
    "--ulimit",
    "memlock=-1",
    "--ulimit",
    "stack=67108864",
]


@dataclass
class BuildInfo:
    """Information about a directory that needs to be built/tested."""

    directory: Path
    commit_sha: str
    has_uncommitted_changes: bool
    image_exists_in_registry: bool
    container_tag: str
    needs_build: bool


class CIManager:
    """Manages the CI build and test process."""

    def __init__(self, console: Console, skip_tests: bool = False, skip_push: bool = False):
        """Initialize the CI manager."""
        self.console = console

        # Check for necessary versions of docker and git
        docker_version = subprocess.check_output(["docker", "--version"], text=True).strip()
        git_version = subprocess.check_output(["git", "--version"], text=True).strip()
        if docker_version < "24.0.0":
            raise ValueError("Docker version 24.0.0 or higher is required to run this script.")
        if git_version < "2.40.0":
            raise ValueError("Git version 2.40.0 or higher is required to run this script.")

        # Ensure docker run with GPUs is supported
        num_gpus = subprocess.check_output(
            [
                "docker",
                "run",
                "--gpus",
                "all",
                "ubuntu",
                "/bin/bash",
                "-c",
                "nvidia-smi -L | wc -l",
            ],
            text=True,
        ).strip()
        if int(num_gpus) == 0:
            raise ValueError("Docker run with GPUs is not supported, or you do not have GPUs available.")

        self.console.print(f"[bold green]Docker version: {docker_version}[/bold green]")
        self.console.print(f"[bold green]Git version: {git_version}[/bold green]")
        self.console.print(f"[bold green]Number of GPUs: {num_gpus}[/bold green]")

        # Use git to determine the workspace root
        git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        self.workspace_root = Path(git_root)
        self.build_infos: List[BuildInfo] = []
        self.skip_tests = skip_tests
        self.skip_push = skip_push

    def discover_build_directories(self) -> List[Path]:
        """Discover all subdirectories in models/ and recipes/ that have Dockerfiles."""
        build_dirs = []

        for parent_dir in ["models", "recipes"]:
            parent_path = self.workspace_root / parent_dir
            if not parent_path.exists():
                continue

            for subdir in parent_path.iterdir():
                if subdir.name.startswith("."):
                    continue
                if subdir.is_dir() and (subdir / "Dockerfile").exists():
                    build_dirs.append(subdir)
                elif subdir.is_dir():
                    self.console.print(f"[red]Skipping {subdir} because it doesn't have a Dockerfile.[/red]")

        return build_dirs

    async def analyze_directory_status(self, directory: Path) -> BuildInfo:
        """Analyze git status and docker registry status for a directory."""
        # Get the relative path for the directory
        rel_path = directory.relative_to(self.workspace_root)

        # Check for uncommitted changes in this directory
        has_uncommitted_changes = await self._has_uncommitted_changes(directory)

        # Get the last commit SHA for this directory
        commit_sha = await self._get_last_commit_sha(directory)

        # Generate container tag
        container_name = f"{CONTAINER_REGISTRY}/{rel_path.parent.name}/{rel_path.name}"
        container_tag = f"{container_name}:{commit_sha}{'-dirty' if has_uncommitted_changes else ''}"

        # Check if image exists in registry
        image_exists = await self._check_image_in_registry(container_tag)

        # Determine if build is needed
        needs_build = has_uncommitted_changes or not image_exists

        return BuildInfo(
            directory=directory,
            commit_sha=commit_sha,
            has_uncommitted_changes=has_uncommitted_changes,
            image_exists_in_registry=image_exists,
            container_tag=container_tag,
            needs_build=needs_build,
        )

    async def _has_uncommitted_changes(self, directory: Path) -> bool:
        """Check if directory has uncommitted changes."""
        try:
            # Check for unstaged changes
            result = await self._run_git_command(
                [
                    "diff",
                    "--quiet",
                    "--",
                    str(directory.relative_to(self.workspace_root)),
                ]
            )
            has_unstaged = result.returncode != 0

            # Check for staged changes
            result = await self._run_git_command(
                [
                    "diff",
                    "--cached",
                    "--quiet",
                    "--",
                    str(directory.relative_to(self.workspace_root)),
                ]
            )
            has_staged = result.returncode != 0

            return has_unstaged or has_staged
        except Exception as e:
            self.console.print(f"[red]Error checking uncommitted changes for {directory}: {e}[/red]")
            return True  # Conservative approach

    async def _get_last_commit_sha(self, directory: Path) -> str:
        """Get the last commit SHA for a directory."""
        try:
            result = await self._run_git_command(
                [
                    "log",
                    "-1",
                    "--format=%H",
                    "--",
                    str(directory.relative_to(self.workspace_root)),
                ]
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()[:8]  # Use short SHA
            else:
                # Fallback to current HEAD
                result = await self._run_git_command(["rev-parse", "--short", "HEAD"])
                return result.stdout.strip()
        except Exception as e:
            self.console.print(f"[red]Error getting commit SHA for {directory}: {e}[/red]")
            return "unknown"

    async def _check_image_in_registry(self, container_tag: str) -> bool:
        """Check if image exists in remote registry using docker manifest inspect."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                process = await asyncio.create_subprocess_exec(
                    "docker",
                    "manifest",
                    "inspect",
                    container_tag,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    text=False,
                )
                stdout, stderr = await process.communicate()
                stdout_str = stdout.decode("utf-8") if stdout else ""
                stderr_str = stderr.decode("utf-8") if stderr else ""

                # Check for digest in output (manifest exists)
                if process.returncode == 0 and ("digest" in stdout_str or "Digest" in stdout_str):
                    return True

                # Check for "no such manifest" in stderr (manifest does not exist)
                if "no such manifest" in stderr_str.lower():
                    return False

                # If neither, retry unless last attempt
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    return False  # Could not determine, assume does not exist
            except Exception:
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1)
                    continue
                return False  # Assume image doesn't exist if we can't check

    async def _run_git_command(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run a git command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            text=False,
            cwd=self.workspace_root,
        )
        stdout, stderr = await process.communicate()
        # Decode bytes to strings
        stdout_str = stdout.decode("utf-8") if stdout else ""
        stderr_str = stderr.decode("utf-8") if stderr else ""
        return subprocess.CompletedProcess(
            args=["git", *args],
            returncode=process.returncode,
            stdout=stdout_str,
            stderr=stderr_str,
        )

    async def build_docker_image(self, build_info: BuildInfo) -> Tuple[bool, str]:
        """Build a docker image for the given build info."""
        try:
            # Build the docker image
            process = await asyncio.create_subprocess_exec(
                "docker",
                "build",
                *DOCKER_BUILD_ARGS,
                "-t",
                build_info.container_tag,
                str(build_info.directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                text=False,
            )

            stdout, _ = await process.communicate()
            stdout_str = stdout.decode("utf-8") if stdout else ""

            if process.returncode == 0:
                return True, "Build successful"
            else:
                return False, f"Build failed:\n{stdout_str}"

        except Exception as e:
            return False, f"Build error: {e!s}"

    async def build_images_parallel(self, progress: Progress) -> Dict[str, Tuple[bool, str]]:
        """Build all required docker images in parallel."""
        task_ids = {}
        results = {}

        # Create tasks for images that need building
        images_to_build = [bi for bi in self.build_infos if bi.needs_build]

        if not images_to_build:
            return {}

        # Create progress tasks
        for build_info in images_to_build:
            task_id = progress.add_task(f"Building {build_info.directory.name}...", total=None)
            task_ids[build_info.container_tag] = task_id

        # Create async tasks
        async def build_with_progress(build_info: BuildInfo):
            task_id = task_ids[build_info.container_tag]
            progress.update(task_id, description=f"Building {build_info.directory.name}...")

            success, message = await self.build_docker_image(build_info)

            if success:
                progress.update(
                    task_id,
                    description=f"✅ {build_info.directory.name}",
                    completed=True,
                )
            else:
                progress.update(
                    task_id,
                    description=f"❌ {build_info.directory.name}",
                    completed=True,
                )

            return build_info.container_tag, (success, message)

        # Execute builds in parallel
        tasks = [build_with_progress(bi) for bi in images_to_build]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in task_results:
            if isinstance(result, Exception):
                self.console.print(f"[red]Unexpected error: {result}[/red]")
            else:
                tag, (success, message) = result
                results[tag] = (success, message)

        return results

    def run_tests(self, build_info: BuildInfo) -> Tuple[bool, str]:
        """Run tests in the docker container with streaming output."""
        try:
            # Run pytest in the container with real-time output
            cmd = [
                "docker",
                "run",
                *DOCKER_RUN_ARGS,
                build_info.container_tag,
                "pytest",
                "-v",
            ]

            self.console.print(f"\n[bold blue]Running tests for {build_info.directory.name}...[/bold blue]")
            self.console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

            # Use subprocess.run to stream output directly to console
            result = subprocess.run(
                cmd,
                check=False,
                text=True,
                capture_output=False,  # Don't capture - let it stream to console
                cwd=self.workspace_root,
            )

            if result.returncode == 0:
                self.console.print(f"[green]✅ Tests passed for {build_info.directory.name}[/green]")
                return True, "Tests passed"
            else:
                self.console.print(f"[red]❌ Tests failed for {build_info.directory.name}[/red]")
                return False, "Tests failed (see output above)"

        except Exception as e:
            error_msg = f"Test error: {e!s}"
            self.console.print(f"[red]{error_msg}[/red]")
            return False, error_msg

    def run_tests_serial(self, build_results: Dict[str, Tuple[bool, str]]) -> Dict[str, Tuple[bool, str]]:
        """Run tests serially for all successfully built containers."""
        test_results = {}

        # Filter to only test successfully built containers
        successful_builds = [
            bi
            for bi in self.build_infos
            if bi.needs_build and bi.container_tag in build_results and build_results[bi.container_tag][0]
        ]

        if not successful_builds:
            return {}

        self.console.print(f"\n[bold]Running tests for {len(successful_builds)} containers...[/bold]")

        for build_info in successful_builds:
            success, message = self.run_tests(build_info)
            test_results[build_info.container_tag] = (success, message)

        return test_results

    async def push_container(self, build_info: BuildInfo) -> Tuple[bool, str]:
        """Push container to remote registry."""
        try:
            # Push the docker image
            process = await asyncio.create_subprocess_exec(
                "docker",
                "push",
                build_info.container_tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                text=False,
            )

            stdout, _ = await process.communicate()
            stdout_str = stdout.decode("utf-8") if stdout else ""

            if process.returncode == 0:
                return True, "Push successful"
            else:
                return False, f"Push failed:\n{stdout_str}"

        except Exception as e:
            return False, f"Push error: {e!s}"

    async def push_successful_containers(
        self, test_results: Dict[str, Tuple[bool, str]], progress: Progress
    ) -> Dict[str, Tuple[bool, str]]:
        """Push all successful containers that don't have uncommitted changes in parallel."""
        push_results = {}
        task_ids = {}

        # Filter to only push successful, non-dirty containers
        containers_to_push = [
            bi
            for bi in self.build_infos
            if (
                bi.container_tag in test_results
                and test_results[bi.container_tag][0]  # Test passed
                and not bi.has_uncommitted_changes
            )  # No dirty flag
        ]

        if not containers_to_push:
            return {}

        # Create progress tasks
        for build_info in containers_to_push:
            task_id = progress.add_task(f"Pushing {build_info.directory.name}...", total=None)
            task_ids[build_info.container_tag] = task_id

        # Create async tasks
        async def push_with_progress(build_info: BuildInfo):
            task_id = task_ids[build_info.container_tag]
            progress.update(task_id, description=f"Pushing {build_info.directory.name}...")

            success, message = await self.push_container(build_info)

            if success:
                progress.update(
                    task_id,
                    description=f"✅ Pushed {build_info.directory.name}",
                    completed=True,
                )
            else:
                progress.update(
                    task_id,
                    description=f"❌ Push failed {build_info.directory.name}",
                    completed=True,
                )

            return build_info.container_tag, (success, message)

        # Execute pushes in parallel
        tasks = [push_with_progress(bi) for bi in containers_to_push]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in task_results:
            if isinstance(result, Exception):
                self.console.print(f"[red]Unexpected push error: {result}[/red]")
            else:
                tag, (success, message) = result
                push_results[tag] = (success, message)

        return push_results

    def print_build_report(self):
        """Print structured report of what will be built and tested."""
        if not self.build_infos:
            self.console.print("[yellow]No directories found for building.[/yellow]")
            return

        # Create summary table
        table = Table(title="Build Summary")
        table.add_column("Directory", style="cyan", no_wrap=True)
        table.add_column("Commit SHA", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Action", style="yellow")
        table.add_column("Container Tag", style="blue")

        for build_info in self.build_infos:
            # Determine status
            if build_info.has_uncommitted_changes:
                status = "[red]Uncommitted changes[/red]"
            elif not build_info.image_exists_in_registry:
                status = "[yellow]Image not in registry[/yellow]"
            else:
                status = "[green]Up to date[/green]"

            # Determine action
            if build_info.needs_build:
                action = "[yellow]Build & Test[/yellow]"
            else:
                action = "[green]Skip[/green]"

            table.add_row(
                str(build_info.directory.relative_to(self.workspace_root)),
                build_info.commit_sha,
                status,
                action,
                build_info.container_tag,
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

        # Print counts
        total = len(self.build_infos)
        to_build = sum(1 for bi in self.build_infos if bi.needs_build)
        to_skip = total - to_build

        self.console.print(f"[bold]Summary:[/bold] {total} directories found, {to_build} to build, {to_skip} to skip")

    def print_final_results(
        self,
        build_results: Dict[str, Tuple[bool, str]],
        test_results: Dict[str, Tuple[bool, str]],
        push_results: Dict[str, Tuple[bool, str]],
    ):
        """Print final results summary."""
        # Create results table
        table = Table(title="Final Results")
        table.add_column("Directory", style="cyan", no_wrap=True)
        table.add_column("Build", style="yellow")
        table.add_column("Tests", style="green")
        table.add_column("Push", style="blue")
        table.add_column("Notes", style="dim")

        for build_info in self.build_infos:
            dir_name = str(build_info.directory.relative_to(self.workspace_root))
            tag = build_info.container_tag

            # Build status
            if not build_info.needs_build:
                build_status = "[dim]Skipped[/dim]"
            elif tag in build_results:
                build_status = "✅ Pass" if build_results[tag][0] else "❌ Fail"
            else:
                build_status = "[dim]Not built[/dim]"

            # Test status
            if self.skip_tests:
                test_status = "[yellow]Skipped[/yellow]"
            elif tag in test_results:
                test_status = "✅ Pass" if test_results[tag][0] else "❌ Fail"
            elif not build_info.needs_build:
                test_status = "[dim]Skipped[/dim]"
            else:
                test_status = "[dim]Not tested[/dim]"

            # Push status
            if self.skip_push:
                push_status = "[yellow]Skipped[/yellow]"
            elif tag in push_results:
                push_status = "✅ Pushed" if push_results[tag][0] else "❌ Failed"
            elif build_info.has_uncommitted_changes:
                push_status = "[yellow]Dirty (skipped)[/yellow]"
            elif not build_info.needs_build:
                push_status = "[dim]Not needed[/dim]"
            else:
                push_status = "[dim]Not pushed[/dim]"

            # Notes
            notes = []
            if build_info.has_uncommitted_changes:
                notes.append("uncommitted changes")
            if not build_info.image_exists_in_registry and not build_info.needs_build:
                notes.append("image exists in registry")

            table.add_row(
                dir_name,
                build_status,
                test_status,
                push_status,
                ", ".join(notes) if notes else "",
            )

        self.console.print()
        self.console.print(table)

        # Print summary counts
        total_dirs = len(self.build_infos)
        built = len([k for k, v in build_results.items() if v[0]])
        tested = len([k for k, v in test_results.items() if v[0]])
        pushed = len([k for k, v in push_results.items() if v[0]])

        self.console.print()
        self.console.print(
            f"[bold]Final Summary:[/bold] {total_dirs} directories, {built} built, {tested} tested, {pushed} pushed"
        )

    async def run_ci_pipeline(self):
        """Run the complete CI pipeline."""
        # Step 1: Discover directories
        with self.console.status("Discovering build directories..."):
            directories = self.discover_build_directories()

        self.console.print(f"Found {len(directories)} directories with Dockerfiles")

        if not directories:
            self.console.print("[yellow]No directories with Dockerfiles found.[/yellow]")
            return

        # Step 2: Analyze each directory
        self.build_infos = await asyncio.gather(
            *[self.analyze_directory_status(directory) for directory in directories]
        )

        # Step 3: Print build report
        self.print_build_report()

        # Step 4: Build images in parallel
        build_results = {}
        images_to_build = [bi for bi in self.build_infos if bi.needs_build]

        if images_to_build:
            self.console.print(f"\n[bold]Building {len(images_to_build)} Docker images in parallel...[/bold]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                build_results = await self.build_images_parallel(progress)
        else:
            self.console.print("\n[green]No images need to be built.[/green]")

        # Step 5: Run tests serially
        test_results = {}
        if self.skip_tests:
            self.console.print("\n[yellow]Skipping tests (--skip-tests flag provided)[/yellow]")
            # For containers that were built successfully, mark tests as "skipped"
            for tag, (success, _) in build_results.items():
                if success:
                    test_results[tag] = (True, "Tests skipped")
        elif build_results:
            successful_builds = [k for k, v in build_results.items() if v[0]]
            if successful_builds:
                test_results = self.run_tests_serial(build_results)
            else:
                self.console.print("\n[red]No successful builds to test.[/red]")

        # Step 6: Push successful containers
        push_results = {}
        if self.skip_push:
            self.console.print("\n[yellow]Skipping push (--skip-push flag provided)[/yellow]")
        elif test_results:
            successful_tests = [k for k, v in test_results.items() if v[0]]
            containers_to_push = [
                bi
                for bi in self.build_infos
                if bi.container_tag in successful_tests and not bi.has_uncommitted_changes
            ]

            if containers_to_push:
                self.console.print(f"\n[bold]Pushing {len(containers_to_push)} containers to registry...[/bold]")
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    push_results = await self.push_successful_containers(test_results, progress)
            else:
                self.console.print(
                    "\n[yellow]No containers to push (either failed tests or have uncommitted changes).[/yellow]"
                )

        # Step 7: Print final results
        self.print_final_results(build_results, test_results, push_results)

        # Check for any failures and exit with appropriate code
        failed_builds = [k for k, v in build_results.items() if not v[0]]
        failed_tests = [k for k, v in test_results.items() if not v[0]] if not self.skip_tests else []
        failed_pushes = [k for k, v in push_results.items() if not v[0]] if not self.skip_push else []

        if failed_builds or failed_tests or failed_pushes:
            self.console.print("\n[red]Some operations failed. Check the logs above for details.[/red]")
            # Print error details
            for tag, (success, message) in build_results.items():
                if not success:
                    self.console.print(f"\n[red]Build error for {tag}:[/red]\n{message}")
            if not self.skip_tests:
                for tag, (success, message) in test_results.items():
                    if not success:
                        self.console.print(f"\n[red]Test error for {tag}:[/red]\n{message}")
            if not self.skip_push:
                for tag, (success, message) in push_results.items():
                    if not success:
                        self.console.print(f"\n[red]Push error for {tag}:[/red]\n{message}")
            sys.exit(1)
        else:
            self.console.print("\n[green]All operations completed successfully! 🎉[/green]")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CI Build and Test Logic for BioNeMo Recipes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run full CI pipeline
  %(prog)s --skip-tests       # Build and push, but skip tests
  %(prog)s --skip-push        # Build and test, but skip push
  %(prog)s --skip-tests --skip-push  # Build only
        """,
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest tests in containers",
    )

    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip pushing containers to remote registry",
    )

    return parser.parse_args()


async def main():
    """Main entry point for the CI script."""
    args = parse_args()
    console = Console()

    # Show configuration
    console.print(Panel.fit("BioNeMo Recipes CI Build and Test", style="bold blue"))

    if args.skip_tests or args.skip_push:
        skip_info = []
        if args.skip_tests:
            skip_info.append("tests")
        if args.skip_push:
            skip_info.append("push")
        console.print(f"[yellow]Skipping: {', '.join(skip_info)}[/yellow]")

    ci_manager = CIManager(console, skip_tests=args.skip_tests, skip_push=args.skip_push)
    await ci_manager.run_ci_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
