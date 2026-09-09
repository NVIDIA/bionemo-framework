# Infrastructure guidance

Use the execution path actually available to the user: local GPU, SSH, Slurm, Lepton, or a manual handoff. Inspect local policy and current command help, and adapt existing scripts when possible.

On a GPU workstation where the agent starts outside Docker, use a GPU-enabled container instead of assuming the host can build the recipe directly. The build scripts are tested against the NVIDIA PyTorch container environment. Build from the recipe `Dockerfile` or use the repository `.devcontainer/` as a convenient base, expose the workstation GPUs through the NVIDIA container runtime, and bind-mount the checkout and selected data, cache, and result locations. Build the code inside the mounted checkout with `./.ci_build.sh`, then source `.ci_test_env.sh` for subsequent commands so the running code and editable checkout remain aligned.

Match the container CUDA runtime to the host driver before building. If that compatible image's cuDNN is older than a required kernel but the recipe installs a newer runtime, select the recipe library at process startup—an explicit preload may be necessary because a search path cannot replace an already-loaded library—and verify effective CUDA, cuDNN, and the kernel on every distributed worker.

The external-asset installer selects native Linux x86_64 or aarch64 downloads for MMseqs2-GPU, DIAMOND, HMMER, and AMRFinderPlus. It keeps non-x86 tools in architecture-qualified extraction directories so copied x86 caches are not reused accidentally. Before scientific stages, verify the selected executables' architecture and versions. A CPU architecture other than x86_64 or aarch64 still needs explicit archive mappings and validation; changing only the base container is insufficient.

Source-built Biotite dependencies under `--no-build-isolation` require Hatchling, hatch-vcs, and
hatch-cython, so `build_requirements.txt` keeps those backends active. On aarch64, a packaging
fallback was validated on 2026-08-21: if the resolver encounters the known Biotite/Biotraj wheel
and sdist-metadata gap, enable the commented `biotite 0.41.2` and pinned `biotraj 1.2.2` examples
in `pyproject.toml`. A root-owned container using a user-owned bind mount may also need
command-scoped Git trust for that exact checkout. Leave those source pins disabled when normal
wheels resolve.

Build architecture-specific compiled extensions, such as dataset helpers, while the image or build
environment is writable, then probe imports as the actual non-root runtime UID. Do not defer
compilation into a root-owned installed environment. Keep large generated data, results, assets,
and checkpoints out of the image build context and mount them at runtime.

On coherent-memory Grace-Blackwell systems, HBM may appear as a memory-only NUMA node; checkpoint page cache can occupy it without a CUDA process. Keep device-required memory nodes visible and place CPU/offload/cache allocations on CPU-attached memory using supported controls. Qualify memory across optimizer materialization, validation, save, and resume as described in the RL operator skill; a single fitting rollout is insufficient.

Use the execution skill's job and runlog handoff for long runs. Keep credentials out of commands and shared logs.
