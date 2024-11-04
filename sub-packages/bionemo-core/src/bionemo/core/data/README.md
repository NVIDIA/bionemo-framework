# BioNeMo test data management

This library manages the downloading and caching of large or binary data files used in the documentation or test suite.
These files should not be committed directly to the repo, and instead should be loaded at test-time when they are
needed.

We currently support two locations for test data or saved models:

SwiftStack

:   SwiftStack or `pbss` is an NVIDIA-internal, s3-compatible object store that allows for very large data and fast,
    parallel read/writes. Most critically, `pbss` can be uploaded to without legal approvals for dataset redistribution.
    These files will not be accessible by external collaborators.

[NGC](https://catalog.ngc.nvidia.com/)

:   NGC hosts containers, models, and resources, some of which require authentication and others that are generally
    available. This library uses the model and resource types to save test data and reference model weights. These items
    are accessible by external collaborators, but require legal approval before re-distributing test data.


## Loading test or example data

Test data are specified via yaml files in `sub-packages/bionemo-testing/src/bionemo/testing/data/resources`. As an
example, in `esm2.yaml`:

```yaml
- tag: nv_650m:1.0
  ngc: "nvidia/clara/esm2nv650m:1.0"
  ngc_registry: model
  pbss: "s3://bionemo-ci/models/esm2nv_650M_converted.nemo"
  sha256: 1e38063cafa808306329428dd17ea6df78c9e5d6b3d2caf04237c555a1f131b7
  owner: Farhad Ramezanghorbani <farhadr@nvidia.com>
  description: >
    A pretrained 650M parameter ESM-2 model.
    See https://ngc.nvidia.com/catalog/models/nvidia:clara:esm2nv650m.
```

To load these model weights during a test, use the [load][bionemo.testing.data.load.load] function with the filename and
tag of the desired asset, which returns a path a the specified file:

```python
path_to_my_checkpoint = load("esm2/nv_650m:1.0")
config = ESM2Config(nemo1_ckpt_path=path_to_my_checkpoint)
```

If this function is called without the data available on the local machine, it will be fetched from the default source
(currently `pbss`.) Otherwise, it will return the cached directory. To download with NGC, pass `source="ngc"` to
[load][bionemo.testing.data.load.load].

## File unpacking and/or decompression

All test artifacts are individual files. If a zip or tar archive is specified, it will be unpacked automatically, and
the path to the directory will be returned via [load][bionemo.testing.data.load.load]. Compressed files ('gzip', 'bz2',
or 'xz') are automatically decompressed before they are returned. The file's compression and/or archive format is
determined based on the filename specified in the `pbss` URL.

!!! note "Files in NGC resources"

    NGC resources are folders, i.e., they may contain multiple files per resource.
    [load][bionemo.testing.data.load.load] will _only_ download the filename matching the stem of the `pbss` url. The
    same NGC resource can therefore be used to host multiple test assets that are used independently.


## Adding new test assets

To add new data, first ensure that the data is available from either NGC or `pbss`. Next, extend or create a new yaml
file in `sub-packages/bionemo-testing/src/bionemo/testing/data/resources` with the required information. Owner emails
must be provided for all assets. The description and `ngc` fields are currently optional. If the `sha256` is left
unspecified, `pooch` will report the downloaded file's sha when loaded.

!!! warning

    SHAs should be provided for all files to ensure the download completes correctly, and to invalidate caches if the
    files change.
