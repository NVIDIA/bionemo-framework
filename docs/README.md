# BioNeMo Framework User Documentation

## Structure

The [contents.md](bionemo/contents.md) page creates outline for the documentation.

The corresponding markdown (`.md`) files are named as `xxxx-xxx-fw.md` indicating that they refer to BioNeMo FrameWork related page. This is done to distinguish them from the older BioNemo Service pages, which are retained for reference but will eventually removed from the repo. 

## Developing

The `launch_docs.sh` script will create and launch the container for you:
* `launch_docs.sh build` to build the Docker container
* `launch_docs.sh dev` to run the container and execute `recompile_html.sh` inside the container. This will regularly recompiles the html

Then run the following inside the repo -- can either be inside the container or directly on the workstation: 

```bash
PORT=8080 && python3 -m http.server ${PORT} --directory build/html
```

## Uploading HTML Files

Command to upload the docs to s3:
    `aws s3 cp build/html/ s3://nv-developer-docs/bionemo-XXX/ --recursive`

Verify upload
    `aws s3 ls s3://nv-developer-docs/bionemo-XXX/`
