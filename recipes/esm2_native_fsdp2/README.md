# ESM-2 training with nvFSDP and custom pytorch training loop

To build the docker image with gitlab authentication, first ensure you have a gitlab user access
token stored in your `~/.netrc` file with the following format:

```
machine gitlab-master.nvidia.com
  login user
  password <your_gitlab_token>
```

Then, build the docker image with the following command:

```bash
docker build --secret id=netrc,src=$HOME/.netrc -t my_image .
```

This will make sure the netrc file is available to the docker build process for authentication.

We can remove this once nvFSDP is publically available on github.com.

## Running training

Run training with

```bash
docker run --rm -it --gpus all my_image torchrun train.py --config-name L0_sanity
```
