# Devcontainer pre-build configuration

This devcontainer file contains the setup for building the image used in
bionemo2 from scratch. Building and pushing the image to a remote repository
can be done with the devcontainer cli via

```shell
# Make sure we have a revent version of node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
nvm install 20.9.0
nvm use 20.9.0

# Install and build th devcontainer (run from the repo root)
npm install -g @devcontainers/cli
devcontainer build --workspace-folder .github/ --image-name nvcr.io/nvidian/cvai_bnmo_trng/bionemo:bionemo2-devcontainer --push true
```
