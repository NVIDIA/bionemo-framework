#!/bin/bash
# Create the mounted config directories if they don't already exist

mkdir -p ~/.devcontainer_cache
mkdir -p ~/.ssh
mkdir -p ~/.cache/pre-commit-devcontainer
mkdir -p ~/.gnupg
[ ! -f ~/.netrc ] && touch ~/.netrc

# Create the ~/.bash_history_devcontainer file if it doesn't exist
[ ! -f ~/.bash_history_devcontainer ] && touch ~/.bash_history_devcontainer

exit 0
