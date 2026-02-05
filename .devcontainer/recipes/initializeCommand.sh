#!/bin/bash
# Create the mounted config directories if they don't already exist

mkdir -p ~/.devcontainer_cache
mkdir -p ~/.ssh
mkdir -p ~/.cache/pre-commit-devcontainer
mkdir -p ~/.config
mkdir -p ~/.cursor

[ ! -f ~/.netrc ] && touch ~/.netrc
[ ! -f ~/.nvidia-api-key ] && touch ~/.nvidia-api-key
[ ! -f ~/.claude-devcontainer.json ] && touch ~/.claude-devcontainer.json
[ ! -f ~/.bash_history_devcontainer ] && touch ~/.bash_history_devcontainer


exit 0
