#!/bin/bash
# Create the mounted config directories if they don't already exist

mkdir -p ~/.aws
mkdir -p ~/.ngc
mkdir -p ~/.cache
mkdir -p ~/.ssh
[ ! -f ~/.netrc ] && touch ~/.netrc
exit 0
