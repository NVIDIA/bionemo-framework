To get started, first download [`just`](https://github.com/casey/just). You can use [Homebrew](https://brew.sh/) on OS X & Linux:
```bash
brew install just
```

**Once you have `just`, you need to run the `just setup` command once _before_ you can run any other command.**
Thus, if it's your first time, you will need to do this first:
```bash
just setup
just <command you want to run>
```

You can see all of the commands for the development cycle by running `just`. These commands are executable as
`just X` for each command `X` listed:
```
build-dev              # Builds the development image.
build-release          # Builds the release image.
run-dev cmd='bash'     # Runs an interactive program in the development bionemo image.
run-release cmd='bash' # Runs an interactive program in the release bionemo image.
setup                  # Checks for installed programs (docker, git, etc.), their versions, and grabs the latest cache image.
test                   # Executes pytest in the release image.
```

You can combine `just` commands together. For example, run `just build-dev build-release` to build both images.
