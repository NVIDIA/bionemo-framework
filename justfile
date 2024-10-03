#!/usr/bin/env -S just --justfile

######

set shell := ["sh", "-c"]
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set allow-duplicate-recipes
set positional-arguments
set dotenv-load
set export

alias s := serve

bt := '0'

export RUST_BACKTRACE_1 := bt

log := "warn"

export JUST_LOG := (log + "ing" + `grep loop /etc/networks | cut -f2`)

tmpdir  := `mktemp`
version := "0.2.7"
tardir  := tmpdir / "awesomesauce-" + version
foo1    := / "tmp"
foo2_3  := "a/"
tarball := tardir + ".tar.gz"

export RUST_BACKTRACE_2 := "1"
string-with-tab             := "\t"
string-with-newline         := "\n"
string-with-carriage-return := "\r"
string-with-double-quote    := "\""
string-with-slash           := "\\"
string-with-no-newline      := "\
"

# Newlines in variables
single := '
hello
'

double := "
goodbye
"
escapes := '\t\n\r\"\\'

# this string will evaluate to `foo\nbar\n`
x := '''
  foo
  bar
'''

# this string will evaluate to `abc\n  wuv\nbar\n`
y := """
  abc
    wuv
  xyz
"""

for:
  for file in `ls .`; do \
    echo $file; \
  done

serve:
  touch {{tmpdir}}/file

# This backtick evaluates the command `echo foo\necho bar\n`, which produces the value `foo\nbar\n`.
stuff := ```
    echo foo
    echo bar
  ```


an_arch := trim(lowercase(justfile())) + arch()
trim_end := trim_end("99.99954%   ")
home_dir := replace(env_var('HOME') / "yep", 'yep', '')
quoted := quote("some things beyond\"$()^%#@!|-+=_*&'`")
smartphone := trim_end_match('blah.txt', 'txt')
museum := trim_start_match(trim_start(trim_end_matches('   yep_blah.txt.txt', '.txt')), 'yep_')
water := trim_start_matches('ssssssoup.txt', 's')
congress := uppercase(os())
fam := os_family()
path_1 := absolute_path('test')
path_2 := '/tmp/subcommittee.txt'
ext_z := extension(path_2)
exe_name := file_name(just_executable())
a_stem := file_stem(path_2)
a_parent := parent_directory(path_2)
sans_ext := without_extension(path_2)
camera := join('tmp', 'dir1', 'dir2', path_2)
cleaned := clean('/tmp/blah/..///thing.txt')
id__path := '/tmp' / sha256('blah') / sha256_file(justfile())
_another_var := env_var_or_default("HOME", justfile_directory())
python := `which python`

exists := if path_exists(just_executable()) =~ '^/User' { uuid() } else { 'yeah' }

foo   := if env_var("_") == "/usr/bin/env" { `touch /tmp/a_file` } else { "dummy-value" }
foo_b := if "hello" == "goodbye" { "xyz" } else { if "no" == "no" { "yep"} else { error("123") } }
foo_c := if "hello" == "goodbye" {
  "xyz"
} else if "a" == "a" {
  "abc"
} else {
  "123"
}

bar:
  @echo {{foo}}


bar2 foo_stuff:
  echo {{ if foo_stuff == "bar" { "hello" } else { "goodbye" } }}

executable:
  @echo The executable is at: {{just_executable()}}


rustfmt:
  find {{invocation_directory()}} -name \*.rs -exec rustfmt {} \;

test:
  echo "{{home_dir}}"


linewise:
  Write-Host "Hello, world!"

serve2:
  @echo "Starting server with database $DATABASE_ADDRESS on port $SERVER_PORT…"


shebang := if os() == 'windows' {
  'powershell.exe'
} else {
  '/usr/bin/env pwsh'
}

shebang:
	#!{{shebang}}
	$PSV = $PSVersionTable.PSVersion | % {"$_" -split "\." }
	$psver = $PSV[0] + "." + $PSV[1]
	if ($PSV[2].Length -lt 4) {
		$psver += "." + $PSV[2] + " Core"
	} else {
		$psver += " Desktop"
	}
	echo "PowerShell $psver"

@foo:
  echo bar

@test5 *args='':
  bash -c 'while (( "$#" )); do echo - $1; shift; done' -- "$@"

test2 $RUST_BACKTRACE="1":
  # will print a stack trace if it crashes
  cargo test


notify m="":
	keybase chat send --topic-type "chat" --channel <channel> <team> "upd(<repo>): {{m}}"

# Sample project script 2
script2 *ARGS:
    {{ python }} script2.py {{ ARGS }}

braces:
  echo 'I {{{{LOVE}} curly braces!'

_braces2:
  echo '{{'I {{LOVE}} curly braces!'}}'

_braces3:
  echo 'I {{ "{{" }}LOVE}} curly braces!'

foo2:
  -@cat foo
  echo 'Done!'

test3 target tests=path_1:
  @echo 'Testing {{target}}:{{tests}}…'
  ./test --tests {{tests}} {{target}}

test4 triple=(an_arch + "-unknown-unknown") input=(an_arch / "input.dat"):
  ./test {{triple}}

variadic $VAR1_1 VAR2 VAR3 VAR4=("a") +$FLAGS='-q': foo2 braces
  cargo test {{FLAGS}}

time:
  @-date +"%H:%S"
  -cat /tmp/nonexistent_file.txt
  @echo "finished"

justwords:
  grep just \
    --text /usr/share/dict/words \
    > /tmp/justwords

# Subsequent dependencies
# https://just.systems/man/en/chapter_37.html
# To test, run `$ just -f test-suite.just b`
a:
  echo 'A!'

b: a && d
  echo 'B start!'
  just -f {{justfile()}} c
  echo 'B end!'

c:
  echo 'C!'

d:
  echo 'D!'

#######

alias t := test

log := "warn"

export JUST_LOG := log

[group: 'dev']
watch +args='test':
  cargo watch --clear --exec '{{ args }}'

[group: 'test']
test:
  cargo test --all

[group: 'check']
ci: forbid test build-book clippy
  cargo fmt --all -- --check
  cargo update --locked --package just

[group: 'check']
fuzz:
  cargo +nightly fuzz run fuzz-compiler

[group: 'misc']
run:
  cargo run

# only run tests matching PATTERN
[group: 'test']
filter PATTERN:
  cargo test {{PATTERN}}

[group: 'misc']
build:
  cargo build

[group: 'misc']
fmt:
  cargo fmt --all

[group: 'check']
shellcheck:
  shellcheck www/install.sh

[group: 'doc']
man:
  mkdir -p man
  cargo run -- --man > man/just.1

[group: 'doc']
view-man: man
  man man/just.1

# add git log messages to changelog
[group: 'release']
update-changelog:
  echo >> CHANGELOG.md
  git log --pretty='format:- %s' >> CHANGELOG.md

[group: 'release']
update-contributors:
  cargo run --release --package update-contributors

[group: 'check']
outdated:
  cargo outdated -R

# publish current GitHub master branch
[group: 'release']
publish:
  #!/usr/bin/env bash
  set -euxo pipefail
  rm -rf tmp/release
  git clone git@github.com:casey/just.git tmp/release
  cd tmp/release
  ! grep '<sup>master</sup>' README.md
  VERSION=`sed -En 's/version[[:space:]]*=[[:space:]]*"([^"]+)"/\1/p' Cargo.toml | head -1`
  git tag -a $VERSION -m "Release $VERSION"
  git push origin $VERSION
  cargo publish
  cd ../..
  rm -rf tmp/release

[group: 'release']
readme-version-notes:
  grep '<sup>master</sup>' README.md

# clean up feature branch BRANCH
[group: 'dev']
done BRANCH=`git rev-parse --abbrev-ref HEAD`:
  git checkout master
  git diff --no-ext-diff --quiet --exit-code
  git pull --rebase github master
  git diff --no-ext-diff --quiet --exit-code {{BRANCH}}
  git branch -D {{BRANCH}}

# install just from crates.io
[group: 'misc']
install:
  cargo install -f just

# install development dependencies
[group: 'dev']
install-dev-deps:
  rustup install nightly
  rustup update nightly
  cargo +nightly install cargo-fuzz
  cargo install cargo-check
  cargo install cargo-watch
  cargo install mdbook mdbook-linkcheck

# everyone's favorite animate paper clip
[group: 'check']
clippy:
  cargo clippy --all --all-targets --all-features -- --deny warnings

[group: 'check']
forbid:
  ./bin/forbid

[group: 'dev']
replace FROM TO:
  sd '{{FROM}}' '{{TO}}' src/*.rs

[group: 'demo']
test-quine:
  cargo run -- quine

# make a quine, compile it, and verify it
[group: 'demo']
quine:
  mkdir -p tmp
  @echo '{{quine-text}}' > tmp/gen0.c
  cc tmp/gen0.c -o tmp/gen0
  ./tmp/gen0 > tmp/gen1.c
  cc tmp/gen1.c -o tmp/gen1
  ./tmp/gen1 > tmp/gen2.c
  diff tmp/gen1.c tmp/gen2.c
  rm -r tmp
  @echo 'It was a quine!'

quine-text := '
  int printf(const char*, ...);

  int main() {
    char *s =
      "int printf(const char*, ...);"
      "int main() {"
      "   char *s = %c%s%c;"
      "  printf(s, 34, s, 34);"
      "  return 0;"
      "}";
    printf(s, 34, s, 34);
    return 0;
  }
'

[group: 'test']
test-completions:
  ./tests/completions/just.bash

[group: 'check']
build-book:
  cargo run --package generate-book
  mdbook build book/en
  mdbook build book/zh

# run all polyglot recipes
[group: 'demo']
polyglot: _python _js _perl _sh _ruby

_python:
  #!/usr/bin/env python3
  print('Hello from python!')

_js:
  #!/usr/bin/env node
  console.log('Greetings from JavaScript!')

_perl:
  #!/usr/bin/env perl
  print "Larry Wall says Hi!\n";

_sh:
  #!/usr/bin/env sh
  hello='Yo'
  echo "$hello from a shell script!"

_nu:
  #!/usr/bin/env nu
  let hellos = ["Greetings", "Yo", "Howdy"]
  $hellos | each {|el| print $"($el) from a nushell script!" }

_ruby:
  #!/usr/bin/env ruby
  puts "Hello from ruby!"

# Print working directory, for demonstration purposes!
[group: 'demo']
pwd:
  echo {{invocation_directory()}}

[group: 'test']
test-bash-completions:
  rm -rf tmp
  mkdir -p tmp/bin
  cargo build
  cp target/debug/just tmp/bin
  ./tmp/bin/just --completions bash > tmp/just.bash
  echo 'mod foo' > tmp/justfile
  echo 'bar:' > tmp/foo.just
  cd tmp && PATH="`realpath bin`:$PATH" bash --init-file just.bash

[group: 'test']
test-release-workflow:
  -git tag -d test-release
  -git push origin :test-release
  git tag test-release
  git push origin test-release

# Local Variables:
# mode: makefile
# End:
# vim: set ft=make :
