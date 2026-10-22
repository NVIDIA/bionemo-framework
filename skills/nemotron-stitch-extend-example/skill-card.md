## Description

Add or adapt a Nemotron Stitch example for an external encoder and paired data,
using the LLaVA recipe as the reference for training and serving integration.

## Owner

NVIDIA

### License/Terms of Use

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)

## Use Case

Locate a compatible BioNeMo Recipes checkout, identify the application-owned
encoder, data, and prompt boundaries, and build a small example with explicit
training and artifact handoffs. Use the genetics and chemistry examples for
typed domain chats, YAML-owned encoder settings, native-processor checkpoint
saves, worker serialization, isolated GRPO checkpoint paths, bounded SFT
checkpoint retention, and packed full decoder training. Keep the upstream-gap reference limited to substantive open
custom-modality integration gaps with evidence and a deletion condition.

## Requirements / Dependencies

A compatible BioNeMo Recipes checkout with `recipes/nemotron-stitch` and an
agent that can read and edit repository files. Framework tests and training
use the image pinned by the selected recipe. Model, data, registry, and compute
credentials depend on the requested example.

## Known Risks and Mitigations

- The installed skill may be separate from the source checkout: locate and
  record the recipe root before resolving implementation paths.
- Runtime contracts can change: inspect the selected revision and framework
  pins before applying the snapshot's topology or scheduler guidance.
- Encoder features and prompt slots can be misaligned: validate token geometry,
  masking, trainability, and artifact handoffs using the recipe tests.
- Tiny packed datasets can leave distributed samplers empty: use the documented
  repeated manifest view and explicit `--` CLI bounds for sanity runs.
- LLaVA full-parameter Lightning reload remains unsupported pending resolution of upstream refit corruption: keep
  its configuration for failure reproduction and do not use it for training.
- Smoke checks do not measure model quality: report quality results with the
  dataset split, hardware, training budget, checkpoint, and metric definition.

## References

- [Skill instructions](SKILL.md)
- [Evaluation cases](evals/evals.json)
- [BioNeMo Recipes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes)

## Skill Output

Example source, stage configurations, runnable documentation, and verification
results in the selected recipe checkout.

## Evaluation Tasks

Declarative cases cover a supplied checkout, an independently installed skill,
joint projector GRPO constraints, and domain-example selection. Automated tests check release packaging,
checkout reference paths, and evaluation-case structure.

## Evaluation Results

Automated checks validate packaging and integration references. Declarative
cases are prompts for future agent evaluation; they have not been executed as
end-to-end model or GPU evaluations.

## Skill Version

BioNeMo Recipes revision containing this skill; initial root package 2026-10-02.
