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
training and artifact handoffs. Keep the upstream-gap reference limited to open
framework issues explicitly cited in the recipe source.

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
and joint projector GRPO constraints. Automated tests check release packaging,
checkout reference paths, and evaluation-case structure.

## Evaluation Results

Automated checks validate packaging and integration references. Declarative
cases are prompts for future agent evaluation; they have not been executed as
end-to-end model or GPU evaluations.

## Skill Version

BioNeMo Recipes revision containing this skill; initial root package 2026-10-02.
