---
name: bionemo-phage-design-plan-rl-objectives
description: Use when converting a phage-design goal into target-specific RL rewards, validation criteria, and final QC filters, especially for a new reference phage or altered objective.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Design Phage RL Objectives

Work inside the recipe and result roots selected by the controller. Translating the user's desired final phage product into online rewards, validation criteria, and final hard filters is a core agentic capability. Default to whole-genome design unless the user explicitly requests a locus-, module-, or RBP-only task.

Prefer modifying tested rewards when their measurements, direction, and failure semantics scientifically transfer. Existing functions are starting points rather than a closed catalog: beyond a faithful experiment rerun, novel reward functions are expected for important requirements the current portfolio does not capture. For those gaps, creatively invent functions from literature, biological mechanisms, viable references, available tools, and prior or partial-run evidence, then apply the score-definition, control, telemetry, and final-QC requirements below.

Cover complete-genome viability and the productive-infection lifecycle, not only host binding: adsorption/entry, defense and counter-defense, replication, morphogenesis/packaging, lysis/progeny, essential/key genes, regulatory features including the non-coding leader/promoter and transcription terminators, synteny, topology, composition including host restriction-site avoidance and host-derived DNA exclusion, similarity to viable references, desired host direction, diversity, and intended-use safety. A host-range model is one signal and does not prove productive infection.

When host-range or bootability rewards are sparse, do not rely on them as the only learning signal. Add biologically justified intermediate rewards—such as essential-gene completeness and reasonable synteny—with carefully calibrated partial credit toward evidence-supported ranges. Log them separately. Ground required functions in deletion/complementation or other functional studies for the intended host and endpoint, recording conditional exceptions. Reference presence and shared annotation labels do not establish essentiality or interchangeability. Build detection alternatives from diverse natural homologs and functional/domain evidence rather than only swaps reported in one design study; distinguish plausible function recognition from demonstrated compatibility in a new genome. The [PhiX required-function profile](../../configs/required_genes.md) illustrates named functions, reviewed family alternatives, and the separation between experimental biology and calibrated detection thresholds.

For whole-genome designs—including custom or adapted runs—include AMR, toxin, and lysogeny as separate RL objectives. An explicitly scoped locus or module edit may omit objectives that the edited region cannot affect.

Use only the verified termini class from collection: it determines circular ORF calling, whether prompts may start at arbitrary rotations, whether synteny may be rotation-invariant, and whether a terminal-repeat reward applies at all. For collapsed DTRs, choose one molecular form—deposited repeat unit or packaged genome—and use it consistently for length, coding density, coordinates, prompts, synteny, and rewards; packaged length is deposit length plus one DTR. Genuinely circular ssDNA Microviridae such as PhiX174 have no DTR, so deposited and packaged unit lengths are identical. A wrong class or mixed form fails silently—nothing errors, but rewards can score a dead design as viable.

Before the sampling handoff, identify every gene, module, motif, or uncertain interval the model should remain free to redesign. Record their circular coordinates as origin-aware prompt-exclusion intervals, plus any defensible neutral anchor intervals. Prompt overlap fixes those bases even when it covers only part of a feature; if no neutral anchor exists, require calibration to revisit the prompting strategy rather than silently locking a design target. Biological rotation compatibility alone does not qualify arbitrary prompt starts: carry forward the collection's related-assembly start-site and SFT-exposure evidence. The maintained PhiX default uses coordinate 1; see the [prompt-origin rationale](../../examples/README.md#prompt-origin).

Before defining family-copy and synteny rewards, mark each gene slot as reference-preserved or functionally swappable; use reference homology only for the former.

Treat the approved objective portfolio as a scientific baseline, not an immutable contract. Reconcile the planned, configured, and emitted components, then continue autonomously: a discrepancy is evidence to diagnose and record, not a reason to pause a late-stage run. Dropping an agreed objective is a last resort; first try to repair or recover it, then, if omission is the strongest defensible path, record its evidence, scientific impact, scope, and restoration criteria while retaining complementary nonredundant evidence.

Proactively adding well-supported shaping objectives is important when literature review or partial-run evidence indicates they will better guide the project toward the desired endpoint. Calibrate and test the added term, version the active objective set and change point, and do not compare aggregate rewards across different sets as the same metric. Make these decisions autonomously, and summarize them in the next natural user update and whenever asked.

Define components using the concise [objective guidance](references/objective-guidance.md). Each reward remains on `[0, 1]`: zero is a meaningful random/baseline or clearly unacceptable outcome, not merely the raw metric's numeric zero, and one is the supported target. Missing, invalid, empty, or failed measurements must not crash the portfolio or receive accidental positive credit.

For every new or modified component, identify strong positive and negative controls: biologically relevant genomes or counterfactuals known or expected to score high and low, with provenance, rationale, and expected ordering. Also include baseline/random, boundary, no-hit/empty, too-short, missing-gene, and tool-failure cases as applicable. Specify what is logged so a silently skipped or fixed-zero component is visible. Classify each hard gate as non-directional viability/naturalness, intended-use safety, or an explicitly directional goal, and apply the matching control contract in the objective guidance. Calibrate final gate decisions independently from reward shaping, while allowing shared measurements and versioned post-hoc replay.

Check the portfolio for duplicate signals, scale dominance, incompatible directions, easy gaming, and combinations that reward the wrong biological endpoint. Compare reference, baseline/random, desired, and counterexample designs. Preserve component-level validation when aggregate component sets differ.

For the PhiX174 case study, retain filters 1–6, 8, and 9 with filter 7 disabled for the target profile, and run filter 7 separately as a diagnostic. Add relevant safety objectives without pretending this computational profile establishes biological viability.

Write a concise `artifacts/RL_OBJECTIVES.yaml` or table containing the selected molecular form, component definitions, baselines/targets, missing-data behavior, controls, hard-QC relationship, validation/selection criteria, prompt-exclusion and neutral-anchor intervals, and unresolved biological choices.

Also write `artifacts/RL_SCORE_DEFINITIONS.md` in the selected result root for a scientist reading the run record. Give every enabled objective one compact row or subsection with its reward column, measured quantity and units, direction, exact formula and resolved target-specific settings, zero-credit region or state (lower and upper sides when applicable), full-credit region or state, and how partial credit changes on either side. State categorical and missing/invalid/empty/no-hit/missing-gene/tool-failure behavior directly when numeric ranges do not apply. Include the biological rationale, evidence citations, controls, telemetry, and relationship to final hard QC. This is an agent-produced objective-design artifact, not a required stage of the fully scripted run.

Use the **Current PhiX174 GDPO score definitions** section in `examples/README.md` as a concise worked example of this writeup, but resolve every setting and source for the selected target rather than copying PhiX-specific thresholds.

Summarize the decision and both artifact paths in `RUNLOG.md`.

For the PhiX profile, distinguish `core_gene_ordered_conservation` from
`accessory_gene_diversification`. The former preserves the existing core order/copy
rules; the latter rewards K/X repertoire changes, excludes partial core matches,
and penalizes accessory copies. It is an RL reward and diagnostic only, not a new
final filter or a claim of exact agreement with Arc's historical cluster counts.
Use the [accessory definition](../../configs/accessory_genes.md) before adapting
families or thresholds. K-family absence requires available search evidence; a
missing artifact is never a deletion reward. AAI includes all hit-bearing proteins.
