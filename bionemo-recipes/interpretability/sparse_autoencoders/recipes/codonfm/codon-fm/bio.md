# Biology Primer for CodonFM SAE Work

Everything an MLE needs to know about codons, amino acids, and nucleotides
to work on this project — especially coming from ESM2/protein SAE experience.

---

## The Central Dogma (30-second version)

```
DNA  ──transcription──>  mRNA  ──translation──>  Protein
(nucleotides)           (nucleotides)           (amino acids)
```

CodonFM operates at the DNA/mRNA level. ESM2 operates at the protein level.

---

## Nucleotides

The atomic units of DNA/RNA. There are 4 DNA bases:

| Base | Letter | Name |
|------|--------|------|
| A | Adenine | Purine |
| T | Thymine | Pyrimidine (DNA only) |
| G | Guanine | Purine |
| C | Cytosine | Pyrimidine |

RNA uses **U (Uracil)** instead of T. CodonFM supports both via `seq_type="dna"` or `"rna"`.

A DNA sequence is just a string of these: `ATGAAAGCCTTTGAC...`

---

## Codons

A **codon** is a contiguous triplet of nucleotides. Codons are the fundamental unit
that the ribosome reads during translation.

- 4 bases × 4 bases × 4 bases = **64 possible codons**
- Each codon encodes exactly one amino acid (or a stop signal)
- Codons are read in a fixed, non-overlapping reading frame starting from a start codon

Example: `ATGAAAGCC` → `ATG | AAA | GCC` → Met-Lys-Ala

**This is exactly how CodonFM tokenizes.** Each token = one codon = 3 nucleotides.

---

## Amino Acids

Proteins are chains of amino acids. There are **20 standard amino acids**:

| 1-Letter | 3-Letter | Name | # of Codons |
|----------|----------|------|-------------|
| A | Ala | Alanine | 4 |
| R | Arg | Arginine | 6 |
| N | Asn | Asparagine | 2 |
| D | Asp | Aspartic acid | 2 |
| C | Cys | Cysteine | 2 |
| E | Glu | Glutamic acid | 2 |
| Q | Gln | Glutamine | 2 |
| G | Gly | Glycine | 4 |
| H | His | Histidine | 2 |
| I | Ile | Isoleucine | 3 |
| L | Leu | Leucine | 6 |
| K | Lys | Lysine | 2 |
| M | Met | Methionine | 1 |
| F | Phe | Phenylalanine | 2 |
| P | Pro | Proline | 4 |
| S | Ser | Serine | 6 |
| T | Thr | Threonine | 4 |
| W | Trp | Tryptophan | 1 |
| Y | Tyr | Tyrosine | 2 |
| V | Val | Valine | 4 |
| * | --- | Stop | 3 |

**Total: 64 codons → 20 amino acids + 1 stop signal.**

---

## The Genetic Code: Codon → Amino Acid Mapping

This is the critical table. It is **many-to-one, NOT one-to-one.**

```
                    Second Position
              U         C         A         G
        ┌─────────┬─────────┬─────────┬─────────┐
    U   │ UUU  F  │ UCU  S  │ UAU  Y  │ UGU  C  │  U
        │ UUC  F  │ UCC  S  │ UAC  Y  │ UGC  C  │  C
 F      │ UUA  L  │ UCA  S  │ UAA  *  │ UGA  *  │  A   T
 i      │ UUG  L  │ UCG  S  │ UAG  *  │ UGG  W  │  G   h
 r      ├─────────┼─────────┼─────────┼─────────┤       i
 s      │ CUU  L  │ CCU  P  │ CAU  H  │ CGU  R  │  U   r
 t  C   │ CUC  L  │ CCC  P  │ CAC  H  │ CGC  R  │  C   d
        │ CUA  L  │ CCA  P  │ CAA  Q  │ CGA  R  │  A
 P      │ CUG  L  │ CCG  P  │ CAG  Q  │ CGG  R  │  G   P
 o      ├─────────┼─────────┼─────────┼─────────┤       o
 s  A   │ AUU  I  │ ACU  T  │ AAU  N  │ AGU  S  │  U   s
 i      │ AUC  I  │ ACC  T  │ AAC  N  │ AGC  S  │  C   i
 t      │ AUA  I  │ ACA  T  │ AAA  K  │ AGA  R  │  A   t
 i      │ AUG  M  │ ACG  T  │ AAG  K  │ AGG  R  │  G   i
 o      ├─────────┼─────────┼─────────┼─────────┤       o
 n  G   │ GUU  V  │ GCU  A  │ GAU  D  │ GGU  G  │  U   n
        │ GUC  V  │ GCC  A  │ GAC  D  │ GGC  G  │  C
        │ GUA  V  │ GCA  A  │ GAA  E  │ GGA  G  │  A
        │ GUG  V  │ GCG  A  │ GAG  E  │ GGG  G  │  G
        └─────────┴─────────┴─────────┴─────────┘
```

### Key properties of the genetic code

1. **Degenerate (redundant):** Most amino acids are encoded by 2-6 codons.
   Only Met (AUG) and Trp (UGG) have exactly one codon.

2. **Not ambiguous:** Each codon maps to exactly one amino acid. The mapping
   is many-to-one, never one-to-many.

3. **Synonymous codons:** Codons that encode the same amino acid.
   E.g., GCU/GCC/GCA/GCG all encode Alanine. These are interchangeable
   at the protein level but NOT at the DNA level.

4. **Start codon:** AUG (Methionine) — signals the start of translation.

5. **Stop codons:** UAA, UAG, UGA — signal the end of translation.
   CodonFM maps these to `*` in its amino acid lookup.

---

## Why Degeneracy Matters for CodonFM vs ESM2

This is the single most important concept for this project:

```
ESM2 sees:       ... Ala - Lys - Gly ...    (amino acid level)
CodonFM sees:    ... GCU - AAA - GGC ...    (codon level)
                 or  GCC - AAG - GGG ...    (same protein, different codons!)
```

**ESM2 loses synonymous codon information.** Two DNA sequences that encode the
exact same protein look identical to ESM2 but different to CodonFM.

What CodonFM can see that ESM2 cannot:
- **Codon usage bias:** Organisms prefer certain synonymous codons. E.g., E. coli
  prefers GCG for Alanine while humans prefer GCC. This affects translation speed.
- **tRNA adaptation:** Codon choice reflects tRNA availability in the organism.
- **mRNA secondary structure:** Synonymous mutations can change mRNA folding.
- **Translational pausing:** Rare codons cause ribosome slowdowns, which can
  affect co-translational protein folding.
- **CpG dinucleotides:** CG pairs at codon boundaries have regulatory significance
  (methylation targets). ESM2 can't see nucleotide-level patterns.

---

## CodonFM Tokenization (How It Works in the Code)

The tokenizer lives at `src/tokenizer/tokenizer.py`. Vocab = 69 tokens:

```
Token IDs 0-4:   <CLS>, <SEP>, <UNK>, <PAD>, <MASK>   (special tokens)
Token IDs 5-68:  AAA, AAC, AAG, AAT, ACA, ..., TTT     (64 codons, lexicographic)
```

Tokenization pipeline:
```
Raw DNA:    "ATGAAAGCCTTTGAC"
                ↓ (chunk into triplets)
Codons:     ["ATG", "AAA", "GCC", "TTT", "GAC"]
                ↓ (map to token IDs)
Token IDs:  [<CLS>, ATG_id, AAA_id, GCC_id, TTT_id, GAC_id, <SEP>, <PAD>, ...]
```

No BPE, no subword tokenization. Clean 1:1 codon-to-token mapping.

---

## Position Mapping: Codons ↔ Nucleotides ↔ Amino Acids

This matters for visualization and annotation alignment:

```
Nucleotide index:  0  1  2  |  3  4  5  |  6  7  8  |  9 10 11  | 12 13 14
Nucleotides:       A  T  G  |  A  A  A  |  G  C  C  |  T  T  T  |  G  A  C
                   ─────────   ─────────   ─────────   ─────────   ─────────
Codon index:          0            1            2            3            4
Codon:               ATG          AAA          GCC          TTT          GAC
                   ─────────   ─────────   ─────────   ─────────   ─────────
Amino acid index:     0            1            2            3            4
Amino acid:          Met          Lys          Ala          Phe          Asp
```

The mapping is positionally aligned:
- **Codon i** = nucleotides `[3i, 3i+1, 3i+2]`
- **Codon i** = amino acid `i` (1:1 positional correspondence)
- **SAE feature at position i** (after removing CLS/SEP) = codon i = amino acid i

This means:
- You can always translate codon activations to amino acid positions
- AlphaFold structures (indexed by amino acid / residue) can be reused
- Codon position `k` maps to residue `k` in the protein structure

---

## Training Setup in CodonFM

### Encodon (Bidirectional, MLM — the one relevant for SAE)

```
Task:           Masked Language Modeling (same concept as BERT/ESM2)
Input:          Codon token sequence with 15% of codons masked
Target:         Predict the original codon at masked positions
Loss:           Cross-entropy over 69-token vocab (at masked positions only)
Context length: 2048 codons = 6,144 nucleotides
```

Masking strategy (standard BERT-style):
- 80% of masked positions → replaced with `<MASK>` token
- 10% → replaced with a random codon
- 10% → kept as original

### Decodon (Autoregressive, CLM)

```
Task:           Next-codon prediction (like GPT)
Input:          Organism token + codon sequence
Target:         Predict next codon autoregressively
Loss:           Cross-entropy over vocab (shifted by 1)
Extra:          Organism conditioning token prepended (e.g., <9606> for human)
```

For SAE work, you'll use **Encodon** (bidirectional representations are richer).

---

## Annotation Landscape: ESM2 vs CodonFM

### ESM2 SAE annotations (what you have now)

Per-residue annotations from UniProt:
- Active sites, binding sites, domains, motifs
- Post-translational modifications (phosphorylation, glycosylation)
- Secondary structure (helix, strand, turn)
- Disulfide bonds, signal peptides, transmembrane regions

These are **protein-level** annotations. They describe what the protein does.

### CodonFM SAE annotations (what you'd need)

Per-codon or per-region annotations from genomic databases. These would describe
what the **DNA** does, not just what protein it encodes:

| Annotation Type | Source | Granularity | Description |
|----------------|--------|-------------|-------------|
| Codon usage bias | Kazusa, CoCoPUTs | Per-codon | How common is this codon choice in this organism |
| Codon Adaptation Index (CAI) | Computed | Per-codon | Optimality score for translation efficiency |
| tRNA Adaptation Index (tAI) | GtRNAdb | Per-codon | Match to available tRNAs |
| CpG islands | UCSC Genome Browser | Per-nucleotide pair | Methylation-prone dinucleotides at codon boundaries |
| Synonymous vs nonsynonymous | Computed | Per-codon | Is this codon conserved across species? (dN/dS) |
| Exon/intron boundaries | Ensembl, GENCODE | Per-region | Splice site proximity (if working with pre-mRNA) |
| Rare codon clusters | Computed | Per-codon | Stretches of low-frequency codons (translational pauses) |
| mRNA secondary structure | RNAfold, ViennaRNA | Per-nucleotide | Local folding energy |
| Kozak sequence context | Known consensus | Per-position (near start) | Translation initiation strength |

Note: **there is no standardized codon annotation database equivalent to UniProt
per-residue features.** You'll likely need to compute many of these annotations
yourself from sequence data and codon usage tables.

---

## AlphaFold Structure Reuse

**Yes, you can reuse the AlphaFold/Mol* rendering from the ESM2 dashboard.**

Why it works:
- AlphaFold predicts **protein** structures, indexed by UniProt accession
- Each codon position maps 1:1 to an amino acid residue position
- So codon SAE activations at position `k` → residue `k` in the AlphaFold structure

What you'd change:
```
ESM2 dashboard:    activation[residue_k] → color residue k
CodonFM dashboard: activation[codon_k]   → color residue k  (same residue!)
```

The Mol* viewer, CIF file fetching, and color-mapping logic all stay the same.
The only difference is the **sequence display** — you'd show the DNA codon sequence
alongside (or instead of) the amino acid sequence.

Caveat: If a CodonFM SAE feature captures something purely DNA-level (e.g., codon
usage bias, mRNA structure), coloring a protein structure may not be the most
meaningful visualization. You might also want:
- A linear codon sequence viewer (color each triplet)
- A codon usage heatmap
- mRNA secondary structure overlay

---

## Glossary

| Term | Definition |
|------|-----------|
| **Codon** | 3-nucleotide unit; the fundamental token in CodonFM |
| **Synonymous codons** | Different codons encoding the same amino acid |
| **Degeneracy** | The redundancy of the genetic code (64 codons → 20 AAs) |
| **Codon usage bias (CUB)** | Organism-specific preference for certain synonymous codons |
| **CAI** | Codon Adaptation Index — measures how "optimal" codon choices are |
| **tAI** | tRNA Adaptation Index — measures match to tRNA pool |
| **dN/dS (ω)** | Ratio of nonsynonymous to synonymous substitution rates; measures selection pressure |
| **Reading frame** | The grouping of nucleotides into codons; shifted by 1 or 2 bases = completely different protein |
| **CDS** | Coding DNA Sequence — the part of a gene that encodes a protein |
| **ORF** | Open Reading Frame — a stretch of codons from start to stop codon |
| **CpG** | Cytosine-Guanine dinucleotide; methylation target with regulatory function |
| **Wobble position** | Third nucleotide of a codon; most synonymous variation occurs here |
| **Missense mutation** | A codon change that changes the amino acid |
| **Synonymous (silent) mutation** | A codon change that does NOT change the amino acid |
| **Nonsense mutation** | A codon change that creates a premature stop codon |
