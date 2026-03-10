#!/bin/bash
# Simple script to optimize a single amino acid sequence using a Decodon model.
#
# Usage: ./optimize_decodon.sh --aa <AA_SEQUENCE> --ckpt <CHECKPOINT_PATH> [OPTIONS]
#
# Required arguments:
#   --aa          Amino acid sequence to optimize
#   --ckpt        Path to the pre-trained Decodon checkpoint
#
# Optional arguments:
#   --organism             Organism for codon usage (human or mouse, default: human)
#   --organism-tokens-file Path to organism tokens file (default: /data/nopathogen_organism_tokens.txt)
#   --out-json             Output JSON file path (default: ./optimized_output.json)
#   --beam-width           Beam width for generation (default: 10)
#   --ga-generations       Number of GA generations (default: 50, set to 0 to disable GA)
#   --ga-population-size   GA population size (default: 100)
#
#   Fitness weights (derived from correlation with expression, notebook 11.9.5):
#   --weight-naturalness   Weight for model naturalness (default: 0.0)
#   --weight-mfe           Weight for MFE (default: 0.33)
#   --weight-te            Weight for TE (default: 1.0)
#   --weight-gc            Weight for GC content (default: 0.75)
#   --weight-u             Weight for U content (default: 0.75)
#   --weight-cai           Weight for CAI (default: 0.67)
#   --weight-cbi           Weight for CBI (default: 0.75)
#   --weight-enc           Weight for ENC (default: 0.5)
#
#   --temperature          Sampling temperature (default: 1.0)
#   --sample               Enable sampling instead of argmax
#   --device               Device to use (e.g., cuda:0, cpu)
#   --seed                 Random seed for reproducibility
#   -h, --help             Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
AA_SEQUENCE=""
CHECKPOINT=""
ORGANISM="human"
ORGANISM_TOKENS_FILE="/data/nopathogen_organism_tokens.txt"
OUT_JSON="./optimized_output.json"
BEAM_WIDTH=10
GA_GENERATIONS=50
GA_POPULATION_SIZE=100

# Fitness weights derived from correlation with expression (notebook 11.9.5)
# |r| values: naturalness=0.75, u=0.73, gc=0.72, cbi=0.72, cai=0.62, enc=0.53, mfe=0.35
WEIGHT_NATURALNESS=0.0
WEIGHT_MFE=0.33
WEIGHT_TE=1.0
WEIGHT_GC=0.75
WEIGHT_U=0.75
WEIGHT_CAI=0.67
WEIGHT_CBI=0.75
WEIGHT_ENC=0.5

TEMPERATURE=1.0
SAMPLE=""
DEVICE=""
SEED=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --aa)
            AA_SEQUENCE="$2"
            shift 2
            ;;
        --ckpt)
            CHECKPOINT="$2"
            shift 2
            ;;
        --organism)
            ORGANISM="$2"
            shift 2
            ;;
        --organism-tokens-file)
            ORGANISM_TOKENS_FILE="$2"
            shift 2
            ;;
        --out-json)
            OUT_JSON="$2"
            shift 2
            ;;
        --beam-width)
            BEAM_WIDTH="$2"
            shift 2
            ;;
        --ga-generations)
            GA_GENERATIONS="$2"
            shift 2
            ;;
        --ga-population-size)
            GA_POPULATION_SIZE="$2"
            shift 2
            ;;
        --weight-naturalness)
            WEIGHT_NATURALNESS="$2"
            shift 2
            ;;
        --weight-mfe)
            WEIGHT_MFE="$2"
            shift 2
            ;;
        --weight-te)
            WEIGHT_TE="$2"
            shift 2
            ;;
        --weight-gc)
            WEIGHT_GC="$2"
            shift 2
            ;;
        --weight-u)
            WEIGHT_U="$2"
            shift 2
            ;;
        --weight-cai)
            WEIGHT_CAI="$2"
            shift 2
            ;;
        --weight-cbi)
            WEIGHT_CBI="$2"
            shift 2
            ;;
        --weight-enc)
            WEIGHT_ENC="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --sample)
            SAMPLE="--sample"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            head -30 "$0" | tail -28
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$AA_SEQUENCE" ]]; then
    echo "Error: --aa (amino acid sequence) is required"
    echo "Usage: $0 --aa <AA_SEQUENCE> --ckpt <CHECKPOINT_PATH>"
    exit 1
fi

if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --ckpt (checkpoint path) is required"
    echo "Usage: $0 --aa <AA_SEQUENCE> --ckpt <CHECKPOINT_PATH>"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "Decodon Codon Optimization"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Organism: $ORGANISM"
echo "Beam width: $BEAM_WIDTH"
echo "GA generations: $GA_GENERATIONS"
echo "Weights (correlation-derived):"
echo "  mfe=$WEIGHT_MFE, te=$WEIGHT_TE, gc=$WEIGHT_GC, u=$WEIGHT_U"
echo "  cai=$WEIGHT_CAI, cbi=$WEIGHT_CBI, enc=$WEIGHT_ENC, naturalness=$WEIGHT_NATURALNESS"
echo "Output: $OUT_JSON"
echo "Sequence length: ${#AA_SEQUENCE} aa"
echo ""

# Build command
CMD=(python "$SCRIPT_DIR/codon_optimize.py"
    --aa "$AA_SEQUENCE"
    --model-type decodon
    --decodon-ckpt "$CHECKPOINT"
    --organism "$ORGANISM"
    --organism-tokens-file "$ORGANISM_TOKENS_FILE"
    --out-json "$OUT_JSON"
    --decode-mode autoregressive
    --beam-width "$BEAM_WIDTH"
    --ga-generations "$GA_GENERATIONS"
    --ga-population-size "$GA_POPULATION_SIZE"
    --weight-naturalness "$WEIGHT_NATURALNESS"
    --weight-mfe "$WEIGHT_MFE"
    --weight-te "$WEIGHT_TE"
    --weight-gc "$WEIGHT_GC"
    --weight-u "$WEIGHT_U"
    --weight-cai "$WEIGHT_CAI"
    --weight-cbi "$WEIGHT_CBI"
    --weight-enc "$WEIGHT_ENC"
    --temperature "$TEMPERATURE"
)

# Add optional arguments
if [[ -n "$DEVICE" ]]; then
    CMD+=(--device "$DEVICE")
fi

if [[ -n "$SAMPLE" ]]; then
    CMD+=($SAMPLE)
fi

if [[ -n "$SEED" ]]; then
    CMD+=(--seed "$SEED")
fi

# Run optimization
"${CMD[@]}"

echo ""
echo "=========================================="
echo "Optimization complete!"
echo "Results saved to: $OUT_JSON"
echo "=========================================="
