#!/bin/bash
# ============================================================================
# run.sh — Process all CCTV clips and emit events
# ============================================================================
#
# Usage:
#   ./run.sh [--input-dir /path/to/clips] [--output-dir /path/to/output]
#            [--api-url http://localhost:8000] [--synthetic N]
#
# This script:
#   1. Discovers all video files in the input directory
#   2. Maps filenames to store_id and camera_id based on naming convention
#   3. Runs detect.py for each clip
#   4. Merges all output JSONL files into a single events file
#   5. Optionally ingests into the API
#
# Naming convention for video files:
#   {STORE_ID}_{CAMERA_ID}_{timestamp}.{ext}
#   Example: STORE_BLR_002_CAM_ENTRY_01_20260303T1422.mp4
#
# If no videos are found, runs in synthetic mode.
# ============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
#  Defaults                                                                   #
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="${PROJECT_DIR}/data/clips"
OUTPUT_DIR="${PROJECT_DIR}/output"
API_URL=""
SYNTHETIC=0
STORE_LAYOUT="${PROJECT_DIR}/data/store_layout.json"
FPS=5
DETECTOR="dummy"

# --------------------------------------------------------------------------- #
#  Parse arguments                                                            #
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)
            INPUT_DIR="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --api-url)
            API_URL="$2"; shift 2 ;;
        --synthetic)
            SYNTHETIC="$2"; shift 2 ;;
        --fps)
            FPS="$2"; shift 2 ;;
        --detector)
            DETECTOR="$2"; shift 2 ;;
        --help|-h)
            head -30 "$0" | tail -25
            exit 0 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- #
#  Setup                                                                      #
# --------------------------------------------------------------------------- #
mkdir -p "$OUTPUT_DIR"
MERGED_OUTPUT="${OUTPUT_DIR}/all_events.jsonl"
: > "$MERGED_OUTPUT"  # Clear/create merged output file

echo "=============================================="
echo "  Store Intelligence — Detection Pipeline"
echo "=============================================="
echo "  Input:    ${INPUT_DIR}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Detector: ${DETECTOR}"
echo "  FPS:      ${FPS}"
echo "  API:      ${API_URL:-disabled}"
echo "=============================================="

# --------------------------------------------------------------------------- #
#  Discover and process video files                                           #
# --------------------------------------------------------------------------- #
VIDEO_EXTS="mp4 avi mov mkv wmv flv"
FOUND_VIDEOS=0

if [[ -d "$INPUT_DIR" ]]; then
    for ext in $VIDEO_EXTS; do
        for video in "$INPUT_DIR"/*."$ext" "$INPUT_DIR"/*."${ext^^}" 2>/dev/null; do
            [[ -f "$video" ]] || continue
            FOUND_VIDEOS=$((FOUND_VIDEOS + 1))

            filename="$(basename "$video")"
            echo ""
            echo ">>> Processing: $filename"

            # Parse store_id and camera_id from filename
            # Convention: {STORE_ID}_{CAM_...}_{rest}.ext
            # Example: STORE_BLR_002_CAM_ENTRY_01_20260303.mp4
            if [[ "$filename" =~ ^(STORE_[A-Z]+_[0-9]+)_(CAM_[A-Z]+_[0-9]+) ]]; then
                STORE_ID="${BASH_REMATCH[1]}"
                CAMERA_ID="${BASH_REMATCH[2]}"
            else
                STORE_ID="STORE_BLR_002"
                CAMERA_ID="CAM_ENTRY_01"
                echo "    (Could not parse store/camera from filename, using defaults)"
            fi

            OUTPUT_FILE="${OUTPUT_DIR}/${filename%.*}_events.jsonl"

            # Build command
            CMD="python -m pipeline.detect"
            CMD+=" --input \"$video\""
            CMD+=" --store $STORE_ID"
            CMD+=" --camera $CAMERA_ID"
            CMD+=" --output \"$OUTPUT_FILE\""
            CMD+=" --detector $DETECTOR"
            CMD+=" --fps $FPS"
            CMD+=" --store-layout \"$STORE_LAYOUT\""

            if [[ -n "$API_URL" ]]; then
                CMD+=" --api-url $API_URL"
            fi

            echo "    Store:  $STORE_ID"
            echo "    Camera: $CAMERA_ID"
            echo "    Output: $OUTPUT_FILE"

            # Run the pipeline
            eval "$CMD"

            # Append to merged output
            if [[ -f "$OUTPUT_FILE" ]]; then
                cat "$OUTPUT_FILE" >> "$MERGED_OUTPUT"
                LINES=$(wc -l < "$OUTPUT_FILE")
                echo "    ✓ Emitted $LINES events"
            fi
        done
    done
fi

# --------------------------------------------------------------------------- #
#  Synthetic fallback                                                         #
# --------------------------------------------------------------------------- #
if [[ $FOUND_VIDEOS -eq 0 ]]; then
    if [[ $SYNTHETIC -eq 0 ]]; then
        SYNTHETIC=300
    fi
    echo ""
    echo ">>> No video files found — running synthetic mode ($SYNTHETIC frames)"

    OUTPUT_FILE="${OUTPUT_DIR}/synthetic_events.jsonl"
    CMD="python -m pipeline.detect"
    CMD+=" --synthetic $SYNTHETIC"
    CMD+=" --store STORE_BLR_002"
    CMD+=" --camera CAM_ENTRY_01"
    CMD+=" --output \"$OUTPUT_FILE\""
    CMD+=" --detector $DETECTOR"
    CMD+=" --fps $FPS"
    CMD+=" --store-layout \"$STORE_LAYOUT\""

    if [[ -n "$API_URL" ]]; then
        CMD+=" --api-url $API_URL"
    fi

    eval "$CMD"

    if [[ -f "$OUTPUT_FILE" ]]; then
        cat "$OUTPUT_FILE" >> "$MERGED_OUTPUT"
        LINES=$(wc -l < "$OUTPUT_FILE")
        echo "    ✓ Emitted $LINES events"
    fi
fi

# --------------------------------------------------------------------------- #
#  Summary                                                                    #
# --------------------------------------------------------------------------- #
TOTAL_LINES=$(wc -l < "$MERGED_OUTPUT")
echo ""
echo "=============================================="
echo "  Pipeline Complete"
echo "  Total events: $TOTAL_LINES"
echo "  Merged output: $MERGED_OUTPUT"
echo "=============================================="

# Optional: POST merged events to API
if [[ -n "$API_URL" && -f "$MERGED_OUTPUT" && $TOTAL_LINES -gt 0 ]]; then
    echo "  Events were POSTed to $API_URL during processing"
fi
