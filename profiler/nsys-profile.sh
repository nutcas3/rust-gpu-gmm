#!/bin/bash
set -e

BINARY="../target/release/gemm-optimizer"
OUTPUT_DIR="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/gemm_timeline_${TIMESTAMP}.nsys-rep"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "NVIDIA Nsight Systems Profiling"
echo "=========================================="
echo ""

if ! command -v nsys &> /dev/null; then
    echo "Error: NVIDIA Nsight Systems (nsys) not found in PATH"
    exit 1
fi

if [ ! -f "${BINARY}" ]; then
    echo "Error: Binary not found at ${BINARY}"
    exit 1
fi

echo "Starting timeline profiling..."
echo ""

# Profile with Nsight Systems
# --trace: What to trace (cuda, nvtx, osrt)
# --stats: Generate statistics
# --force-overwrite: Overwrite existing reports
nsys profile \
    --trace cuda,nvtx,osrt \
    --stats true \
    --force-overwrite true \
    --output "${REPORT_FILE}" \
    "${BINARY}"

echo ""
echo "=========================================="
echo "Timeline Profiling Complete!"
echo "=========================================="
echo ""
echo "Report saved to: ${REPORT_FILE}.qdrep"
echo ""
echo "To view the report:"
echo "  nsys-ui ${REPORT_FILE}.qdrep"
echo ""
