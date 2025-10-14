#!/bin/bash

set -e

# Configuration
BINARY="../target/release/gemm-optimizer"
OUTPUT_DIR="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/gemm_profile_${TIMESTAMP}.ncu-rep"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "NVIDIA Nsight Compute Profiling"
echo "=========================================="
echo ""
echo "Binary: ${BINARY}"
echo "Report: ${REPORT_FILE}"
echo ""

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "Error: NVIDIA Nsight Compute (ncu) not found in PATH"
    echo "Please install NVIDIA Nsight Compute and add it to your PATH"
    exit 1
fi

# Check if binary exists
if [ ! -f "${BINARY}" ]; then
    echo "Error: Binary not found at ${BINARY}"
    echo "Please build the project first: cargo build --release"
    exit 1
fi

echo "Starting profiling..."
echo ""

# Run comprehensive profiling
# --set full: Collect all available metrics
# --kernel-regex: Target specific kernel(s)
# --target-processes all: Profile all processes
# --export: Save report to file
ncu \
    --set full \
    --kernel-regex "gemm_kernel" \
    --target-processes all \
    --export "${REPORT_FILE}" \
    --force-overwrite \
    "${BINARY}"

echo ""
echo "=========================================="
echo "Profiling Complete!"
echo "=========================================="
echo ""
echo "Report saved to: ${REPORT_FILE}"
echo ""
echo "To view the report:"
echo "  ncu-ui ${REPORT_FILE}"
echo ""
echo "To generate a text summary:"
echo "  ncu --import ${REPORT_FILE} --page raw"
echo ""

# Generate a quick text summary
echo "Generating quick summary..."
ncu --import "${REPORT_FILE}" --page raw > "${OUTPUT_DIR}/summary_${TIMESTAMP}.txt" 2>&1 || true

echo "Summary saved to: ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
