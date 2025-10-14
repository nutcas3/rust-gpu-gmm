#!/bin/bash
set -e

BINARY="../target/release/gemm-optimizer"
OUTPUT_DIR="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${OUTPUT_DIR}/gemm_quick_${TIMESTAMP}.ncu-rep"

mkdir -p "${OUTPUT_DIR}"

echo "Running quick profile (essential metrics only)..."

# Profile with basic metrics
ncu \
    --metrics sm__cycles_elapsed.avg,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
    --kernel-regex "gemm_kernel" \
    --export "${REPORT_FILE}" \
    --force-overwrite \
    "${BINARY}"

echo ""
echo "Quick profile complete: ${REPORT_FILE}"
