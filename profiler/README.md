# Profiler Scripts and Results

This directory contains profiling scripts and stores profiling results for the GEMM kernel.

## Scripts

### ncu-profile.sh
Full profiling with NVIDIA Nsight Compute. Collects comprehensive metrics including:
- SM utilization and occupancy
- Memory bandwidth and cache hit rates
- Instruction mix and throughput
- Warp execution efficiency
- Tensor Core utilization

**Usage:**
```bash
./ncu-profile.sh
```

**Output:** `results/gemm_profile_<timestamp>.ncu-rep`

### ncu-quick.sh
Quick profiling with essential metrics only. Faster iteration for optimization cycles.

**Usage:**
```bash
./ncu-quick.sh
```

**Output:** `results/gemm_quick_<timestamp>.ncu-rep`

### nsys-profile.sh
Timeline profiling with NVIDIA Nsight Systems. Useful for:
- Kernel launch overhead analysis
- Host-device synchronization
- Multi-stream execution
- CPU-GPU interaction

**Usage:**
```bash
./nsys-profile.sh
```

**Output:** `results/gemm_timeline_<timestamp>.nsys-rep`

## Viewing Reports

### Nsight Compute Reports
```bash
# GUI viewer
ncu-ui results/gemm_profile_<timestamp>.ncu-rep

# Command-line summary
ncu --import results/gemm_profile_<timestamp>.ncu-rep --page raw
```

### Nsight Systems Reports
```bash
# GUI viewer
nsys-ui results/gemm_timeline_<timestamp>.nsys-rep.qdrep

# Command-line statistics
nsys stats results/gemm_timeline_<timestamp>.nsys-rep.qdrep
```

## Key Metrics to Monitor

### Performance Metrics
- **GFLOPS**: Achieved floating-point operations per second
- **Memory Bandwidth**: GB/s achieved vs. theoretical peak
- **Occupancy**: Active warps vs. maximum possible
- **Tensor Core Utilization**: Percentage of time Tensor Cores are active

### Optimization Targets
- **Coalescing Efficiency**: Should be >80% for optimal memory access
- **Cache Hit Rate**: L1/L2 cache effectiveness
- **Warp Execution Efficiency**: Minimize divergence and stalls
- **SM Utilization**: Keep SMs busy with sufficient work

## Profiling Workflow

1. **Baseline**: Run full profile to establish baseline metrics
2. **Identify Bottlenecks**: Look for low utilization or high stall reasons
3. **Optimize**: Modify kernel based on profiling insights
4. **Quick Verify**: Use quick profile to verify improvements
5. **Full Verify**: Run full profile to confirm optimization
6. **Iterate**: Repeat until performance targets are met

## Results Directory

The `results/` directory stores all profiling reports. Reports are timestamped to track optimization progress over time.

**Note:** `.ncu-rep` and `.nsys-rep` files can be large. Consider adding them to `.gitignore` if not needed in version control.
