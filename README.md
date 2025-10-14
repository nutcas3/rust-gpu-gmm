# Rust GEMM Microkernel Optimizer 🦀

A high-performance Matrix Multiplication (GEMM) microkernel implementation in Rust, targeting NVIDIA Tensor Cores on Hopper and Blackwell architectures.

## Project Goals

- **Performance**: Achieve near-peak TFLOPS on modern NVIDIA GPUs using Tensor Cores
- **Safety**: Leverage Rust's memory safety guarantees for host-side code
- **Modularity**: Clean separation between host (CPU), device (GPU), and utility code
- **Profiling**: Integrated performance analysis using NVIDIA Nsight Compute

## Project Structure

```
rust-gpu-gemm/
├── src/                    # Host-side (CPU) application
│   ├── main.rs            # Entry point, CLI, benchmarking
│   └── lib.rs             # CUDA context, memory management, kernel launcher
├── cuda-kernel/           # Device-side (GPU) kernel crate
│   ├── src/lib.rs         # GEMM kernel implementations
│   ├── build.rs           # PTX compilation script
│   └── Cargo.toml         # Kernel dependencies
├── utils/                 # Shared utilities
│   └── src/
│       ├── lib.rs         # Public API
│       └── tensor_defs.rs # Tensor layout abstractions (CuTe-inspired)
├── profiler/              # Profiling scripts and results
│   ├── ncu-profile.sh     # Full Nsight Compute profiling
│   ├── ncu-quick.sh       # Quick profiling for iteration
│   └── nsys-profile.sh    # Timeline profiling
└── data/                  # Sample input/output matrices
```

### Components

1. **Host Application** (`src/`): Manages CUDA context, memory allocation, data transfer, and kernel launches using the `cust` crate
2. **CUDA Kernel** (`cuda-kernel/`): Contains optimized GEMM kernels with warp-level matrix operations and shared memory tiling
3. **Utilities** (`utils/`): Reusable tensor layout definitions and data structures for software-hardware co-design
4. **Profiler** (`profiler/`): Scripts and configurations for performance analysis with Nsight Compute/Systems

## Quick Start

### Prerequisites

**Required:**
- **Rust**: 1.70+ with nightly toolchain
- **CUDA Toolkit**: 11.0+ (12.0+ recommended)
- **NVIDIA GPU**: Compute Capability 7.0+ (Volta or newer)

**Optional:**
- **NVIDIA Nsight Compute**: For profiling
- **NVIDIA Nsight Systems**: For timeline analysis

### Installation

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly

# 2. Install Rust-CUDA toolchain
cargo install rustc_codegen_nvvm

# 3. Set environment variables (add to ~/.bashrc or ~/.zshrc)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 4. Verify CUDA installation
nvcc --version
nvidia-smi
```

### Building

```bash
# Build the entire project
cargo build --release

# This will:
# 1. Build the host application
# 2. Compile the CUDA kernel to PTX
# 3. Link everything together
```

### Running

```bash
# Run with default parameters (1024x1024x1024)
cargo run --release

# Expected output:
# 🦀 Rust GEMM Microkernel Optimizer
# ===================================
# 
# Matrix dimensions: M=1024, N=1024, K=1024
# ...
# Performance: ~900+ GFLOPS
# ✓ Results verified successfully!
```

##  Profiling

### Quick Profile
```bash
cd profiler
./ncu-quick.sh
```

### Full Profile with All Metrics
```bash
./ncu-profile.sh
```

### Timeline Analysis
```bash
./nsys-profile.sh
```

### View Results
```bash
# GUI viewer
ncu-ui results/gemm_profile_*.ncu-rep

# Command-line summary
ncu --import results/gemm_profile_*.ncu-rep --page raw
```

## Development

### Project Workflow

```bash
# Build
cargo build --release              # Build everything
cargo build -p cuda-kernel         # Build kernel only
cargo clean                        # Clean build artifacts

# Test
cargo test                         # Run all tests
cargo test -p utils                # Test specific crate
cargo test -- --nocapture          # Show test output

# Profile
cd profiler
./ncu-profile.sh                   # Full profile
./ncu-quick.sh                     # Quick iteration
./nsys-profile.sh                  # Timeline profile

# Debug
RUST_BACKTRACE=1 cargo run         # Show backtrace
cuda-memcheck ./target/release/gemm-optimizer  # Memory check
```

### Optimization Strategy

**Phase 1: Baseline** [done]
- Basic kernel with global memory access
- Correct GEMM computation
- Host-device memory management
- Verification against CPU reference

**Phase 2: Memory Optimization** [in progress]
- [ ] Implement shared memory tiling
- [ ] Optimize memory coalescing
- [ ] Tune tile sizes for target architecture
- [ ] Add prefetching for global memory loads

**Phase 3: Compute Optimization** 
- [ ] Implement WMMA (Warp Matrix Multiply-Accumulate)
- [ ] Optimize register usage
- [ ] Minimize warp divergence
- [ ] Maximize instruction-level parallelism

**Phase 4: Advanced** 🎯
- [ ] Double buffering
- [ ] Async copy for Hopper
- [ ] Multi-GPU support with MPI
- [ ] Auto-tuning

## Architecture Overview

### Memory Hierarchy
```
Global Memory (HBM)      → 3+ TB/s   → Input/Output matrices
    ↓
L2 Cache (40+ MB)        → ~10 TB/s  → Shared across SMs
    ↓
Shared Memory (~200KB)   → ~20 TB/s  → Tile buffers
    ↓
Registers (255 max)      → ~100 TB/s → Accumulators
```

### Kernel Variants

1. **`gemm_kernel`**: Basic implementation with global memory access
2. **`gemm_kernel_tiled`**: Shared memory optimization with hierarchical tiling
3. **`gemm_kernel_wmma`**: Tensor Core acceleration using WMMA intrinsics

### Key Design Patterns

- **RAII**: Automatic resource cleanup for CUDA contexts and memory
- **Type Safety**: Compile-time guarantees for device memory operations
- **Zero-Cost Abstractions**: Hardware-aware layouts without runtime overhead

## Troubleshooting

### PTX compilation failed
**Cause**: `rustc_codegen_nvvm` not installed or CUDA not in PATH

**Solution**:
```bash
cargo install rustc_codegen_nvvm
export PATH=/usr/local/cuda/bin:$PATH
```

### No CUDA-capable device detected
**Cause**: No NVIDIA GPU or driver not installed

**Solution**:
```bash
nvidia-smi  # Check GPU status
# Install/update driver from https://www.nvidia.com/Download/index.aspx
```

### Kernel launch failed
**Cause**: Insufficient GPU memory or invalid launch configuration

**Solution**:
- Reduce matrix size
- Check GPU memory: `nvidia-smi`
- Verify compute capability matches kernel requirements

### Verification failed
**Cause**: Numerical precision issues or kernel bug

**Solution**:
- Check tolerance in verification (default: 1e-3)
- Profile with Nsight Compute to identify issues
- Enable debug output: `RUST_LOG=debug cargo run`

## Learning Resources

### Rust-CUDA
- [Rust-CUDA Book](https://rust-gpu.github.io/Rust-CUDA/)
- [cuda-std Documentation](https://docs.rs/cuda-std/)
- [cust Documentation](https://docs.rs/cust/)

### CUDA Programming
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/)

### GEMM Optimization
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [How to Optimize GEMM](https://siboehm.com/articles/22/CUDA-MMM)
- [Tensor Core Programming](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run profiling to verify performance
5. Submit a pull request

## License

This project is dual-licensed under MIT OR Apache-2.0.

## Acknowledgments

Inspired by NVIDIA's CUTLASS library and the Rust-CUDA ecosystem.
