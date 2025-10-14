use anyhow::Result;
use rust_gpu_gemm::{
    calculate_block_size, CudaContext, DeviceBuffer, GemmKernel, verify_gemm,
};
use std::time::Instant;

struct GemmConfig {
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    verify: bool,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self {
            m: 1024,
            n: 1024,
            k: 1024,
            alpha: 1.0,
            beta: 0.0,
            verify: true,
        }
    }
}

fn main() -> Result<()> {
    println!("🦀 Rust GEMM Microkernel Optimizer");
    println!("===================================\n");
    
    let config = GemmConfig::default();
    
    println!("Matrix dimensions: M={}, N={}, K={}", config.m, config.n, config.k);
    println!("Alpha={}, Beta={}\n", config.alpha, config.beta);
    
    println!("Initializing CUDA...");
    let ctx = CudaContext::new()?;
    let block_size = calculate_block_size(ctx.device())?;
    println!("Using block size: {:?}\n", block_size);
    
    println!("Allocating and initializing matrices...");
    let a_host = vec![1.0f32; config.m * config.k];
    let b_host = vec![1.0f32; config.k * config.n];
    let c_host = vec![0.0f32; config.m * config.n];
    
    let d_a = DeviceBuffer::from_slice(&a_host)?;
    let d_b = DeviceBuffer::from_slice(&b_host)?;
    let mut d_c = DeviceBuffer::from_slice(&c_host)?;
    
    println!("Host memory: {} MB", 
             (a_host.len() + b_host.len() + c_host.len()) * 4 / 1_000_000);
    println!("Device memory allocated successfully\n");
    
    println!("Loading PTX kernel...");
    let kernel_path = "cuda-kernel/target/nvptx64-nvidia-cuda/release/gemm_kernel.ptx";
    let kernel = GemmKernel::load(kernel_path)?;
    println!("Kernel loaded successfully\n");
    
    println!("Performing warm-up run...");
    kernel.launch(
        config.m as u32,
        config.n as u32,
        config.k as u32,
        config.alpha,
        &d_a,
        &d_b,
        config.beta,
        &mut d_c,
        block_size,
    )?;
    println!("Warm-up completed\n");
    
    println!("Running benchmark (5 iterations)...");
    let num_runs = 5;
    let mut total_time = 0.0;
    
    for i in 0..num_runs {
        let start = Instant::now();
        
        kernel.launch(
            config.m as u32,
            config.n as u32,
            config.k as u32,
            config.alpha,
            &d_a,
            &d_b,
            config.beta,
            &mut d_c,
            block_size,
        )?;
        
        let elapsed = start.elapsed().as_secs_f64();
        total_time += elapsed;
        
        println!("  Run {}: {:.3} ms", i + 1, elapsed * 1000.0);
    }
    
    let avg_time = total_time / num_runs as f64;
    let gflops = (2.0 * config.m as f64 * config.n as f64 * config.k as f64) 
                 / (avg_time * 1e9);
    
    println!("\nPerformance Results:");
    println!("  Average time: {:.3} ms", avg_time * 1000.0);
    println!("  Performance: {:.2} GFLOPS", gflops);
    
    if config.verify {
        println!("\nVerifying results...");
        let mut c_result = vec![0.0f32; config.m * config.n];
        d_c.copy_to_host(&mut c_result)?;
        
        let is_correct = verify_gemm(
            config.m,
            config.n,
            config.k,
            config.alpha,
            &a_host,
            &b_host,
            config.beta,
            &c_result,
            1e-3,
        );
        
        if is_correct {
            println!("✓ Results verified successfully!");
        } else {
            println!("✗ Verification failed!");
        }
    }
    
    println!("\n🎉 GEMM optimization complete!");
    
    Ok(())
}
