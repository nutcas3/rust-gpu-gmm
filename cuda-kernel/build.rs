use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    
    let target_dir = manifest_dir.join("target/nvptx64-nvidia-cuda/release");
    std::fs::create_dir_all(&target_dir).expect("Failed to create target directory");
    
    println!("cargo:warning=Building CUDA kernel to PTX...");
    println!("cargo:warning=Output directory: {}", target_dir.display());
    
    let status = Command::new("cargo")
        .args(&[
            "+nightly",
            "rustc",
            "--release",
            "--target=nvptx64-nvidia-cuda",
            "--",
            "-C", "target-cpu=sm_80",
            "-C", "opt-level=3",
        ])
        .current_dir(&manifest_dir)
        .status();
    
    match status {
        Ok(status) if status.success() => {
            println!("cargo:warning=PTX compilation successful");
        }
        Ok(status) => {
            println!("cargo:warning=PTX compilation failed with status: {}", status);
            println!("cargo:warning=Note: This requires rustc_codegen_nvvm to be installed");
            println!("cargo:warning=Install with: cargo install rustc_codegen_nvvm");
        }
        Err(e) => {
            println!("cargo:warning=Failed to execute PTX compilation: {}", e);
            println!("cargo:warning=Ensure CUDA toolkit and rustc_codegen_nvvm are installed");
        }
    }
    
    println!("cargo:rustc-env=PTX_PATH={}", target_dir.join("gemm_kernel.ptx").display());
}
