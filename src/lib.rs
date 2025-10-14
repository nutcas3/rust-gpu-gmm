use anyhow::{Context, Result};
use cust::prelude::*;
use std::path::Path;
use utils::TensorLayout;

pub struct CudaContext {
    _context: Context,
    device: Device,
}

impl CudaContext {
    pub fn new() -> Result<Self> {
        cust::init(CudaFlags::empty())?;
        
        let device = Device::get_device(0)?;
        let _context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device,
        )?;
        
        println!("Initialized CUDA device: {}", device.name()?);
        println!("Compute Capability: {}.{}", 
                 device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?,
                 device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?);
        
        Ok(Self { _context, device })
    }
    
    pub fn device(&self) -> Device {
        self.device
    }
}

pub struct DeviceBuffer<T> {
    buffer: DeviceBox<T>,
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let mut buffer = unsafe { DeviceBuffer::alloc(data.len())? };
        buffer.copy_from_host(data)?;
        Ok(buffer)
    }
    
    pub unsafe fn alloc(len: usize) -> Result<Self> {
        let buffer = cust::memory::malloc::<T>(len)
            .context("Failed to allocate device memory")?;
        Ok(Self { buffer })
    }
    
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        self.buffer.copy_from(data)
            .context("Failed to copy data to device")?;
        Ok(())
    }
    
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<()> {
        self.buffer.copy_to(data)
            .context("Failed to copy data from device")?;
        Ok(())
    }
    
    pub fn as_device_ptr(&self) -> DevicePointer<T> {
        *self.buffer.as_device_ptr()
    }
}

pub struct GemmKernel {
    module: Module,
    stream: Stream,
}

impl GemmKernel {
    pub fn load<P: AsRef<Path>>(ptx_path: P) -> Result<Self> {
        let ptx = std::fs::read_to_string(ptx_path)
            .context("Failed to read PTX file")?;
        
        let module = Module::load_from_string(&ptx)
            .context("Failed to load PTX module")?;
        
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .context("Failed to create CUDA stream")?;
        
        Ok(Self { module, stream })
    }
    
    pub fn launch(
        &self,
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        block_size: (u32, u32, u32),
    ) -> Result<()> {
        let grid_x = (n + block_size.0 - 1) / block_size.0;
        let grid_y = (m + block_size.1 - 1) / block_size.1;
        let grid_size = (grid_x, grid_y, 1);
        
        println!("Launching kernel with grid: {:?}, block: {:?}", grid_size, block_size);
        
        let kernel = self.module.get_function("gemm_kernel")
            .context("Failed to get kernel function")?;
        
        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, self.stream>>>(
                    m,
                    n,
                    k,
                    alpha,
                    a.as_device_ptr(),
                    b.as_device_ptr(),
                    beta,
                    c.as_device_ptr()
                )
            )?;
        }
        
        self.stream.synchronize()
            .context("Kernel execution failed")?;
        
        Ok(())
    }
}

pub fn calculate_block_size(device: Device) -> Result<(u32, u32, u32)> {
    let max_threads_per_block = device
        .get_attribute(DeviceAttribute::MaxThreadsPerBlock)? as u32;
    
    // For GEMM, typically use 16x16 or 32x32 thread blocks
    // To be  tuned based on profiling results
    let block_dim = if max_threads_per_block >= 1024 {
        32
    } else {
        16
    };
    
    Ok((block_dim, block_dim, 1))
}

pub fn verify_gemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &[f32],
    tolerance: f32,
) -> bool {
    let mut c_ref = vec![0.0f32; m * n];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c_ref[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
    
    for i in 0..(m * n) {
        let diff = (c[i] - c_ref[i]).abs();
        if diff > tolerance {
            println!("Mismatch at index {}: GPU={}, CPU={}, diff={}", 
                     i, c[i], c_ref[i], diff);
            return false;
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_context_creation() {
        let ctx = CudaContext::new();
        assert!(ctx.is_ok(), "Failed to create CUDA context");
    }
}
