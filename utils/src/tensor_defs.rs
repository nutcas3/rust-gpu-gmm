use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Tiled { tile_m: usize, tile_n: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    pub rows: usize,
    pub cols: usize,
}

impl TensorShape {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
    
    pub const fn size(&self) -> usize {
        self.rows * self.cols
    }
    
    pub const fn is_valid(&self) -> bool {
        self.rows > 0 && self.cols > 0
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.rows, self.cols)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorLayout {
    pub shape: TensorShape,
    pub layout: MemoryLayout,
    pub leading_dim: usize,
}

impl TensorLayout {
    pub fn row_major(rows: usize, cols: usize) -> Self {
        Self {
            shape: TensorShape::new(rows, cols),
            layout: MemoryLayout::RowMajor,
            leading_dim: cols,
        }
    }
    
    pub fn column_major(rows: usize, cols: usize) -> Self {
        Self {
            shape: TensorShape::new(rows, cols),
            layout: MemoryLayout::ColumnMajor,
            leading_dim: rows,
        }
    }
    
    pub fn tiled(rows: usize, cols: usize, tile_m: usize, tile_n: usize) -> Self {
        Self {
            shape: TensorShape::new(rows, cols),
            layout: MemoryLayout::Tiled { tile_m, tile_n },
            leading_dim: cols,
        }
    }
    
    pub fn index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.shape.rows, "Row index out of bounds");
        debug_assert!(col < self.shape.cols, "Column index out of bounds");
        
        match self.layout {
            MemoryLayout::RowMajor => row * self.leading_dim + col,
            MemoryLayout::ColumnMajor => col * self.leading_dim + row,
            MemoryLayout::Tiled { tile_m, tile_n } => {
                let tile_row = row / tile_m;
                let tile_col = col / tile_n;
                let in_tile_row = row % tile_m;
                let in_tile_col = col % tile_n;
                
                let tiles_per_row = (self.shape.cols + tile_n - 1) / tile_n;
                let tile_idx = tile_row * tiles_per_row + tile_col;
                let in_tile_idx = in_tile_row * tile_n + in_tile_col;
                
                tile_idx * (tile_m * tile_n) + in_tile_idx
            }
        }
    }
    
    pub fn row_stride(&self) -> usize {
        match self.layout {
            MemoryLayout::RowMajor => self.leading_dim,
            MemoryLayout::ColumnMajor => 1,
            MemoryLayout::Tiled { tile_m, tile_n } => tile_n,
        }
    }
    
    pub fn col_stride(&self) -> usize {
        match self.layout {
            MemoryLayout::RowMajor => 1,
            MemoryLayout::ColumnMajor => self.leading_dim,
            MemoryLayout::Tiled { tile_m, .. } => tile_m,
        }
    }
    
    pub fn is_gemm_compatible(a: &Self, b: &Self, c: &Self) -> bool {
        a.shape.cols == b.shape.rows
            && a.shape.rows == c.shape.rows
            && b.shape.cols == c.shape.cols
    }
}

impl fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorLayout({}, {:?}, ld={})",
            self.shape, self.layout, self.leading_dim
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub warp_m: usize,
    pub warp_n: usize,
    pub warp_k: usize,
}

impl TileConfig {
    pub const fn ampere_default() -> Self {
        Self {
            tile_m: 128,
            tile_n: 128,
            tile_k: 16,
            warp_m: 32,
            warp_n: 32,
            warp_k: 16,
        }
    }
    
    pub const fn hopper_default() -> Self {
        Self {
            tile_m: 256,
            tile_n: 128,
            tile_k: 64,
            warp_m: 64,
            warp_n: 64,
            warp_k: 16,
        }
    }
    
    pub const fn warps_per_block(&self) -> usize {
        let warps_m = (self.tile_m + self.warp_m - 1) / self.warp_m;
        let warps_n = (self.tile_n + self.warp_n - 1) / self.warp_n;
        warps_m * warps_n
    }
    
    pub const fn threads_per_block(&self) -> usize {
        self.warps_per_block() * 32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_shape() {
        let shape = TensorShape::new(10, 20);
        assert_eq!(shape.rows, 10);
        assert_eq!(shape.cols, 20);
        assert_eq!(shape.size(), 200);
        assert!(shape.is_valid());
    }
    
    #[test]
    fn test_row_major_layout() {
        let layout = TensorLayout::row_major(4, 3);
        assert_eq!(layout.index(0, 0), 0);
        assert_eq!(layout.index(0, 1), 1);
        assert_eq!(layout.index(1, 0), 3);
        assert_eq!(layout.index(1, 1), 4);
    }
    
    #[test]
    fn test_column_major_layout() {
        let layout = TensorLayout::column_major(4, 3);
        assert_eq!(layout.index(0, 0), 0);
        assert_eq!(layout.index(1, 0), 1);
        assert_eq!(layout.index(0, 1), 4);
        assert_eq!(layout.index(1, 1), 5);
    }
    
    #[test]
    fn test_gemm_compatibility() {
        let a = TensorLayout::row_major(10, 20);
        let b = TensorLayout::row_major(20, 30);
        let c = TensorLayout::row_major(10, 30);
        
        assert!(TensorLayout::is_gemm_compatible(&a, &b, &c));
    }
    
    #[test]
    fn test_tile_config() {
        let config = TileConfig::ampere_default();
        assert_eq!(config.tile_m, 128);
        assert_eq!(config.tile_n, 128);
        assert!(config.threads_per_block() > 0);
    }
}
