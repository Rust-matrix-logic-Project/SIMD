//use core::arch::x86_64::*;
use crate::Tensor;
#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn create_tensor(rows: usize, cols: usize) -> Tensor{

    let size: usize = rows * cols;
    let buf = vec![0i32; size];

    Tensor{
        rows: rows,
        cols: cols,
        data: buf
    }
   
}