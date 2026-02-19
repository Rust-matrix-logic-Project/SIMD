use core::arch::x86_64::*;
use crate::Tensor;

pub unsafe fn create_tensor(rows: usize, cols: usize) -> Tensor{

    let size: usize = rows * cols;
    let buf = vec![0i32; size];

    Tensor{
        rows: rows,
        cols: cols,
        data: buf
    }

}
#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn tensor_calc(mut data1:Tensor, data2: i32) -> Tensor{
    let x = data1.data.as_mut_ptr();
    let len = data1.data.len();
    if len < 8 {
        eprint!("配列の要素が不足しています。");
        return Tensor { rows:data1.rows, cols:data1.cols, data:data1.data };
    }
        unsafe {
            let add =  _mm256_set1_epi32(data2);
            let mut i = 0;
            while i + 8 <= len{
                let current_ptr = x.add(i) as *mut __m256i;
                let load_ptr = _mm256_loadu_si256(current_ptr);
                let add_result= _mm256_add_epi32(load_ptr, add);
                i += 8;
                _mm256_storeu_si256(current_ptr, add_result);
            }
            while i < len{
                *x.add(i) += data2;
                i += 1;
            }
        }
        
        data1
}