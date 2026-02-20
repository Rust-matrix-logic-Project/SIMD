use core::arch::x86_64::*;
use crate::Tensor;

pub unsafe fn add_tensor(mut tensor1: Tensor, tensor2: Tensor) -> Tensor{
    let data1 = tensor1.data.len();
    let data2 = tensor2.data.len();
    if data1 != data2 {
        eprintln!("配列の要素が一致していません。");
        return Tensor { rows:tensor1.rows, cols:tensor1.cols, data:tensor1.data }
    }
    let ptr1 = tensor1.data.as_mut_ptr();
    let ptr2 = tensor2.data.as_ptr();
    unsafe {
        let mut i = 0;
        while i + 8 <= data1 {
            let data_ptr1 = ptr1.add(i) as *mut __m256i;
            let data_ptr2 = ptr2.add(i) as *const __m256i;
            let load_ptr1 = _mm256_loadu_si256(data_ptr1);
            let load_ptr2 = _mm256_loadu_si256(data_ptr2);

            let add_result = _mm256_add_epi32(load_ptr1, load_ptr2);
            i += 8;
            _mm256_storeu_si256(data_ptr1, add_result);
        }
        while i < data1 {
            *ptr1.add(i) += *ptr2.add(i);
            i += 1;
        }
    }
    tensor1
}