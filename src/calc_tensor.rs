use core::arch::x86_64::*;
use crate::{Tensor, create_tensor::create_tensor};

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

            let mul_result = _mm256_add_epi32(load_ptr1, load_ptr2);
            i += 8;
            _mm256_storeu_si256(data_ptr1, mul_result);
        }
        while i < data1 {
            *ptr1.add(i) *= *ptr2.add(i);
            i += 1;
        }
    }
    tensor1
}

pub unsafe fn mul_tensor_elementwise(mut tensor1: Tensor, tensor2: Tensor) -> Tensor{
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

            let add_result = _mm256_mullo_epi32(load_ptr1, load_ptr2);
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

pub unsafe fn dot_tensor(tensor1: Tensor, tensor2: Tensor) -> Tensor{
    let shape1 = tensor1.cols;
    let shape2 = tensor2.rows;
    if  shape1 != shape2{
        eprintln!("配列の形状が一致しません。");
        return tensor1;
    }
    unsafe {
        let mut dot_result = create_tensor(tensor1.rows,tensor2.cols);
        for i in 0..tensor1.rows     {
            for k in 0..tensor2.rows {
                let val_a = tensor1.data[i * tensor1.cols + k];
                let result_a = _mm256_set1_epi32(val_a);
                let mut j = 0;
                while j + 8 <= tensor2.cols {
                    let b = tensor2.data.as_ptr().add(k * tensor2.cols + j);
                    let c = dot_result.data.as_mut_ptr().add(i * tensor2.cols + j);
                    let load_b = _mm256_loadu_si256(b as *const __m256i);
                    let load_c = _mm256_loadu_si256(c  as *const __m256i);
                    let result_b = _mm256_mullo_epi32(result_a, load_b);
                    let add_result = _mm256_add_epi32(result_b, load_c);    
                    _mm256_storeu_si256(c as *mut __m256i, add_result);
                    
                    j += 8;
                }
                while j < tensor2.cols {
                    let val_b = tensor2.data[k * tensor1.cols + j];
                    dot_result.data[i * tensor2.cols + j] += val_a * val_b;

                    j += 1;
                }
            }
        }
        dot_result
    }
    
}

pub fn normal_dot_tensor(tensor1: &Tensor, tensor2: &Tensor) -> Tensor {
    let mut dot_result = unsafe { crate::create_tensor::create_tensor(tensor1.rows, tensor2.cols) };
    
    for i in 0..tensor1.rows {
        for k in 0..tensor1.cols {
            let val_a = tensor1.data[i * tensor1.cols + k];
            for j in 0..tensor2.cols {
                let val_b = tensor2.data[k * tensor2.cols + j];
                dot_result.data[i * tensor2.cols + j] += val_a * val_b;
            }
        }
    }
    dot_result
}