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
        while i + 32 <= data1 {
            let data_ptr1 = ptr1.add(i);
            let data_ptr2 = ptr2.add(i);
            let load_ptr1 = _mm256_loadu_ps(data_ptr1);
            let load_ptr2 = _mm256_loadu_ps(data_ptr2);

            let mul_result = _mm256_add_ps(load_ptr1, load_ptr2);
            i += 32;
            _mm256_storeu_ps(data_ptr1, mul_result);
        }
        while i < data1 {
            *ptr1.add(i) += *ptr2.add(i);
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
        while i + 32 <= data1 {
            let data_ptr1 = ptr1.add(i);
            let data_ptr2 = ptr2.add(i);
            let load_ptr1 = _mm256_loadu_ps(data_ptr1);
            let load_ptr2 = _mm256_loadu_ps(data_ptr2);

            let add_result = _mm256_mul_ps(load_ptr1, load_ptr2);
            i += 32;
            _mm256_storeu_ps(data_ptr1, add_result);
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
                let result_a = _mm256_set1_ps(val_a);
                let mut j = 0;
                while j + 32 <= tensor2.cols {
                    let b = tensor2.data.as_ptr().add(k * tensor2.cols + j);
                    let c = dot_result.data.as_mut_ptr().add(i * tensor2.cols + j);
                    let load_b = _mm256_loadu_ps(b);
                    let load_c = _mm256_loadu_ps(c);
                    let result_b = _mm256_mul_ps(result_a, load_b);
                    let add_result = _mm256_add_ps(result_b, load_c);    
                    _mm256_storeu_ps(c, add_result);
                    
                    j += 32;
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
    let mut dot_result = unsafe { create_tensor(tensor1.rows, tensor2.cols) };
    
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

pub unsafe fn div_tensor(mut tensor1: Tensor, tensor2: Tensor) -> Tensor{
    let data1 = tensor1.data.len();
    let data2 = tensor2.data.len();
    if data1 != data2 {
        eprintln!("配列の要素が一致していません。");
        return Tensor { rows:tensor1.rows, cols:tensor1.cols, data:tensor1.data }
    }
    if tensor1.data[0] == 0.0 && tensor2.data[0] == 0.0{
        eprintln!("要素に0が含まれています。このままでは除算できません。");
        return Tensor { rows:tensor1.rows, cols:tensor1.cols, data:tensor1.data }
    }

    let ptr1 = tensor1.data.as_mut_ptr();
    let ptr2 = tensor2.data.as_ptr();
    unsafe {
        let mut i = 0;
        while i + 32 <= data1 {
            let data_ptr1 = ptr1.add(i);
            let data_ptr2 = ptr2.add(i);
            let load_ptr1 = _mm256_loadu_ps(data_ptr1);
            let load_ptr2 = _mm256_loadu_ps(data_ptr2);

            let div_result = _mm256_div_ps(load_ptr1, load_ptr2);

            i += 32;
            _mm256_storeu_ps(data_ptr1, div_result);
        }
        while i < data1 {
            *ptr1.add(i) /= *ptr2.add(i);
            i += 1;
        }
    }
    tensor1
}

pub fn normal_div(mut tensor1: Tensor, tensor2: Tensor) -> Tensor{
    let len = tensor1.data.len();
    for i in 0..len {
        tensor1.data[i] /= tensor2.data[i];
    }
    tensor1
}

pub unsafe fn sub_tensor(mut tensor1: Tensor, tensor2: Tensor) -> Tensor{
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
        while i + 32 <= data1 {
            let data_ptr1 = ptr1.add(i);
            let data_ptr2 = ptr2.add(i);
            let load_ptr1 = _mm256_loadu_ps(data_ptr1);
            let load_ptr2 = _mm256_loadu_ps(data_ptr2);

            let sub_result = _mm256_sub_ps(load_ptr1, load_ptr2);
            i += 32;
            _mm256_storeu_ps(data_ptr1, sub_result);
        }
        while i < data1 {
            *ptr1.add(i) -= *ptr2.add(i);
            i += 1;
        }
    }
    tensor1
}

pub fn normal_sub(mut tensor1: Tensor, tensor2: Tensor) -> Tensor{
    let len = tensor1.data.len();
    for i in 0..len {
        tensor1.data[i] -= tensor2.data[i];
    }
    tensor1
}

///cast
pub unsafe  fn add_cast(mut tensor1: Tensor, tensor2: Tensor)-> Tensor {
    if tensor1.cols != tensor2.cols{
        eprintln!("列数が一致せずブロードキャストを行えません。");
        return Tensor { rows: tensor1.rows, cols: tensor1.cols, data: tensor1.data };
    }
    let ptr1 = tensor1.data.as_mut_ptr();
    let ptr2 = tensor2.data.as_ptr();
    unsafe {
        for i in 0..tensor1.rows {
            let mut j = 0;
            while j + 8 <= tensor1.cols {
                let result_ptr1 =  _mm256_loadu_ps(ptr1.add(i * tensor1.cols + j));
                let result_ptr2 = _mm256_loadu_ps(ptr2.add(j));
                let result = _mm256_add_ps(result_ptr1, result_ptr2);
                _mm256_storeu_ps(ptr1.add(i * tensor1.cols + j), result);
                j += 8;
        }
        while j  < tensor1.cols {
            *ptr1.add(i * tensor1.cols + j) += *ptr2.add(j);
            j += 1;
        }
    }
}
    tensor1
}

pub unsafe  fn mul_cast(mut tensor1: Tensor, tensor2: Tensor)-> Tensor {
    if tensor1.cols != tensor2.cols{
        eprintln!("列数が一致せずブロードキャストを行えません。");
        return Tensor { rows: tensor1.rows, cols: tensor1.cols, data: tensor1.data };
    }
    let ptr1 = tensor1.data.as_mut_ptr();
    let ptr2 = tensor2.data.as_ptr();
    unsafe {
        for i in 0..tensor1.rows {
            let mut j = 0;
            while j + 8 <= tensor1.cols {
                let result_ptr1 =  _mm256_loadu_ps(ptr1.add(i * tensor1.cols + j));
                let result_ptr2 = _mm256_loadu_ps(ptr2.add(j));
                let result = _mm256_mul_ps(result_ptr1, result_ptr2);
                _mm256_storeu_ps(ptr1.add(i * tensor1.cols + j), result);
                j += 8;
        }
        while j  < tensor1.cols {
            *ptr1.add(i * tensor1.cols + j) *= *ptr2.add(j);
            j += 1;
        }
    }
}
    tensor1
}


pub unsafe  fn sub_cast(mut tensor1: Tensor, tensor2: Tensor)-> Tensor {
    if tensor1.cols != tensor2.cols{
        eprintln!("列数が一致せずブロードキャストを行えません。");
        return Tensor { rows: tensor1.rows, cols: tensor1.cols, data: tensor1.data };
    }
    let ptr1 = tensor1.data.as_mut_ptr();
    let ptr2 = tensor2.data.as_ptr();
    unsafe {
        for i in 0..tensor1.rows {
            let mut j = 0;
            while j + 8 <= tensor1.cols {
                let result_ptr1 =  _mm256_loadu_ps(ptr1.add(i * tensor1.cols + j));
                let result_ptr2 = _mm256_loadu_ps(ptr2.add(j));
                let result = _mm256_sub_ps(result_ptr1, result_ptr2);
                _mm256_storeu_ps(ptr1.add(i * tensor1.cols + j), result);
                j += 8;
        }
        while j  < tensor1.cols {
            *ptr1.add(i * tensor1.cols + j) -= *ptr2.add(j);
            j += 1;
        }
    }
}
    tensor1
}


pub unsafe  fn div_cast(mut tensor1: Tensor, tensor2: Tensor)-> Tensor {
    if tensor1.cols != tensor2.cols{
        eprintln!("列数が一致せずブロードキャストを行えません。");
        return Tensor { rows: tensor1.rows, cols: tensor1.cols, data: tensor1.data };
    }
    let ptr1 = tensor1.data.as_mut_ptr();
    let ptr2 = tensor2.data.as_ptr();
    unsafe {
        for i in 0..tensor1.rows {
            let mut j = 0;
            while j + 8 <= tensor1.cols {
                let result_ptr1 =  _mm256_loadu_ps(ptr1.add(i * tensor1.cols + j));
                let result_ptr2 = _mm256_loadu_ps(ptr2.add(j));
                let result = _mm256_div_ps(result_ptr1, result_ptr2);
                _mm256_storeu_ps(ptr1.add(i * tensor1.cols + j), result);
                j += 8;
        }
        while j  < tensor1.cols {
            *ptr1.add(i * tensor1.cols + j) /= *ptr2.add(j);
            j += 1;
        }
    }
}
    tensor1
}