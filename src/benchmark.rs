use core::arch::x86_64::*;
use std::time::Instant;
use std::hint::black_box;
use std::fs::OpenOptions;
use std::io::Write;


#[cfg(all(target_arch = "x86_64"))]
use crate::Tensor;
use crate::{calc_tensor::{div_cast, sub_cast}, create_tensor::tensor_calc};
#[cfg(all(target_arch = "x86_64"))]
use crate::calc_tensor::{add_cast, normal_div};

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn foo1(){

   let nanos = {
        let start_time = Instant::now();
        for _ in 0..100_000_000 {
        let add_results = _mm256_add_epi64(_mm256_set1_epi64x(100), _mm256_set1_epi64x(100));
        let  res1 = add_results;
        black_box(res1);
        }
        
        black_box(start_time.elapsed().as_nanos())

    };
    let mut  writer = OpenOptions::new();
    let mut add_header = writer.write(true).create(true).append(true).open("tests/add_test.csv").expect("書き込み先のファイルがありません。");
    let mut add_file = writer.write(true).create(true).append(true).open("tests/add_test.csv").expect("書き込みが正常に行われませんでした。");
    let add_result = format!("add_f32, {:?}\n", nanos);
    if add_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        add_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    add_file.write_all(add_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");

}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub  unsafe fn tesor_add_benchmark(tensor1: Tensor, tensor2: Tensor){
     use crate::calc_tensor::add_tensor;

    let len1 = tensor1.data.len();
    let len2 = tensor2.data.len();
    if len1 != len2 {
        eprintln!("配列の要素が一致しません。");
    }
    let start_time = Instant::now();
    let result =  unsafe { 
        let result: Tensor = add_tensor(tensor1, tensor2);
        black_box(result);
        black_box(start_time.elapsed().as_nanos())
    };
    
    let mut  writer = OpenOptions::new();
    let mut add_header = writer.write(true).create(true).append(true).open("tests/tensor_add_test.csv").expect("書き込み先のファイルがありません。");
    let mut add_file = writer.write(true).create(true).append(true).open("tests/tensor_add_test.csv").expect("書き込みが正常に行われませんでした。");
    let add_result = format!("add_tensor, {}\n", result);
    if add_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        add_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    add_file.write_all(add_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");
}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn foo2(){

   let nanos = {
        let start_time = Instant::now();
        for _ in 0..100_000_000 {
        let mul_results = _mm256_mul_epi32(_mm256_set1_epi64x(100), _mm256_set1_epi64x(100));
        let  res1 = mul_results;
        black_box(res1);
        }
        
        black_box(start_time.elapsed().as_nanos())

    };
    let mut  writer = OpenOptions::new();
    let mut add_header = writer.write(true).create(true).append(true).open("tests/mul_test.csv").expect("書き込み先のファイルがありません。");
    let mut add_file = writer.write(true).create(true).append(true).open("tests/mul_test.csv").expect("書き込みが正常に行われませんでした。");
    let add_result = format!("mul_epi32, {:?}\n", nanos);
    if add_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        add_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    add_file.write_all(add_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");

}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub  unsafe fn tesor_mul_benchmark(tensor1: Tensor, tensor2: Tensor){
     use crate::calc_tensor::mul_tensor_elementwise;

    let len1 = tensor1.data.len();
    let len2 = tensor2.data.len();
    if len1 != len2 {
        eprintln!("配列の要素が一致しません。");
    }
    let start_time = Instant::now();
    let result =  unsafe { 
        let result: Tensor = mul_tensor_elementwise(tensor1, tensor2);
        black_box(result);
        black_box(start_time.elapsed().as_nanos())
    };
    
    let mut  writer = OpenOptions::new();
    let mut add_header = writer.write(true).create(true).append(true).open("tests/tensor_mul_test.csv").expect("書き込み先のファイルがありません。");
    let mut add_file = writer.write(true).create(true).append(true).open("tests/tensor_mul_test.csv").expect("書き込みが正常に行われませんでした。");
    let add_result = format!("mul_tensor, {}\n", result);
    if add_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        add_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    add_file.write_all(add_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");
}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub  unsafe fn tesor_dot_benchmark(tensor1: Tensor, tensor2: Tensor){
     use crate::calc_tensor::dot_tensor;

    let len1 = tensor1.cols;
    let len2 = tensor2.rows;
    if len1 != len2 {
        eprintln!("配列の要素が一致しません。");
    }
    let start_time = Instant::now();
    let result =  unsafe { 
        let result: Tensor = dot_tensor(tensor1, tensor2);
        black_box(result);
        black_box(start_time.elapsed().as_nanos())
    };
    
    let mut  writer = OpenOptions::new();
    let mut add_header = writer.write(true).create(true).append(true).open("tests/tensor_dot_test.csv").expect("書き込み先のファイルがありません。");
    let mut add_file = writer.write(true).create(true).append(true).open("tests/tensor_dot_test.csv").expect("書き込みが正常に行われませんでした。");
    let add_result = format!("dot_tensor, {}\n", result);
    if add_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        add_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    add_file.write_all(add_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");
}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn foo3(){

   let nanos = {
    use crate::calc_tensor::normal_dot_tensor;
    use crate::create_tensor::create_tensor;
        let start_time = Instant::now();
        unsafe {
        
        let a = create_tensor(1000, 1000);
        let b = create_tensor(1000, 1000);
        let res1 = normal_dot_tensor(&a, &b);

        black_box(res1);
        
        }
        black_box(start_time.elapsed().as_nanos())

    };
    let mut  writer = OpenOptions::new();
    let mut add_header = writer.write(true).create(true).append(true).open("tests/dot_test.csv").expect("書き込み先のファイルがありません。");
    let mut add_file = writer.write(true).create(true).append(true).open("tests/dot_test.csv").expect("書き込みが正常に行われませんでした。");
    let add_result = format!("dot_f32, {:?}\n", nanos);
    if add_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        add_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    add_file.write_all(add_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");

}


#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub  unsafe fn tesor_div_benchmark(tensor1: Tensor, tensor2: Tensor){
     use crate::calc_tensor::div_tensor;

    let len1 = tensor1.cols;
    let len2 = tensor2.rows;
    if len1 != len2 {
        eprintln!("配列の要素が一致しません。");
    }
    let start_time = Instant::now();
    let result =  unsafe { 
        let result: Tensor = div_tensor(tensor1, tensor2);
        black_box(result);
        black_box(start_time.elapsed().as_nanos())
    };
    
    let mut  writer = OpenOptions::new();
    let mut add_header = writer.write(true).create(true).append(true).open("tests/tensor_div_test.csv").expect("書き込み先のファイルがありません。");
    let mut add_file = writer.write(true).create(true).append(true).open("tests/tensor_div_test.csv").expect("書き込みが正常に行われませんでした。");
    let add_result = format!("div_tensor, {}\n", result);
    if add_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        add_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    add_file.write_all(add_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");
}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn foo4(){

   let nanos = {
        let start_time = Instant::now();
        use crate::create_tensor::create_tensor;
        unsafe {

        let a = create_tensor(1000, 1000);
        let b = create_tensor(1000, 1000);

        let res1 = normal_div(a, b);

        black_box(res1);
        
        }
        black_box(start_time.elapsed().as_nanos())

    };
    let mut  writer = OpenOptions::new();
    let mut add_header = writer.write(true).create(true).append(true).open("tests/div_test.csv").expect("書き込み先のファイルがありません。");
    let mut add_file = writer.write(true).create(true).append(true).open("tests/div_test.csv").expect("書き込みが正常に行われませんでした。");
    let add_result = format!("div_f32, {:?}\n", nanos);
    if add_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        add_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    add_file.write_all(add_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");

}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn foo5(){

   let nanos = {
        let start_time = Instant::now();
        use crate::create_tensor::create_tensor;
        use crate::calc_tensor::normal_sub;
        unsafe {

        let a = create_tensor(1000, 1000);
        let b = create_tensor(1000, 1000);

        let res1 = normal_sub(a, b);

        black_box(res1);
        
        }
        black_box(start_time.elapsed().as_nanos())

    };
    let mut  writer = OpenOptions::new();
    let mut sub_header = writer.write(true).create(true).append(true).open("tests/sub_test.csv").expect("書き込み先のファイルがありません。");
    let mut sub_file = writer.write(true).create(true).append(true).open("tests/sub_test.csv").expect("書き込みが正常に行われませんでした。");
    let sub_result = format!("sub_f32, {:?}\n", nanos);
    if sub_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        sub_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    sub_file.write_all(sub_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");

}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub  unsafe fn tesor_sub_benchmark(tensor1: Tensor, tensor2: Tensor){
     use crate::calc_tensor::sub_tensor;

    let len1 = tensor1.data.len();
    let len2 = tensor2.data.len();
    if len1 != len2 {
        eprintln!("配列の要素が一致しません。");
    }
    let start_time = Instant::now();
    let result =  unsafe { 
        let result: Tensor = sub_tensor(tensor1, tensor2);
        black_box(result);
        black_box(start_time.elapsed().as_nanos())
    };
    
    let mut  writer = OpenOptions::new();
    let mut sub_header = writer.write(true).create(true).append(true).open("tests/tensor_sub_test.csv").expect("書き込み先のファイルがありません。");
    let mut sub_file = writer.write(true).create(true).append(true).open("tests/tensor_sub_test.csv").expect("書き込みが正常に行われませんでした。");
    let sub_result = format!("sub_tensor, {}\n", result);
    if sub_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        sub_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    sub_file.write_all(sub_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");
}

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn foo6(tensor1: Tensor, tensor2: Tensor){

   let nanos = {
        let start_time = Instant::now();
        unsafe {

        let res1 = add_cast(tensor1, tensor2);

        black_box(res1);
        
        }
        black_box(start_time.elapsed().as_nanos())

    };
    let mut  writer = OpenOptions::new();
    let mut sub_header = writer.write(true).create(true).append(true).open("tests/add_cast_test.csv").expect("書き込み先のファイルがありません。");
    let mut sub_file = writer.write(true).create(true).append(true).open("tests/add_cast_test.csv").expect("書き込みが正常に行われませんでした。");
    let sub_result = format!("add_f32, {:?}\n", nanos);
    if sub_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        sub_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    sub_file.write_all(sub_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");

}

pub unsafe fn foo7(tensor1: Tensor,tensor2: Tensor){

   let nanos = {
        use crate::calc_tensor::mul_cast;
        let start_time = Instant::now();
        
        unsafe {
        
        let res1 = mul_cast(tensor1, tensor2);

        black_box(res1);
        
        }
        black_box(start_time.elapsed().as_nanos())

    };
    let mut  writer = OpenOptions::new();
    let mut sub_header = writer.write(true).create(true).append(true).open("tests/mul_cast_test.csv").expect("書き込み先のファイルがありません。");
    let mut sub_file = writer.write(true).create(true).append(true).open("tests/mul_cast_test.csv").expect("書き込みが正常に行われませんでした。");
    let sub_result = format!("mul_f32, {:?}\n", nanos);
    if sub_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        sub_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    sub_file.write_all(sub_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");

}

pub unsafe fn foo8(){

   let nanos = {
        let start_time = Instant::now();
        use crate::create_tensor::create_tensor;
        use crate::calc_tensor::sub_tensor;
        unsafe {

        let a = create_tensor(1000, 1000);
        let b = create_tensor(1000, 1000);
        let res1 = sub_tensor(a, b);
        let a = create_tensor(1000, 1000);
        let b = create_tensor(1000, 1000);
        let res2 = sub_tensor(a, b);
        let res1 = sub_cast(res1, res2);

        black_box(res1);
        
        }
        black_box(start_time.elapsed().as_nanos())

    };
    let mut  writer = OpenOptions::new();
    let mut sub_header = writer.write(true).create(true).append(true).open("tests/sub_cast_test.csv").expect("書き込み先のファイルがありません。");
    let mut sub_file = writer.write(true).create(true).append(true).open("tests/sub_cast_test.csv").expect("書き込みが正常に行われませんでした。");
    let sub_result = format!("sub_f32, {:?}\n", nanos);
    if sub_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        sub_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    sub_file.write_all(sub_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");

}


pub unsafe fn foo9(){

   let nanos = {
        let start_time = Instant::now();
        use crate::create_tensor::create_tensor;
        use crate::calc_tensor::div_tensor;
        unsafe {

        let a = create_tensor(1000, 1000);
        let b = create_tensor(1000, 1000);
        let a = tensor_calc(a, 5.0);
        let b = tensor_calc(b, 5.0);
        let res1 = div_tensor(a, b);
        let a = create_tensor(1000, 1000);
        let b = create_tensor(1000, 1000);
        let a = tensor_calc(a, 5.0);
        let b = tensor_calc(b, 5.0);
        let res2 = div_tensor(a, b);
        let res1 = div_cast(res1, res2);

        black_box(res1);
        
        }
        black_box(start_time.elapsed().as_nanos())

    };
    let mut  writer = OpenOptions::new();
    let mut sub_header = writer.write(true).create(true).append(true).open("tests/div_cast_test.csv").expect("書き込み先のファイルがありません。");
    let mut sub_file = writer.write(true).create(true).append(true).open("tests/div_cast_test.csv").expect("書き込みが正常に行われませんでした。");
    let sub_result = format!("div_f32, {:?}\n", nanos);
    if sub_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        sub_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    sub_file.write_all(sub_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");

}