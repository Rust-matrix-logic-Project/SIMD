use core::arch::x86_64::*;
use std::time::Instant;
use std::hint::black_box;
use std::fs::OpenOptions;
use std::io::Write;


#[cfg(all(target_arch = "x86_64"))]
use crate::Tensor;

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
    let add_result = format!("add_epi64, {:?}\n", nanos);
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
        let add_results = _mm256_mul_epi32(_mm256_set1_epi64x(100), _mm256_set1_epi64x(100));
        let  res1 = add_results;
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