use core::arch::x86_64::*;
use std::time::Instant;
use std::hint::black_box;
use std::fs::OpenOptions;
use std::io::Write;

#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn foo(){

   let nanos = {
        let start_time = Instant::now();
        for _ in 0..100_000_000 {
        let add_results = _mm256_add_epi64(_mm256_set1_epi64x(100), _mm256_set1_epi64x(100));
        let  res1 = add_results;
        black_box(res1);
        }
        
        black_box(start_time.elapsed().as_nanos());

        let start_time = Instant::now();
        for _ in 0..100_000_000 {
        let mul_results = _mm256_mul_epi32(_mm256_set1_epi64x(100), _mm256_set1_epi64x(100));
        let  res = mul_results;
        black_box(res);
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

    let mut mul_header = writer.write(true).create(true).append(true).open("tests/mul_test.csv").expect("書き込み先のファイルがありません。");
    let mut mul_file = writer.write(true).create(true).append(true).open("tests/mul_test.csv").expect("書き込みが正常に行われませんでした。");
    let mul_result = format!("mul_epi32, {:?}\n", nanos);
    if mul_header.metadata().expect("バイト数の読み込みに失敗しました。").len() == 0{
        mul_header.write_all(b"method, result\n").expect("書き込みに失敗しました。");
    }
    mul_file.write_all(mul_result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");

}
fn main() {
    println!("test start");
    unsafe {
        if is_x86_feature_detected!("avx2"){
            foo();
        }else {
            eprintln!("avx2がCPUによってサポートされていません。");
            return;           
        }

    }
}
