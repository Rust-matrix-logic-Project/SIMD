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
        _mm256_add_epi64(_mm256_set1_epi64x(100), _mm256_set1_epi64x(100));
        }
        black_box(start_time.elapsed().as_nanos())
    };
    let mut  writer = OpenOptions::new();
    let mut file = writer.write(true).create(true).append(true).open("test.csv").expect("書き込みが正常に行われませんでした。");
    let result = format!("add_epi64: {:?}ナノ秒\n", nanos);
    file.write_all(result.as_bytes()).expect("ファイルへの書き込みに失敗しました。");


}
fn main() {
    println!("Hello, world!");
    unsafe {
        if is_x86_feature_detected!("avx2"){
            foo();
        }else {
            eprintln!("avx2がCPUによってサポートされていません。");
            return;           
        }

    }
}
